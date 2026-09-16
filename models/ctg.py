"""Rule-based CTG analysis following the FIGO-oriented implementation used in this project.

The module contains signal preprocessing, baseline estimation, deceleration and
variability analysis, uterine-contraction detection, tachysystole assessment,
and the final rule-based CTG classification.

FHR and UC signals are stored as NumPy arrays. Pandas is used locally for
rolling-window operations where appropriate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import commentjson
from scipy.ndimage import label
from matplotlib import pyplot as plt
from typing import Union, Dict

from config.config_methods import AppConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.df_ctg import DF_CTG


class CTG:

    def __init__(
        self,
        fhr: np.ndarray,  # Fetal Heart Rate signal (list or array)
        uc: np.ndarray,  # Uterine Contraction signal (list or array)
        clinical_data: pd.Series = None,
        config_file: str = None,
        ph: float = None,  # Optional pH value
        id: int = None,  # Optional ID for the record or patient
        group: DF_CTG = None,
    ):

        self.group = group

        if group is None and config_file is None:
            print(
                "ERROR: If a group is not provided, a config_file path must be provided."
            )

        if group is not None:
            self.config = group.config

        else:
            print("Reading config file from {config_file}...")
            try:
                with open("config.jsonc", "r", encoding="utf-8") as file:
                    config = AppConfig(commentjson.load(file))
                    self.config = config

            except Exception as e:
                print("ERROR! It was not possible to read the config file")
                print(f"Error details: {e}")
                self.config = None

        freq = self.config.freq

        if len(fhr) != len(uc):
            raise ValueError("ERROR: The size of the FHR and UC must be equal.")

        self._time = np.arange(0, len(fhr) / freq, 1 / freq)
        self.id = id
        self.fhr = np.array(fhr)
        self.uc = np.array(uc)
        self.clinical_data = clinical_data
        self.ph = ph
        self.frequency = freq

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------

    def preprocess_signal(self, prep_type=None):

        freq = self.config.freq

        if prep_type == None:
            prep_type = self.config.preprocessing.prep_type

        cut_time = self.config.preprocessing.cut_time
        rm_tail_nan = self.config.preprocessing.rm_tail_nan

        min_value_fhr = self.config.limit.min_value_fhr
        max_value_fhr = self.config.limit.max_value_fhr
        min_value_uc = self.config.limit.min_value_uc
        max_value_uc = self.config.limit.max_value_uc

        if np.isnan(self.fhr).all() or np.isnan(self.uc).all():
            print(
                f"{self.id}: FHR or UC signals are empty, preprocessing cannot be applied."
            )

            return 1

        fhr_aux = self.fhr.copy()
        uc_aux = self.uc.copy()

        fhr_aux[(fhr_aux <= min_value_fhr) | (fhr_aux > max_value_fhr)] = np.nan
        uc_aux[(uc_aux < min_value_uc) | (uc_aux > max_value_uc)] = np.nan

        if rm_tail_nan:
            fhr_aux, uc_aux = self._drop_tail_nans(fhr_aux, uc_aux)

        num_cut_data = freq * cut_time * 60

        fhr_aux = fhr_aux[-num_cut_data:]
        uc_aux = uc_aux[-num_cut_data:]

        self._time = np.arange(0, (len(fhr_aux) / freq), (1 / freq))

        match prep_type:
            case "FIGO":
                self._preprocess_type = "FIGO"

                fhr_aux = self._replace_gaps(fhr_aux)
                uc_aux = self._replace_gaps(uc_aux)

        self._time = np.arange(0, len(fhr_aux) / freq, 1 / freq)

        self.fhr = fhr_aux
        self.uc = uc_aux

        return 0

    # -------------------------------------------------------------------------
    # PUBLIC RULE-BASED ANALYSIS API
    # -------------------------------------------------------------------------

    def apply_rules(
        self,
        rules_type: str,
        center: bool = False,
        get_dicc: bool = False,
        # -- preprocess param --
        preprocess: bool = False,
        prep_cut_time: int = 60,
        prep_max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
        # -- baseline param --
        window_time_baseline: int = 10,
        baseline_graph: bool = False,
        # -- dec param --
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        time_in_rep_dec: int = 30,
        correlation_threshold: float = -0.4,
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        time_in_prolonged_dec: int = 3,
        time_in_rep_dec_for_red_var: int = 20,
        time_patholog_prolonged_dec: int = 5,
        deceleration_graph: bool = False,
        # -- variability param --
        window_var: int = 1,
        variability_graph: bool = False,
        incr_var_min_time: int = 30,
        incr_var_bandwidth: int = 25,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        # -- conclusion graph --
        conclusion_graph: bool = True,
    ) -> None:
        """
        Apply fetal monitoring interpretation rules to the CTG data.

        Currently, only the FIGO ruleset is supported. This method can optionally
        preprocess the signal, calculate baseline, analyze decelerations, and assess
        variability. Each analysis section includes specific tunable parameters.

        Args:
            center (bool): Whether to center the signal.
            get_dicc (bool): If True, return a dictionary of results.

            rules_type (str): The ruleset to apply. Only 'FIGO' is currently supported.
            preprocess (bool): Whether to preprocess the signal before applying rules.
            prep_cut_time (int): Duration in seconds to cut from the beginning for preprocessing.
            prep_max_size_gaps (int): Maximum gap size (in seconds) to interpolate during preprocessing.
            rm_tail_nan (bool): Whether to remove trailing NaNs before cutting the signal.

            # Baseline parameters
            window_time_baseline (int): Baseline estimation window, in minutes.
            baseline_graph (bool): Whether to generate a graph for the baseline.

            # Deceleration parameters
            amplitude_dec_cond (int): Minimum amplitude (bpm) to consider a deceleration.
            time_in_deceleration (int): Minimum duration (in seconds) to consider a deceleration.
            time_in_rep_dec (int): Time window for repeated decelerations.
            correlation_threshold (float): Threshold for correlation with contractions.
            time_in_late_dec_cond (int): Time threshold for late decelerations.
            contraction_time_shift (int): Shift in seconds for aligning contractions.
            contraction_value_threshold (int): Value threshold for contraction detection.
            contraction_duration_threshold (int): Minimum duration of contractions (in seconds).
            contraction_baseline_duration (int): Baseline duration used in contraction analysis.
            time_in_prolonged_dec (int): Duration to define prolonged deceleration.
            time_in_rep_dec_for_red_var (int): Duration to evaluate variability within repeated decelerations.
            time_patholog_prolonged_dec (int): Time threshold for pathological prolonged deceleration.
            deceleration_graph (bool): Whether to generate a deceleration graph.

            # Variability parameters
            variability_graph (bool): Whether to plot variability.
            incr_var_min_time (int): Minimum duration to identify increased variability.
            incr_var_bandwidth (int): Bandwidth threshold for increased variability.
            red_var_duration_baseline (int): Duration to evaluate reduced variability during baseline.
            red_var_duration_deceleration (int): Duration to evaluate reduced variability during deceleration.
            red_var_bandwith (int): Bandwidth threshold for reduced variability.

        Raises:
            ValueError: If `rules_type` is not supported.
        """

        # Validate the requested rule set.
        correct_rules_type = ["FIGO"]
        if rules_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # Dispatch to the FIGO rule-based analysis.
        if rules_type == "FIGO":
            return self.apply_rules_FIGO(
                center=center,
                get_dicc=get_dicc,
                # -- preprocess param --
                preprocess=preprocess,
                prep_cut_time=prep_cut_time,
                prep_max_size_gaps=prep_max_size_gaps,
                rm_tail_nan=rm_tail_nan,
                # -- baseline param --
                window_time_baseline=window_time_baseline,
                baseline_graph=baseline_graph,
                # -- dec param --
                amplitude_dec_cond=amplitude_dec_cond,
                time_in_deceleration=time_in_deceleration,
                time_in_rep_dec=time_in_rep_dec,
                correlation_threshold=correlation_threshold,
                time_in_late_dec_cond=time_in_late_dec_cond,
                contraction_time_shift=contraction_time_shift,
                contraction_value_threshold=contraction_value_threshold,
                contraction_duration_threshold=contraction_duration_threshold,
                contraction_baseline_duration=contraction_baseline_duration,
                time_in_prolonged_dec=time_in_prolonged_dec,
                time_in_rep_dec_for_red_var=time_in_rep_dec_for_red_var,
                time_patholog_prolonged_dec=time_patholog_prolonged_dec,
                deceleration_graph=deceleration_graph,
                # -- variability param --
                window_var=window_var,
                variability_graph=variability_graph,
                incr_var_min_time=incr_var_min_time,
                incr_var_bandwidth=incr_var_bandwidth,
                red_var_duration_baseline=red_var_duration_baseline,
                red_var_duration_deceleration=red_var_duration_deceleration,
                red_var_bandwith=red_var_bandwith,
                # -- conclusion graph --
                conclusion_graph=conclusion_graph,
            )

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------

    def get_contractions_graph(
        self,
        # -- contraction info --
        contraction_time_shift=0,
        contraction_value_threshold=10,
        contraction_duration_threshold=20,
        contraction_baseline_duration=120,
        contraction_center=True,
    ):

        t = self._time / 60
        UC = self.uc
        is_contraction = self._get_contraction_condition(
            time_shift=contraction_time_shift,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=contraction_center,
        )

        plt.figure(figsize=(18, 6))
        plt.plot(t, UC, label="UC", color="black")

        is_contraction = np.where(
            (is_contraction == "NaN") | (is_contraction == "None"),
            False,
            is_contraction,
        )
        in_condition = False
        start = None

        for i in range(len(is_contraction)):
            if is_contraction[i] and not in_condition:
                in_condition = True
                start = t[i]
            elif not is_contraction[i] and in_condition:
                in_condition = False
                end = t[i]
                plt.axvspan(start, end, color="#2849C1", alpha=0.4)

        if in_condition:
            plt.axvspan(start, np.asarray(t)[-1], color="blue", alpha=0.5)

        plt.tick_params(axis="both", labelsize=20)

        plt.xlabel("Time (min)", fontsize=20)
        plt.ylabel("Pressure (mmHg)", fontsize=20)
        plt.grid(False)
        plt.tight_layout()

        plt.savefig("contractions_graph.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    def plot_ctg(self, desplazamiento_fhr=0):
        """
        Plot the fetal heart rate (FHR) and uterine contraction (UC) signals in two time segments.

        This method splits the signals into two halves and displays them using four vertically
        stacked subplots. Each half of the recording is shown with its FHR and UC signals
        aligned for visual inspection. This format facilitates clinical-style interpretation
        of long CTG recordings.

        The top two plots show the first half of the recording (FHR and UC),
        and the bottom two plots show the second half.

        Args:
            None

        Returns:
            None: Displays a matplotlib figure with 4 subplots (FHR and UC for two halves).
        """

        pd_fhr = pd.DataFrame(self.fhr, self._time)
        pd_uc = pd.DataFrame(self.uc, self._time)

        # Split signal length in half
        mid = len(self.fhr) // 2

        # Divide the FHR and UC signals into first and second halves
        fhr1, fhr2 = pd_fhr.iloc[:mid], pd_fhr.iloc[mid:]
        uc1, uc2 = pd_uc.iloc[:mid], pd_uc.iloc[mid:]

        # Create 4 vertically stacked subplots (FHR and UC for both halves)
        _, axs = plt.subplots(4, 1, figsize=(25, 12), sharex=False)

        # Plot FHR (first half)
        ax1 = axs[0]
        ax1.plot(fhr1.index / 60, fhr1.values, color="black", label="FHR")
        ax1.set_ylabel("FHR", fontsize=14)
        ax1.grid(True)
        ax1.tick_params(axis="both", labelsize=13)
        ax1.set_ylim(40, 180)

        # Plot UC (first half)
        ax2 = axs[1]
        ax2.plot(uc1.index / 60, uc1.values, color="blue", alpha=0.6, label="UC")
        ax2.set_ylabel("UC", fontsize=14)
        ax2.grid(True)
        ax2.tick_params(axis="both", labelsize=13)
        ax2.set_ylim(0, 100)

        # Plot FHR (second half)
        ax3 = axs[2]
        ax3.plot(fhr2.index / 60, fhr2.values, color="black", label="FHR")
        ax3.set_ylabel("FHR", fontsize=14)
        ax3.grid(True)
        ax3.tick_params(axis="both", labelsize=13)
        ax3.set_ylim(40, 180)

        # Plot UC (second half)
        ax4 = axs[3]
        ax4.plot(uc2.index / 60, uc2.values, color="blue", alpha=0.6, label="UC")
        ax4.set_xlabel("Time (min)", fontsize=14)
        ax4.set_ylabel("UC", fontsize=14)
        ax4.grid(True)
        ax4.tick_params(axis="both", labelsize=13)
        ax4.set_ylim(0, 100)

        # Add a global title with the CTG identifier
        title = "CTG: " + str(self.id)
        plt.suptitle(title, fontsize=18)

        # Adjust layout to leave space for the main title
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        plt.savefig("ctg_preprocess.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    def plot_fhr(self, title_eje_x=False, save_name=None):

        plt.figure(figsize=(24, 6))
        plt.plot(
            self._time / 60,
            self.fhr,
            color="#2849C1",
            linewidth=1.5,
            alpha=0.8,
        )

        plt.ylim(0, 200)
        plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

        if title_eje_x:
            plt.xlabel("Time (min)", fontsize=25)
        else:
            plt.xticks([])

        plt.tick_params(axis="both", labelsize=25)

        plt.savefig(save_name, format="pdf", bbox_inches="tight")

    # -------------------------------------------------------------------------
    # FIGO-BASED RULES
    # -------------------------------------------------------------------------

    def apply_rules_FIGO(
        self,
        center: bool = False,
        get_dicc: bool = False,
        # -- preprocess param --
        preprocess: bool = False,
        prep_cut_time: int = 60,
        prep_max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
        # -- baseline param --
        window_time_baseline: int = 10,
        baseline_graph: bool = False,
        # -- dec param --
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        time_in_rep_dec: int = 30,
        correlation_threshold: float = -0.4,
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        time_in_prolonged_dec: int = 3,
        time_in_rep_dec_for_red_var: int = 20,
        time_patholog_prolonged_dec: int = 5,
        deceleration_graph: bool = False,
        # -- variability param --
        window_var: int = 1,
        variability_graph: bool = False,
        incr_var_min_time: int = 30,
        incr_var_bandwidth: int = 25,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        # -- conclusion graph --
        conclusion_graph: bool = True,
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Apply the project's FIGO-based CTG interpretation rules, integrating baseline,
        decelerations, and variability into a final classification.

        Args:
            center (bool): Whether to center rolling windows in calculations.
            get_dicc (bool): If True, return a dictionary with intermediate results.
            preprocess (bool): Whether to apply preprocessing before analysis.
            prep_cut_time (int): Time (in seconds) to cut from start/end during preprocessing.
            prep_max_size_gaps (int): Maximum allowed gap size during preprocessing.
            rm_tail_nan (bool): Whether to remove trailing NaNs before cutting the signal.
            window_time_baseline (int): Window size (minutes) to compute baseline.
            baseline_graph (bool): Whether to plot baseline graph.
            amplitude_dec_cond (int): Amplitude threshold for deceleration detection.
            time_in_deceleration (int): Minimum deceleration duration (seconds).
            time_in_rep_dec (int): Time window for repetitive decelerations (minutes).
            correlation_threshold (float): Correlation threshold for repetitive decels.
            time_in_late_dec_cond (int): Time window to identify late decelerations (seconds).
            contraction_time_shift (int): Time shift for contraction detection (seconds).
            contraction_value_threshold (int): Value threshold for contraction detection.
            contraction_duration_threshold (int): Duration threshold for contraction detection.
            contraction_baseline_duration (int): Baseline duration for contractions.
            time_in_prolonged_dec (int): Time threshold for prolonged decelerations (minutes).
            time_in_rep_dec_for_red_var (int): Time window for repetitive decels with reduced variability.
            time_patholog_prolonged_dec (int): Time threshold for pathological prolonged decelerations (minutes).
            deceleration_graph (bool): Whether to plot deceleration graph.
            window_var (int): Window size (minutes) for variability calculation.
            variability_graph (bool): Whether to plot variability graph.
            incr_var_min_time (int): Minimum time to detect increased variability (minutes).
            incr_var_bandwidth (int): Bandwidth threshold for increased variability detection.
            red_var_duration_baseline (int): Duration for baseline reduced variability (minutes).
            red_var_duration_deceleration (int): Duration for deceleration reduced variability (minutes).
            red_var_bandwith (int): Bandwidth threshold for reduced variability.
            conclusion_graph (bool): Whether to plot final conclusion graph.

        Returns:
            np.ndarray: Final classification labels (NORMAL, SUSPICIOUS, PATHOLOGICAL).
            dict (optional): Dictionary with intermediate results if `get_dicc` is True.
        """

        # Optionally preprocess FHR and UC before applying the rules.
        if preprocess:

            # Report previous preprocessing to avoid applying it unintentionally twice.
            if hasattr(self, "_preprocess_type"):
                print(
                    "The ctg has been previously preprocessed following: ",
                    self._preprocess_type,
                )

            self.preprocess_rules_figo(
                cut_time=prep_cut_time,
                max_sec_gaps=prep_max_size_gaps,
                rm_tail_nan=rm_tail_nan,
            )

        # Evaluate the three CTG components used in the final classification.
        base_labels = self._get_baseline_labels(
            window_min_size=window_time_baseline, graph=baseline_graph, center=center
        )
        decelerations_labels = self._get_decelerations_labels(
            center=center,
            window_time_baseline=window_time_baseline,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            time_in_rep_dec=time_in_rep_dec,
            correlation_threshold=correlation_threshold,
            time_in_late_dec_cond=time_in_late_dec_cond,
            contraction_time_shift=contraction_time_shift,
            contraction_value_threshold=contraction_value_threshold,
            contraction_duration_threshold=contraction_duration_threshold,
            contraction_baseline_duration=contraction_baseline_duration,
            time_in_prolonged_dec=time_in_prolonged_dec,
            time_in_rep_dec_for_red_var=time_in_rep_dec_for_red_var,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
            time_patholog_prolonged_dec=time_patholog_prolonged_dec,
            graph=deceleration_graph,
        )

        variability_labels = self._get_variability_labels(
            graph=variability_graph,
            center=center,
            window_var=window_var,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            incr_var_min_time=incr_var_min_time,
            incr_var_bandwidth=incr_var_bandwidth,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
        )

        # Combine component labels into the final CTG classification.
        conclusion_labels = self._get_conclusion_labels(graph=conclusion_graph)

        # Return intermediate component labels when requested.
        if get_dicc:
            tachysystole = self._get_tachysystole_condition(center=center)
            return {
                "baseline": base_labels,
                "deceleration": decelerations_labels,
                "variability": variability_labels,
                "tachysystole": pd.Series(tachysystole),
                "conclusion": conclusion_labels,
            }

        return conclusion_labels

    def _get_baseline_labels(
        self,
        window_min_size: int = 10,
        graph: bool = False,
        center: bool = False,
    ) -> pd.Series:
        """
        Estimate and classify the FHR baseline using the implemented FIGO-based thresholds.

        The baseline is obtained with the event-exclusion procedure implemented in
        `_get_baseline()`. Each valid estimate is then assigned to the corresponding
        baseline category. Optionally, the resulting classification can be plotted
        over the FHR signal.

        Args:
            window_min_size (int): Baseline estimation window, in minutes.
            graph (bool): Whether to display the baseline and classification graphs.
            center (bool): Whether the rolling mean should be centered.

        Returns:
            pd.Series: A pandas Series containing the classification labels for each time point.
        """

        # Initialize labels as missing until a valid baseline estimate is available.
        labels = np.nan * np.ones_like(self.fhr)

        # Estimate the FHR baseline.
        baseline = self._get_baseline(window_min_size=window_min_size, center=center)

        # Assign the baseline category using the configured FIGO thresholds.
        labels[(baseline >= 110.0) & (baseline <= 160.0)] = self.config.labels.normal
        labels[(baseline < 110.0) | (baseline > 160.0)] = self.config.labels.suspicious
        labels[(baseline < 100)] = self.config.labels.pathological

        # Store the baseline and labels for subsequent rule evaluation.
        self._baseline = baseline
        self._baseline_labels = labels

        # Plot the result when requested.
        if graph:
            self._baseline_graph(show=1, grid=None)

        return pd.Series(labels)

    def _get_decelerations_labels(
        self,
        center: bool = False,
        window_time_baseline: int = 10,
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        time_in_rep_dec: int = 30,
        correlation_threshold: float = -0.4,
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        time_in_prolonged_dec: int = 3,
        time_in_rep_dec_for_red_var: int = 20,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        time_patholog_prolonged_dec: int = 5,
        graph: bool = False,
    ) -> pd.Series:
        """
        Classify deceleration-related CTG patterns.

        Basic detected decelerations are initially classified as suspicious.
        Pathological labels are subsequently assigned for:

        - repetitive late or prolonged decelerations over 30 min,
        - repetitive late or prolonged decelerations over 20 min
        when associated with reduced variability,
        - a single prolonged deceleration lasting >5 min.
        """

        fhr = np.asarray(
            self.fhr,
            dtype=float,
        )

        # Invalid FHR

        if np.isnan(fhr).all():

            self._decelerations_labels = pd.Series(
                np.nan,
                index=np.arange(len(fhr)),
                dtype=float,
            )

            return self._decelerations_labels

        # Baseline

        if not hasattr(
            self,
            "_baseline",
        ):

            self._baseline = self._get_baseline(
                window_min_size=window_time_baseline,
                center=center,
            )

        # Basic deceleration detection

        dec_cond = self._get_deceleration_condition(
            amplitude_deceleration=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
        )

        dec_label = pd.Series(
            np.nan,
            index=np.arange(len(fhr)),
            dtype=float,
        )

        dec_valid = ~pd.isna(dec_cond)

        # No detected deceleration
        dec_label[dec_valid & (dec_cond == False)] = self.config.labels.normal

        # Basic deceleration detected
        dec_label[dec_valid & (dec_cond == True)] = self.config.labels.suspicious

        # LATE DECELERATIONS

        late = self._get_late_deceleration_condition(
            dec_cond=dec_cond,
            center=center,
            time_in_late_dec_cond=time_in_late_dec_cond,
            contraction_time_shift=contraction_time_shift,
            contraction_value_threshold=contraction_value_threshold,
            contraction_duration_threshold=contraction_duration_threshold,
            contraction_baseline_duration=contraction_baseline_duration,
        )

        late_bool = np.array(
            [x is True or x == 1 for x in late],
            dtype=bool,
        )

        # PROLONGED >3 MIN

        prolonged_3 = self._get_prolonged_deceleration_condition(
            dec_cond,
            time_in_prolonged_dec=time_in_prolonged_dec,
            center=center,
        )

        prolonged_3_bool = np.array(
            [x is True or x == 1 for x in prolonged_3],
            dtype=bool,
        )

        # REPETITIVE >50% OVER 30 MIN

        rep_30 = self._get_repetitive_deceleration_condition(
            time_in_rep_dec=time_in_rep_dec,
            contraction_value_threshold=contraction_value_threshold,
            contraction_duration_threshold=contraction_duration_threshold,
            contraction_baseline_duration=contraction_baseline_duration,
            center=center,
            # retained for backwards compatibility
            correlation_threshold=correlation_threshold,
        )

        rep_30_bool = np.array(
            [x is True or x == 1 for x in rep_30],
            dtype=bool,
        )

        pathological_30 = rep_30_bool & (late_bool | prolonged_3_bool)

        dec_label[pathological_30] = self.config.labels.pathological

        # REPETITIVE >50% OVER 20 MIN + REDUCED VARIABILITY

        rep_20 = self._get_repetitive_deceleration_condition(
            time_in_rep_dec=time_in_rep_dec_for_red_var,
            contraction_value_threshold=contraction_value_threshold,
            contraction_duration_threshold=contraction_duration_threshold,
            contraction_baseline_duration=contraction_baseline_duration,
            center=center,
            correlation_threshold=correlation_threshold,
        )

        rep_20_bool = np.array(
            [x is True or x == 1 for x in rep_20],
            dtype=bool,
        )

        red_var = self._get_reduced_variability_condition(
            center=center,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
        )

        red_var_bool = np.array(
            [x is True or x == 1 for x in red_var],
            dtype=bool,
        )

        pathological_20 = rep_20_bool & red_var_bool & (late_bool | prolonged_3_bool)

        dec_label[pathological_20] = self.config.labels.pathological

        # SINGLE PROLONGED DECELERATION >5 MIN

        prolonged_5 = self._get_prolonged_deceleration_condition(
            dec_cond,
            time_in_prolonged_dec=time_patholog_prolonged_dec,
            center=center,
        )

        prolonged_5_bool = np.array(
            [x is True or x == 1 for x in prolonged_5],
            dtype=bool,
        )

        dec_label[prolonged_5_bool] = self.config.labels.pathological

        # Store result

        self._decelerations_labels = dec_label

        if graph:
            self._decelerations_graph()

        return dec_label

    def _get_variability_labels(
        self,
        graph: bool = False,
        center: bool = False,
        window_var: int = 1,
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        incr_var_min_time: int = 30,
        incr_var_bandwidth: int = 25,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
    ) -> pd.Series:
        fhr = np.asarray(self.fhr, dtype=float)
        freq = self.frequency
        if not hasattr(self, "_baseline"):
            self._baseline = self._get_baseline(window_min_size=10, center=center)

        event_mask = self._get_event_mask(
            fhr,
            np.asarray(self._baseline, dtype=float),
            amplitude=15,
            duration_sec=15,
        )
        clean = fhr.copy()
        clean[event_mask] = np.nan
        s = pd.Series(clean)
        w = max(1, int(freq * 60 * window_var))
        mp = max(1, int(self.config.correlation.not_nans_threshold_porc * w))
        bandwidth = (
            s.rolling(w, min_periods=mp, center=center).max()
            - s.rolling(w, min_periods=mp, center=center).min()
        )

        var_labels = np.full(len(fhr), np.nan, dtype=float)
        bw = bandwidth.to_numpy()
        valid = np.isfinite(bw)
        var_labels[valid & (bw >= 5) & (bw <= 25)] = self.config.labels.normal
        var_labels[valid & ((bw < 5) | (bw > 25))] = self.config.labels.suspicious

        red_var = self._get_reduced_variability_condition(
            center=center,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
        )
        inc_var = self._get_increased_variability_condition(
            center=center,
            window_var=window_var,
            min_time=incr_var_min_time,
            bandwidth_min=incr_var_bandwidth,
        )
        red_bool = np.array([x is True or x == 1 for x in red_var], dtype=bool)
        inc_bool = np.array([x is True or x == 1 for x in inc_var], dtype=bool)
        var_labels[red_bool | inc_bool] = self.config.labels.pathological

        self._variability_labels = var_labels
        if graph:
            self._variability_graph()
        return pd.Series(var_labels)

    def _get_conclusion_labels(self, graph: bool = True) -> pd.Series:
        """
        Generate a final diagnostic conclusion based on the combined evaluation of:
        - Baseline heart rate
        - Decelerations
        - Variability

        The method follows these clinical rules:
        - If any component is PATHOLOGICAL → Final label is PATHOLOGICAL
        - If all components are NORMAL → Final label is NORMAL
        - If at least one component is SUSPICIOUS and none are PATHOLOGICAL → Final label is SUSPICIOUS

        Args:
            graph (bool): Whether to display a summary graph of the conclusion.

        Returns:
            np.ndarray: Final classification labels (NORMAL, SUSPICIOUS, PATHOLOGICAL) per time point.
        """

        # Extract component-wise labels
        FHR = self.fhr
        baseline_labels = self._baseline_labels
        decelerations_labels = self._decelerations_labels
        variability_labels = self._variability_labels

        # Initialize the final conclusion array with NaNs
        conclusion_labels = np.nan * np.ones_like(FHR)

        # Assign "SUSPICIOUS" if any of the components is suspicious
        conclusion_labels[
            (baseline_labels == self.config.labels.suspicious)
            | (decelerations_labels == self.config.labels.suspicious)
            | (variability_labels == self.config.labels.suspicious)
        ] = self.config.labels.suspicious

        # Assign "NORMAL" only if all three components are normal
        conclusion_labels[
            (baseline_labels == self.config.labels.normal)
            & (decelerations_labels == self.config.labels.normal)
            & (variability_labels == self.config.labels.normal)
        ] = self.config.labels.normal

        # Assign "PATHOLOGICAL" if any of the components is pathological
        conclusion_labels[
            (baseline_labels == self.config.labels.pathological)
            | (decelerations_labels == self.config.labels.pathological)
            | (variability_labels == self.config.labels.pathological)
        ] = self.config.labels.pathological

        # Store result in instance variable
        self._conclusion_labels = conclusion_labels

        # Optionally display graph
        if graph:
            self._conclusion_graph()

        return pd.Series(conclusion_labels)

    # -------------------------------------------------------------------------
    # FIGO ANALYSIS PLOTS
    # -------------------------------------------------------------------------

    def _baseline_graph(self, show: int = 0, grid=None) -> None:
        """Plot the FHR signal with the baseline FIGO classification."""

        t = self._time / 60
        labels = np.asarray(self._baseline_labels)

        if grid is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
        else:
            ax = plt.subplot(grid)
            fig = ax.figure

        # FHR signal
        ax.plot(
            t,
            self.fhr,
            linewidth=1.1,
            color="black",
            zorder=3,
        )

        # FIGO classification regions
        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.normal),
            color="#388E3C",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.suspicious),
            color="#F58220",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.pathological),
            color="#E04A3F",
            alpha=0.6,
            linewidth=0,
        )

        # Formatting
        ax.set_title("Baseline", fontsize=14, pad=8)
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("FHR (bpm)", fontsize=11)

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

        ax.set_ylim(50, 200)
        ax.set_xlim(t.min(), t.max())

        ax.set_xticks(np.linspace(t.min(), t.max(), 6))

        ax.grid(False)

        # Cleaner appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        if grid is None:
            fig.savefig(
                "fig/baseline_graph.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        if show == 0:
            plt.show()

    def _decelerations_graph(self, show: int = 0, grid=None) -> None:
        """Plot the FHR signal with the deceleration FIGO classification."""

        t = self._time / 60
        labels = np.asarray(self._decelerations_labels)

        if grid is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
        else:
            ax = plt.subplot(grid)
            fig = ax.figure

        # FHR signal
        ax.plot(
            t,
            self.fhr,
            linewidth=1.1,
            color="black",
            zorder=3,
        )

        # FIGO classification regions
        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.normal),
            color="#388E3C",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.suspicious),
            color="#F58220",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.pathological),
            color="#E04A3F",
            alpha=0.6,
            linewidth=0,
        )

        # Formatting
        ax.set_title("Decelerations", fontsize=14, pad=8)
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("FHR (bpm)", fontsize=11)

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

        ax.set_ylim(50, 200)
        ax.set_xlim(t.min(), t.max())

        ax.set_xticks(np.linspace(t.min(), t.max(), 6))

        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        if grid is None:
            fig.savefig(
                "fig/deceleration_graph.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        if show == 0:
            plt.show()

    def _variability_graph(self, show: int = 0, grid=None) -> None:
        """Plot the FHR signal with the variability FIGO classification."""

        t = self._time / 60
        labels = np.asarray(self._variability_labels)

        if grid is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
        else:
            ax = plt.subplot(grid)
            fig = ax.figure

        # FHR signal
        ax.plot(
            t,
            self.fhr,
            linewidth=1.1,
            color="black",
            zorder=3,
        )

        # FIGO classification regions
        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.normal),
            color="#388E3C",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.suspicious),
            color="#F58220",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.pathological),
            color="#E04A3F",
            alpha=0.6,
            linewidth=0,
        )

        # Formatting
        ax.set_title("Variability", fontsize=14, pad=8)
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("FHR (bpm)", fontsize=11)

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

        ax.set_ylim(50, 200)
        ax.set_xlim(t.min(), t.max())

        ax.set_xticks(np.linspace(t.min(), t.max(), 6))

        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        if grid is None:
            fig.savefig(
                "fig/variability_graph.pdf",
                format="pdf",
                bbox_inches="tight",
            )

        if show == 0:
            plt.show()

    def _conclusion_graph(self) -> None:
        """Plot the final FIGO-based CTG classification."""

        t = self._time / 60
        labels = np.asarray(self._conclusion_labels)

        fig, ax = plt.subplots(figsize=(10, 3.5))

        # FHR signal
        ax.plot(
            t,
            self.fhr,
            linewidth=1.1,
            color="black",
            zorder=3,
        )

        # Final classification regions
        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.normal),
            color="#388E3C",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.suspicious),
            color="#F58220",
            alpha=0.6,
            linewidth=0,
        )

        ax.fill_between(
            t,
            50,
            200,
            where=(labels == self.config.labels.pathological),
            color="#E04A3F",
            alpha=0.6,
            linewidth=0,
        )

        # Formatting
        ax.set_title("Final classification", fontsize=14, pad=8)
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("FHR (bpm)", fontsize=11)

        ax.tick_params(
            axis="both",
            labelsize=10,
        )

        ax.set_ylim(50, 200)
        ax.set_xlim(t.min(), t.max())

        ax.set_xticks(np.linspace(t.min(), t.max(), 6))

        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        fig.savefig(
            "fig/conclusion_graph.pdf",
            format="pdf",
            bbox_inches="tight",
        )

        plt.show()

    # -------------------------------------------------------------------------
    # AUXILIARY METHODS
    # -------------------------------------------------------------------------
    @staticmethod
    def shift_numpy(arr: np.ndarray, lag: int, fill_value=np.nan) -> np.ndarray:
        if lag == 0:
            return arr.copy()

        result = np.empty_like(arr, dtype=float)
        if lag > 0:
            if lag >= len(arr):
                result.fill(fill_value)
            else:
                result[:lag] = fill_value
                result[lag:] = arr[:-lag]
        else:
            abs_lag = abs(lag)
            if abs_lag >= len(arr):
                result.fill(fill_value)
            else:
                result[-abs_lag:] = fill_value
                result[:-abs_lag] = arr[abs_lag:]
        return result

    def _get_baseline(
        self,
        window_min_size: float,
        center: bool = False,
        exclude_events: bool = True,
    ) -> pd.Series:
        """
        Estimate the FHR baseline over a rolling window.

        A robust preliminary baseline is first obtained using a rolling median.
        Sustained FHR excursions are identified relative to this estimate and
        excluded from the signal. The final baseline is then computed as the
        rolling mean of the remaining samples. If too few valid samples remain,
        the preliminary median estimate is retained.
        """
        fhr = np.asarray(self.fhr, dtype=float)
        window_size = max(1, int(window_min_size * 60 * self.frequency))
        min_valid = max(
            1,
            int(window_size * self.config.correlation.not_nans_threshold_porc),
        )

        fhr_series = pd.Series(fhr)
        baseline_initial = fhr_series.rolling(
            window=window_size,
            min_periods=min_valid,
            center=center,
        ).median()

        if not exclude_events:
            return baseline_initial

        event_mask = self._get_event_mask(
            fhr=fhr,
            baseline=baseline_initial.to_numpy(),
            amplitude=15.0,
            duration_sec=15.0,
            recovery_tolerance_bpm=5.0,
            recovery_sec=5.0,
        )

        fhr_clean = fhr.copy()
        fhr_clean[event_mask] = np.nan

        baseline = (
            pd.Series(fhr_clean)
            .rolling(
                window=window_size,
                min_periods=min_valid,
                center=center,
            )
            .mean()
        )

        # conservamos la estimación robusta inicial en vez de perder toda la baseline.
        baseline = baseline.where(~baseline.isna(), baseline_initial)
        return baseline

    def _get_event_mask(
        self,
        fhr: np.ndarray,
        baseline: np.ndarray,
        amplitude: float = 15.0,
        duration_sec: float = 15.0,
        recovery_tolerance_bpm: float = 5.0,
        recovery_sec: float = 5.0,
    ) -> np.ndarray:
        """
        Identify sustained FHR excursions from a preliminary baseline.

        Candidate events must exceed the amplitude threshold for at least the
        minimum duration. Their boundaries are extended until the FHR returns
        within the recovery tolerance of the baseline for a sustained period.
        """
        fhr = np.asarray(fhr, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        if len(fhr) != len(baseline):
            raise ValueError("fhr and baseline must have the same length")

        valid = ~np.isnan(fhr) & ~np.isnan(baseline)
        diff = fhr - baseline
        candidates = valid & (np.abs(diff) >= amplitude)
        min_samples = max(1, int(duration_sec * self.frequency))
        recovery_samples = max(1, int(recovery_sec * self.frequency))

        mask = np.zeros(len(fhr), dtype=bool)
        padded = np.r_[False, candidates, False]
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)

        for core_start, core_end in zip(starts, ends):
            if (core_end - core_start) < min_samples:
                continue

            sign = np.nanmedian(diff[core_start:core_end])
            if np.isnan(sign):
                continue
            sign = 1.0 if sign >= 0 else -1.0

            # mismo lado de la baseline y no haya recuperado la zona de ±5 bpm.
            start = core_start
            while start > 0:
                d = diff[start - 1]
                if np.isnan(d):
                    break
                if sign * d <= recovery_tolerance_bpm:
                    break
                start -= 1

            end = core_end
            i = core_end
            while i < len(fhr):
                j = min(len(fhr), i + recovery_samples)
                segment = diff[i:j]
                if len(segment) == recovery_samples and np.all(
                    np.isfinite(segment) & (np.abs(segment) <= recovery_tolerance_bpm)
                ):
                    end = i
                    break
                i += 1
            else:
                end = len(fhr)

            mask[start:end] = True

        return mask

    @staticmethod
    def _get_nan_series_from_series(series: pd.Series) -> pd.DataFrame:
        """
        Identify consecutive NaN segments in a time series.

        This method scans a pandas Series to detect sequences of NaN values
        and returns a DataFrame listing the start index and length of each gap.

        Args:
            series (pd.Series): Time series data possibly containing NaN values.

        Returns:
            pd.DataFrame: A DataFrame with two columns:
                - "Start Index": The index in the series where a NaN segment starts.
                - "Length": The number of consecutive NaN values in that segment.
        """
        # Convert NaNs to 1s and non-NaNs to 0s for detection
        is_nan = series.isna().astype(int)

        # Get the index values as a NumPy array
        index_vals = series.index.to_numpy()

        # Identify transitions (0->1 for start, 1->0 for end of NaN segments)
        cambios = np.diff(np.concatenate(([0], is_nan, [0])))

        # Positions where gaps start and end
        inicios = np.where(cambios == 1)[0]
        finales = np.where(cambios == -1)[0]

        resultados = []

        # Build a list of dictionaries with the start index and length of each NaN segment
        for inicio, fin in zip(inicios, finales):
            start_index = index_vals[inicio]
            length = fin - inicio
            resultados.append({"Start Index": start_index, "Length": length})

        return pd.DataFrame(resultados)

    def _drop_tail_nans(
        self, fhr: np.ndarray, uc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes trailing NaN values from the end of two aligned pandas Series (fhr and uc).

        Args:
            fhr (np.ndarray) : Fetal heart rate Series that may contain trailing NaNs.
            uc (np.ndarray) : Uterine contraction Series, assumed to be aligned with fhr.

        Returns:
            Tuple[np.ndarray, np.ndarray]: New Series with trailing NaNs removed from both fhr and uc.
        """
        if not isinstance(fhr, np.ndarray):
            raise TypeError("La entrada debe ser numpy.ndarray")

        valid_mask = ~np.isnan(fhr)

        if not valid_mask.any():
            return fhr[:0], uc[:0]

        last_valid_index = np.where(valid_mask)[0][-1]

        # Slice both Series up to the last valid index (inclusive)
        return fhr[: last_valid_index + 1], uc[: last_valid_index + 1]

    def _replace_gaps(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) == 0:
            print("WARNING (ctg.replace_gaps): Señal vacia!!!")
            return signal.copy()

        # Convert the maximum gap duration from seconds to samples.
        max_sec_gaps = self.config.preprocessing.max_sec_gaps
        freq = self.config.freq
        num_datos_nan = (max_sec_gaps * freq) + 1

        # Identify contiguous missing-data gaps.
        is_nan = np.isnan(signal)

        labeled_gaps, _ = label(is_nan)

        # Measure the length of each gap.
        gap_sizes = np.bincount(labeled_gaps)

        gaps_to_interpolate = np.where(gap_sizes <= num_datos_nan)[0]
        gaps_to_interpolate = gaps_to_interpolate[gaps_to_interpolate != 0]

        # Interpolate only gaps that satisfy the configured duration limit.
        mask_to_fill = np.isin(labeled_gaps, gaps_to_interpolate)

        # Compute a linear interpolation of the signal.
        series = pd.Series(signal)
        interpolated_series = series.interpolate(method="linear")

        # Keep interpolation values only inside eligible gaps.
        res_signal = np.where(mask_to_fill, interpolated_series, signal)

        return res_signal

    def _get_deceleration_condition(
        self,
        amplitude_deceleration: int = 15,
        time_in_deceleration: int = 15,
    ) -> pd.Series:
        fhr = np.asarray(self.fhr, dtype=float)
        baseline = np.asarray(self._baseline, dtype=float)

        events = self._get_deceleration_events(
            amplitude_deceleration=amplitude_deceleration,
            time_in_deceleration=time_in_deceleration,
        )
        mask = np.zeros(len(fhr), dtype=bool)
        for event in events:
            mask[event["start_idx"] : event["end_idx"]] = True

        invalid = np.isnan(fhr) | np.isnan(baseline)
        result = pd.Series(mask.astype(object), index=np.arange(len(mask)))
        result[invalid] = np.nan
        return result

    @staticmethod
    @staticmethod
    def _extend_condition_back(condition, window: int) -> np.ndarray:
        arr = np.asarray(condition, dtype=object).copy()
        if window <= 1 or len(arr) == 0:
            return arr

        is_true = np.array([x is True or x == 1 for x in arr], dtype=bool)
        prev_true = np.r_[False, is_true[:-1]]
        starts = np.flatnonzero(is_true & ~prev_true)
        for i in starts:
            start = max(0, i - int(window) + 1)
            arr[start:i] = True
        return arr

    def _get_repetitive_deceleration_condition(
        self,
        time_in_rep_dec: int,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        association_margin_sec: float = 60.0,
        center: bool = False,
        correlation_threshold: float = -0.4,
    ) -> np.ndarray:
        """
        Detect repetitive decelerations.

        Decelerations are considered repetitive when they are associated
        with more than 50% of the detected uterine contractions within
        the specified analysis window.

        At least two contractions must be present in the analysis window.

        Notes
        -----
        `correlation_threshold` is retained only for backwards compatibility.
        Correlation between FHR and UC is no longer used to determine
        repetitiveness.

        Parameters
        ----------
        time_in_rep_dec : int
            Duration of the analysis window in minutes.

        contraction_value_threshold : int
            UC increase above the estimated uterine baseline required
            for contraction detection.

        contraction_duration_threshold : int
            Minimum contraction duration in seconds.

        contraction_baseline_duration : int
            Window used to estimate the UC baseline, in seconds.

        association_margin_sec : float
            Maximum temporal margin used when associating decelerations
            with contractions.

        center : bool
            Whether the analysis window is centered.

        correlation_threshold : float
            Deprecated. Retained only for backwards compatibility.

        Returns
        -------
        np.ndarray
            Boolean/object array indicating repetitive-deceleration periods.
            NaN is returned where the condition cannot be evaluated.
        """

        n = len(self.fhr)

        if self.uc is None:
            return np.full(
                n,
                np.nan,
                dtype=object,
            )

        # Detect contraction and deceleration events

        contraction_events = self._get_contraction_events(
            time_shift=0,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=center,
        )

        deceleration_events = self._get_deceleration_events()

        if len(contraction_events) == 0:
            return np.full(
                n,
                np.nan,
                dtype=object,
            )

        # Associate contractions and decelerations

        associations = self._associate_decelerations_to_contractions(
            deceleration_events=deceleration_events,
            contraction_events=contraction_events,
            association_margin_sec=association_margin_sec,
        )

        # Represent each contraction by an impulse at its peak

        contraction_impulse = np.zeros(
            n,
            dtype=float,
        )

        associated_impulse = np.zeros(
            n,
            dtype=float,
        )

        for association in associations:

            contraction = association["contraction"]

            peak = contraction["peak_idx"]

            contraction_impulse[peak] = 1.0

            if association["associated"]:
                associated_impulse[peak] = 1.0

        # Complete analysis window

        window = max(
            1,
            int(time_in_rep_dec * 60 * self.frequency),
        )

        # IMPORTANT:
        # min_periods=window means that a 30-min criterion is
        # not evaluated before 30 min of signal are available.
        total_contractions = (
            pd.Series(contraction_impulse)
            .rolling(
                window=window,
                min_periods=window,
                center=center,
            )
            .sum()
        )

        associated_contractions = (
            pd.Series(associated_impulse)
            .rolling(
                window=window,
                min_periods=window,
                center=center,
            )
            .sum()
        )

        # Fraction of contractions with a deceleration

        ratio = associated_contractions / total_contractions.replace(
            0,
            np.nan,
        )

        # More than 50%, not >= 50%.
        repetitive = (total_contractions >= 2) & (ratio > 0.5)

        # Preserve undefined regions as NaN

        result = np.full(
            n,
            np.nan,
            dtype=object,
        )

        valid = total_contractions.notna() & (total_contractions > 0)

        result[valid.to_numpy()] = repetitive[valid].to_numpy(dtype=bool)

        return result

    # FIGO: inicio/retorno gradual (>30 s) y, cuando UC está bien registrada,
    def _get_late_deceleration_condition(
        self,
        dec_cond: np.ndarray,
        center: bool = False,
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
    ) -> np.ndarray:
        """
        Detect late decelerations based on deceleration morphology
        and temporal relationship with uterine contractions.

        A detected deceleration is considered late when:

        1. Its onset-to-nadir time indicates a gradual decrease
        (>= time_in_late_dec_cond seconds).
        2. The deceleration starts after the onset of the contraction
        by at least `contraction_time_shift` seconds.
        3. The FHR nadir occurs after the contraction peak.
        4. FHR recovery occurs after the end of the contraction.

        Parameters
        ----------
        dec_cond : np.ndarray
            Deceleration condition mask. Retained for compatibility.

        center : bool
            Whether rolling windows used for contraction detection are centered.

        time_in_late_dec_cond : int
            Minimum onset-to-nadir time, in seconds, used to define
            a gradual deceleration.

        contraction_time_shift : int
            Minimum delay between contraction onset and deceleration onset.

        contraction_value_threshold : int
            UC amplitude threshold.

        contraction_duration_threshold : int
            Minimum contraction duration in seconds.

        contraction_baseline_duration : int
            UC baseline estimation window in seconds.

        Returns
        -------
        np.ndarray
            Boolean/object array indicating late decelerations.
        """

        n = len(self.fhr)

        result = np.zeros(
            n,
            dtype=object,
        )

        if self.uc is None:
            result[:] = False
            return result

        # Extract events

        deceleration_events = self._get_deceleration_events()

        contraction_events = self._get_contraction_events(
            time_shift=0,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=center,
        )

        delay_samples = int(contraction_time_shift * self.frequency)

        # Evaluate each deceleration

        for deceleration in deceleration_events:

            # Gradual fall: onset -> nadir >= 30 s by default
            gradual = deceleration["onset_to_nadir_sec"] >= time_in_late_dec_cond

            if not gradual:
                continue

            for contraction in contraction_events:

                delayed_onset = (
                    deceleration["start_idx"]
                    >= contraction["start_idx"] + delay_samples
                )

                nadir_after_peak = deceleration["nadir_idx"] > contraction["peak_idx"]

                recovery_after_contraction = (
                    deceleration["end_idx"] > contraction["end_idx"]
                )

                if delayed_onset and nadir_after_peak and recovery_after_contraction:

                    result[deceleration["start_idx"] : deceleration["end_idx"]] = True

                    break

        invalid = np.isnan(
            np.asarray(
                self.fhr,
                dtype=float,
            )
        )

        result[invalid] = np.nan

        return result

    # antes de detectar su inicio, pico y final.
    def _get_contraction_condition(
        self,
        center: bool = False,
        time_shift: int = 20,
        value_threshold: int = 10,
        duration_threshold: int = 20,
        baseline_duration: int = 120,
    ) -> np.ndarray:
        if self.uc is None:
            return np.full(len(self.fhr), np.nan, dtype=object)

        uc = pd.Series(np.asarray(self.uc, dtype=float))
        freq = self.frequency
        baseline_window = max(1, int(baseline_duration * freq))
        duration_window = max(1, int(duration_threshold * freq))
        min_base = max(
            1, int(self.config.correlation.not_nans_threshold_porc * baseline_window)
        )
        min_duration = max(
            1, int(self.config.correlation.not_nans_threshold_porc * duration_window)
        )

        base_min = uc.rolling(
            window=baseline_window,
            min_periods=min_base,
            center=center,
        ).min()
        diff = uc - base_min
        sustained = diff.rolling(
            window=duration_window,
            min_periods=min_duration,
            center=center,
        ).min()

        raw = np.where(np.isnan(sustained), np.nan, sustained > value_threshold).astype(
            object
        )
        raw = self._extend_condition_back(raw, duration_window)

        if time_shift:
            numeric = np.array(
                [
                    (
                        np.nan
                        if (x is None or (isinstance(x, float) and np.isnan(x)))
                        else float(bool(x))
                    )
                    for x in raw
                ],
                dtype=float,
            )
            shifted = self.shift_numpy(
                numeric, int(time_shift * freq), fill_value=np.nan
            )
            return np.where(np.isnan(shifted), np.nan, shifted > 0.5).astype(object)
        return raw

    def _get_prolonged_deceleration_condition(
        self,
        dec_cond: pd.Series,
        time_in_prolonged_dec: int = 3,
        center: bool = False,
    ) -> np.ndarray:
        n = len(self.fhr)
        result = np.zeros(n, dtype=object)
        threshold_sec = float(time_in_prolonged_dec) * 60.0
        for event in self._get_deceleration_events():
            if event["duration_sec"] > threshold_sec:
                result[event["start_idx"] : event["end_idx"]] = True
        invalid = np.isnan(np.asarray(self.fhr, dtype=float))
        result[invalid] = np.nan
        return result

    def _get_reduced_variability_condition(
        self,
        center: bool = False,
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
    ) -> np.ndarray:
        fhr = np.asarray(self.fhr, dtype=float)
        freq = self.frequency
        dec_cond = np.asarray(
            self._get_deceleration_condition(
                amplitude_deceleration=amplitude_dec_cond,
                time_in_deceleration=time_in_deceleration,
            ),
            dtype=object,
        )
        event_mask = self._get_event_mask(
            fhr,
            np.asarray(self._baseline, dtype=float),
            amplitude=15,
            duration_sec=15,
        )

        def bandwidth_1min(values):
            s = pd.Series(values)
            w = max(1, int(60 * freq))
            mp = max(1, int(self.config.correlation.not_nans_threshold_porc * w))
            return (
                s.rolling(w, min_periods=mp, center=center).max()
                - s.rolling(w, min_periods=mp, center=center).min()
            )

        # Estimate baseline variability after excluding transient FHR events.
        fhr_base = fhr.copy()
        fhr_base[event_mask] = np.nan
        bw_base = bandwidth_1min(fhr_base)
        base_low = bw_base < red_var_bandwith
        base_window = max(1, int(red_var_duration_baseline * 60 * freq))
        base_sustained = (
            base_low.astype(float)
            .rolling(
                base_window,
                min_periods=max(
                    1,
                    int(self.config.correlation.not_nans_threshold_porc * base_window),
                ),
                center=center,
            )
            .min()
            == 1.0
        )

        # During decelerations, evaluate variability on the original FHR signal.
        bw_raw = bandwidth_1min(fhr)
        dec_bool = np.array([x is True or x == 1 for x in dec_cond], dtype=bool)
        dec_low = (bw_raw < red_var_bandwith).to_numpy() & dec_bool
        dec_window = max(1, int(red_var_duration_deceleration * 60 * freq))
        dec_sustained = (
            pd.Series(dec_low.astype(float))
            .rolling(
                dec_window,
                min_periods=max(
                    1, int(self.config.correlation.not_nans_threshold_porc * dec_window)
                ),
                center=center,
            )
            .min()
            == 1.0
        )

        out = (base_sustained.to_numpy() | dec_sustained.to_numpy()).astype(object)
        out[np.isnan(fhr)] = np.nan
        return out

    def _get_increased_variability_condition(
        self,
        window_var: int = 1,
        center: bool = False,
        min_time: int = 30,
        bandwidth_min: float = 25,
    ) -> np.ndarray:
        fhr = np.asarray(self.fhr, dtype=float)
        freq = self.frequency
        event_mask = self._get_event_mask(
            fhr,
            np.asarray(self._baseline, dtype=float),
            amplitude=15,
            duration_sec=15,
        )
        clean = fhr.copy()
        clean[event_mask] = np.nan
        s = pd.Series(clean)
        w = max(1, int(60 * freq * window_var))
        mp = max(1, int(self.config.correlation.not_nans_threshold_porc * w))
        bw = (
            s.rolling(w, min_periods=mp, center=center).max()
            - s.rolling(w, min_periods=mp, center=center).min()
        )

        long_w = max(1, int(min_time * 60 * freq))
        long_mp = max(1, int(self.config.correlation.not_nans_threshold_porc * long_w))
        sustained = (bw > bandwidth_min).astype(float).rolling(
            long_w, min_periods=long_mp, center=center
        ).min() == 1.0
        out = sustained.to_numpy(dtype=object)
        out[np.isnan(fhr)] = np.nan
        return out

    def _get_variable_deceleration_condition(self, center: bool = False) -> pd.Series:
        result = np.zeros(len(self.fhr), dtype=object)
        late = self._get_late_deceleration_condition(
            dec_cond=self._get_deceleration_condition(), center=center
        )
        late_bool = np.array([x is True or x == 1 for x in late], dtype=bool)

        for event in self._get_deceleration_events():
            if event["duration_sec"] > 180:
                continue
            rapid_drop = event["onset_to_nadir_sec"] < 30.0
            rapid_recovery = event["recovery_sec"] < 30.0
            if rapid_drop and rapid_recovery and not late_bool[event["nadir_idx"]]:
                result[event["start_idx"] : event["end_idx"]] = True

        result[np.isnan(np.asarray(self.fhr, dtype=float))] = np.nan
        return pd.Series(result, index=np.arange(len(result)))

    def _get_early_deceleration_condition(
        self,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        center: bool = False,
    ) -> pd.Series:
        n = len(self.fhr)
        result = np.zeros(n, dtype=object)
        if self.uc is None:
            result[:] = np.nan
            return pd.Series(result)

        contractions = self._get_contraction_events(
            time_shift=0,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=center,
        )
        fhr = np.asarray(self.fhr, dtype=float)
        tolerance = int(20 * self.frequency)

        for d in self._get_deceleration_events():
            gradual = d["onset_to_nadir_sec"] >= 30.0
            if not gradual:
                continue
            seg = fhr[d["start_idx"] : d["end_idx"]]
            normal_var = len(seg) > 0 and np.nanmax(seg) - np.nanmin(seg) >= 5.0
            for c in contractions:
                coincident_peak = abs(d["nadir_idx"] - c["peak_idx"]) <= tolerance
                overlap = (
                    d["start_idx"] < c["end_idx"] and d["end_idx"] > c["start_idx"]
                )
                if coincident_peak and overlap and normal_var:
                    result[d["start_idx"] : d["end_idx"]] = True
                    break

        result[np.isnan(fhr)] = np.nan
        return pd.Series(result, index=np.arange(n))

    def _get_deceleration_events(
        self,
        amplitude_deceleration: float = 15.0,
        time_in_deceleration: float = 15.0,
        recovery_tolerance_bpm: float = 5.0,
        recovery_sec: float = 5.0,
    ) -> list:
        fhr = np.asarray(self.fhr, dtype=float)
        baseline = np.asarray(self._baseline, dtype=float)
        diff = baseline - fhr
        valid = np.isfinite(fhr) & np.isfinite(baseline)
        core = valid & (diff > amplitude_deceleration)
        min_samples = max(1, int(time_in_deceleration * self.frequency))

        padded = np.r_[False, core, False]
        trans = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(trans == 1)
        ends = np.flatnonzero(trans == -1)
        events = []
        recovery_samples = max(1, int(recovery_sec * self.frequency))

        for core_start, core_end in zip(starts, ends):
            if core_end - core_start <= min_samples:
                continue

            # Extend the onset until the FHR returns close to the baseline.
            start = core_start
            while (
                start > 0
                and np.isfinite(diff[start - 1])
                and diff[start - 1] > recovery_tolerance_bpm
            ):
                start -= 1

            # Extend the end until a sustained recovery to baseline is observed.
            end = core_end
            i = core_end
            while i < len(fhr):
                j = min(len(fhr), i + recovery_samples)
                seg = diff[i:j]
                if len(seg) == recovery_samples and np.all(
                    np.isfinite(seg) & (seg <= recovery_tolerance_bpm)
                ):
                    end = i
                    break
                i += 1
            else:
                end = len(fhr)

            if end <= start:
                continue
            segment = fhr[start:end]
            if not np.isfinite(segment).any():
                continue
            nadir_rel = int(np.nanargmin(segment))
            nadir = start + nadir_rel
            amplitude = baseline[nadir] - fhr[nadir]
            duration_sec = (end - start) / self.frequency
            onset_to_nadir_sec = (nadir - start) / self.frequency
            recovery_sec_value = (end - nadir) / self.frequency

            events.append(
                {
                    "start_idx": int(start),
                    "nadir_idx": int(nadir),
                    "end_idx": int(end),
                    "amplitude": float(amplitude),
                    "duration_sec": float(duration_sec),
                    "onset_to_nadir_sec": float(onset_to_nadir_sec),
                    "recovery_sec": float(recovery_sec_value),
                }
            )
        return events

    def _get_contraction_events(
        self,
        center: bool = False,
        time_shift: int = 0,
        value_threshold: int = 10,
        duration_threshold: int = 20,
        baseline_duration: int = 120,
    ) -> list:
        if self.uc is None:
            return []
        mask = self._get_contraction_condition(
            center=center,
            time_shift=time_shift,
            value_threshold=value_threshold,
            duration_threshold=duration_threshold,
            baseline_duration=baseline_duration,
        )
        bool_mask = np.array([x is True or x == 1 for x in mask], dtype=bool)
        padded = np.r_[False, bool_mask, False]
        trans = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(trans == 1)
        ends = np.flatnonzero(trans == -1)
        uc = np.asarray(self.uc, dtype=float)
        events = []
        for start, end in zip(starts, ends):
            if end <= start:
                continue
            seg = uc[start:end]
            if not np.isfinite(seg).any():
                continue
            peak = start + int(np.nanargmax(seg))
            events.append(
                {
                    "start_idx": int(start),
                    "peak_idx": int(peak),
                    "end_idx": int(end),
                    "duration_sec": float((end - start) / self.frequency),
                }
            )
        return events

    def _associate_decelerations_to_contractions(
        self,
        deceleration_events: list,
        contraction_events: list,
        association_margin_sec: float = 60.0,
    ) -> list:
        """
        Associate detected decelerations with uterine contractions.

        Each contraction can be associated with at most one deceleration and
        each deceleration can be assigned to at most one contraction.

        Association is based on temporal overlap between the deceleration and
        an extended contraction interval. When several possible associations
        exist, the pair with the smallest distance between the contraction peak
        and the deceleration nadir is selected.

        Parameters
        ----------
        deceleration_events : list
            Output of `_get_deceleration_events()`.

        contraction_events : list
            Output of `_get_contraction_events()`.

        association_margin_sec : float
            Temporal margin added before and after each contraction when
            searching for an associated deceleration.

        Returns
        -------
        list of dict
            One dictionary per contraction with:
                - contraction
                - deceleration
                - associated
        """

        associations = []

        if len(contraction_events) == 0:
            return associations

        margin = int(association_margin_sec * self.frequency)

        # Keep track of decelerations already assigned to a contraction.
        used_decelerations = set()

        for contraction in contraction_events:

            c_start = max(0, contraction["start_idx"] - margin)

            c_end = min(len(self.fhr), contraction["end_idx"] + margin)

            candidates = []

            for dec_idx, deceleration in enumerate(deceleration_events):

                if dec_idx in used_decelerations:
                    continue

                # Temporal overlap between deceleration and
                # extended contraction interval.
                overlaps = (
                    deceleration["start_idx"] < c_end
                    and deceleration["end_idx"] > c_start
                )

                if not overlaps:
                    continue

                # Prefer the deceleration whose nadir is closest
                # to the contraction peak.
                distance = abs(deceleration["nadir_idx"] - contraction["peak_idx"])

                candidates.append(
                    (
                        distance,
                        dec_idx,
                        deceleration,
                    )
                )

            if len(candidates) == 0:

                associations.append(
                    {
                        "contraction": contraction,
                        "deceleration": None,
                        "associated": False,
                    }
                )

                continue

            # Smallest peak-nadir distance
            candidates.sort(key=lambda x: x[0])

            _, dec_idx, deceleration = candidates[0]

            used_decelerations.add(dec_idx)

            associations.append(
                {
                    "contraction": contraction,
                    "deceleration": deceleration,
                    "associated": True,
                }
            )

        return associations

    def _get_tachysystole_condition(self, center: bool = False) -> np.ndarray:
        n = len(self.fhr)
        if self.uc is None:
            return np.full(n, np.nan, dtype=object)
        events = self._get_contraction_events(center=center, time_shift=0)
        impulse = np.zeros(n, dtype=float)
        for event in events:
            impulse[event["peak_idx"]] = 1.0

        w10 = max(1, int(10 * 60 * self.frequency))
        c10 = pd.Series(impulse).rolling(w10, min_periods=1, center=center).sum()
        high10 = c10 > 5
        # Require the >5 contractions/10 min condition to persist for 10 minutes.
        two_successive = (
            high10.astype(float).rolling(w10, min_periods=w10, center=center).min()
            == 1.0
        )

        w30 = max(1, int(30 * 60 * self.frequency))
        c30 = pd.Series(impulse).rolling(w30, min_periods=1, center=center).sum()
        averaged30 = c30 > 15  # Equivalent to >5 contractions per 10 min over 30 min
        return (two_successive.to_numpy() | averaged30.to_numpy()).astype(object)
