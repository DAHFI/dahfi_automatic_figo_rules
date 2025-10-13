import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Union, Dict

# TODO: Para los plots podría poner los times en minutos (queda más claro)


class CTG:

    # Label constants for classification
    NORMAL = 0
    SUSPICIOUS = 1
    PATHOLOGICAL = 2

    def __init__(
        self,
        fhr,  # Fetal Heart Rate signal (list or array)
        uc,  # Uterine Contraction signal (list or array)
        freq,  # Frequency (Hz)
        # Optional data
        ph=None,  # Optional pH value
        id=None,  # Optional ID for the record or patient
        # Thresholds and constants
        MAX_TIME_FHR=250,  # Maximum plausible FHR value
        MIN_TIME_FHR=0,  # Minimum plausible FHR value
        MAX_TIME_UC=150,  # Maximum plausible UC value
        MIN_TIME_UC=0,  # Minimum plausible UC value
        NOT_NAN_PERC=0.5,  # Minimum percentage of non-NaN values required in a window
    ):

        # Ensure FHR and UC signals are of the same length
        if len(fhr) != len(uc):
            raise ValueError("ERROR: The size of the FHR and UC must be equal")

        # Define time axis based on frequency (in minutes)
        self._time = np.arange(0, len(fhr) / freq, 1 / freq)

        # Store metadata and time-indexed signals as pandas Series
        self.id = id
        self.fhr = pd.Series(data=fhr, index=self._time)
        self.uc = pd.Series(data=uc, index=self._time)
        self.ph = ph
        self.frequency = freq

        # Store physiological limits for plausibility checks
        self.MAX_TIME_FHR = MAX_TIME_FHR
        self.MIN_TIME_FHR = MIN_TIME_FHR
        self.MAX_TIME_UC = MAX_TIME_UC
        self.MIN_TIME_UC = MIN_TIME_UC

        # Threshold for minimum non-NaN data required in rolling computations
        self.NOT_NAN_PERC = NOT_NAN_PERC

        # Placeholders for preprocessing results (e.g., baseline, decelerations, etc.)
        # self._preprocess_type
        # self._baseline
        # self._baseline_labels
        # self._decelerations_labels
        # self._variability_labels
        # self._conclusion_labels

    ## ---------------------------------------------
    ## --------------- PREPROCESSING ---------------

    def preprocess_rules_figo(
        self,
        cut_time: int = 60,
        max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
    ) -> None:
        """
        Preprocess the FHR and UC signals according to FIGO guidelines.

        This includes:
        - Optionally removing trailing NaNs
        - Cutting the last N minutes of signal
        - Resetting time index
        - Removing physiologically implausible values
        - Interpolating small gaps, preserving large ones

        Args:
            cut_time (int): Number of minutes to retain from the end of the signal.
            max_size_gaps (int): Maximum size (in seconds) of gaps to interpolate. Larger gaps remain NaN.
            rm_tail_nan (bool): Whether to remove trailing NaNs before cutting the signal.
        """

        if self.fhr.dropna().empty or self.uc.dropna().empty:
            raise ValueError(
                "FHR or UC signals are empty, preprocessing cannot be applied."
            )

        # Create copies of the signals
        fhr_aux = self.fhr.copy()
        uc_aux = self.uc.copy()

        # We set the values ​​not allowed for ctg signals to nan
        fhr_aux[(fhr_aux <= self.MIN_TIME_FHR) | (fhr_aux > self.MAX_TIME_FHR)] = np.nan
        uc_aux[(uc_aux < self.MIN_TIME_UC) | (uc_aux > self.MAX_TIME_UC)] = np.nan

        # Optionally remove trailing NaN values
        if rm_tail_nan:
            fhr_aux, uc_aux = self._drop_tail_nans(fhr_aux, uc_aux)

        # We define the number of elements that we are going to cut
        num_cut_rows = self.frequency * cut_time * 60

        # Keep only the last N minutes of data
        fhr_aux = fhr_aux.tail(num_cut_rows)
        uc_aux = uc_aux.tail(num_cut_rows)

        # We updated the new indexes for the cut signals
        time = np.arange(
            0, len(fhr_aux) * (1 / (self.frequency)), (1 / (self.frequency))
        )
        fhr_aux.index = time
        uc_aux.index = time

        # We save the preprocess we have used in a variable
        self._preprocess_type = "FIGO rules"

        # We process the gaps in the signals
        self.fhr = self._replace_gaps(fhr_aux, max_size_gaps * self.frequency)
        self.uc = self._replace_gaps(uc_aux, max_size_gaps * self.frequency)

        # We update the time
        self._time = pd.Series(time)

    ## ---------------------------------------------
    ## ------------------ GENERAL ------------------

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
            window_time_baseline (int): Time window to estimate the baseline (in seconds).
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

        # Supported ruleset(s)
        correct_rules_type = ["FIGO"]
        if rules_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # TODO: Hacer algo con este mensaje
        # if hasattr(self, "_preprocess_type"):
        #     print("Appling FIGO rules on pre-processing: ", self._preprocess_type)

        # Apply FIGO ruleset if specified
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

    ## -----------------------------------------------
    ## --------------- TEST FUNCTIONS ----------------

    def _get_baseline_graph(self):
        """
        Plot the Fetal Heart Rate (FHR) signal along with its computed baseline.

        This method generates a matplotlib figure displaying the FHR signal, the
        baseline, and color-coded zones indicating normal, suspicious, and pathological
        ranges based on FHR values.

        If the FHR signal is entirely NaN, a warning message is printed and the
        method exits without plotting.

        Returns:
            None
        """

        t = self._time
        FHR = self.fhr

        # Check if the entire FHR signal is NaN
        if FHR.isna().all():
            print("The FHR signal is NaN.")
            return

        baseline = self._baseline

        # Create a figure for the plot
        plt.figure(figsize=(10, 3))

        # Plot FHR and baseline
        plt.plot(t, FHR, color="black", label="FHR")
        plt.plot(t, baseline, color="blue", label="Baseline")

        # Fill background regions based on clinical interpretation
        plt.fill_between(t, 110, 160, color="green", alpha=0.3, label="N")
        plt.fill_between(t, 100, 110, color="orange", alpha=0.3)
        plt.fill_between(t, 160, 200, color="orange", alpha=0.3, label="S")
        plt.fill_between(t, 0, 100, color="red", alpha=0.3, label="P")

        # Configure axis labels and limits
        plt.xlabel("Time (sec)")
        plt.ylabel("FHR (bpm)")
        plt.ylim(np.nanmin(FHR) - 5, np.nanmax(FHR) + 5)

        # Add title, legend, and grid
        plt.title("FHR and Baseline")
        plt.legend(loc="upper left")
        plt.grid(True)

        # Display the plot
        plt.show()

    def _get_variability_graph(self, center: bool = False):
        # TODO: AQUÍ HAY QUE AÑADIR MÁS PARAMETROS !

        """
        Plot the fetal heart rate (FHR) signal and highlight variability classifications.

        This method visualizes variability using:
            - Black line: FHR signal
            - Red dashed line: 1-minute rolling maximum
            - Blue dashed line: 1-minute rolling minimum
            - Green line: 1-minute variability bandwidth (max - min)
            - Green shaded area: Normal variability (label 0)
            - Red shaded area: Pathological variability (label 2)

        Args:
            center (bool): Whether to center the rolling window during computation.

        Returns:
            None
        """

        FHR = self.fhr
        window_size = int(self.frequency * 60)
        min_periods = int(self.NOT_NAN_PERC * window_size)

        # Rolling max and min values over 1-minute windows
        rolling_fhr = FHR.rolling(
            window=window_size, min_periods=min_periods, center=center
        )
        max_fhr = rolling_fhr.max()
        min_fhr = rolling_fhr.min()

        # Variability is defined as the difference between max and min
        variability_bandwidth = max_fhr - min_fhr

        # Create figure
        plt.figure(figsize=(10, 5))

        # Plot the FHR signal
        plt.plot(self._time, FHR, label="FHR", color="black", alpha=0.6)

        # Plot 1-minute max and min FHR
        plt.plot(
            self._time,
            max_fhr,
            label="Max FHR (1 min)",
            color="red",
            linestyle="--",
            alpha=0.5,
        )
        plt.plot(
            self._time,
            min_fhr,
            label="Min FHR (1 min)",
            color="blue",
            linestyle="--",
            alpha=0.5,
        )

        # Plot the variability bandwidth
        plt.plot(
            self._time,
            variability_bandwidth,
            label="FHR Variability (1 min)",
            color="green",
        )

        # Fill background for normal and pathological variability
        max_fill = np.nanmax(FHR) + 5
        plt.fill_between(
            self._time,
            0,
            max_fill,
            where=(self._variability_labels == 0),
            color="lightgreen",
            alpha=0.5,
        )
        plt.fill_between(
            self._time,
            -1,
            max_fill,
            where=(self._variability_labels == 2),
            color="red",
            alpha=0.5,
        )

        # Axis labels, title, legend, and layout
        plt.xlabel("Time (min)")
        plt.ylabel("FHR (bpm)")
        plt.title("FHR Variability (1-min bandwidth)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_contractions_graph(
        self,
        # -- contraction info --
        contraction_time_shift=0,
        contraction_value_threshold=10,
        contraction_duration_threshold=20,
        contraction_baseline_duration=120,
        contraction_center=True,
    ):

        t = self._time
        UC = self.uc
        is_contraction = self._get_contraction_condition(
            time_shift=contraction_time_shift,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=contraction_center,
        )

        plt.figure(figsize=(15, 5))
        plt.plot(t, UC, label="UC", color="black")

        is_contraction = is_contraction.fillna(False)
        in_condition = False
        start = None

        for i in range(len(is_contraction)):
            if (
                is_contraction.iloc[i] and not in_condition
            ):  # Comienza una serie de True
                in_condition = True
                start = t[i]
            elif (
                not is_contraction.iloc[i] and in_condition
            ):  # Termina una serie de True
                in_condition = False
                end = t[i]
                plt.axvspan(start, end, color="blue", alpha=0.5)

        # Caso especial: si termina en True, cerramos al final
        if in_condition:
            plt.axvspan(start, t.iloc[-1], color="blue", alpha=0.5)

        # plt.title("Contractions in UC")
        plt.xlabel("Time (min)")
        plt.ylabel("Pressure (mmHg)")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        # TODO 1: Borrar
        plt.savefig("contraction_graph.png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_ctg(self):
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

        # Split signal length in half
        mid = len(self.fhr) // 2

        # Divide the FHR and UC signals into first and second halves
        fhr1, fhr2 = self.fhr.iloc[:mid], self.fhr.iloc[mid:]
        uc1, uc2 = self.uc.iloc[:mid], self.uc.iloc[mid:]

        # Create 4 vertically stacked subplots (FHR and UC for both halves)
        _, axs = plt.subplots(4, 1, figsize=(20, 12), sharex=False)

        # Plot FHR (first half)
        ax1 = axs[0]
        ax1.plot(fhr1.index / 60, fhr1.values, color="black", label="FHR")
        ax1.set_ylabel("FHR")
        ax1.grid(True)

        # Plot UC (first half)
        ax2 = axs[1]
        ax2.plot(uc1.index / 60, uc1.values, color="blue", alpha=0.6, label="UC")
        ax2.set_ylabel("UC")
        ax2.grid(True)

        # Plot FHR (second half)
        ax3 = axs[2]
        ax3.plot(fhr2.index / 60, fhr2.values, color="black", label="FHR")
        ax3.set_ylabel("FHR")
        ax3.grid(True)

        # Plot UC (second half)
        ax4 = axs[3]
        ax4.plot(uc2.index / 60, uc2.values, color="blue", alpha=0.6, label="UC")
        ax4.set_xlabel("Time (sec)")
        ax4.set_ylabel("UC")
        ax4.grid(True)

        # Add a global title with the CTG identifier
        title = "CTG: " + str(self.id)
        plt.suptitle(title, fontsize=16)

        # Adjust layout to leave space for the main title
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    ## ----------------------------------------------------
    ## --------------- RULES FIGO METODS ------------------

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
        Applies the FIGO guidelines for fetal heart rate analysis, integrating baseline,
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

        # Optionally apply preprocessing before analysis
        if preprocess:

            # Warn if the signal has already been preprocessed
            if hasattr(self, "_preprocess_type"):
                print(
                    "The ctg has been previously preprocessed following: ",
                    self._preprocess_type,
                )

            self.preprocess_rules_figo(
                cut_time=prep_cut_time,
                max_size_gaps=prep_max_size_gaps,
                rm_tail_nan=rm_tail_nan,
            )

        # Compute individual FIGO criteria
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

        # Combine criteria into overall classification
        conclusion_labels = self._get_conclusion_labels(graph=conclusion_graph)

        # Optionally return a dictionary with all intermediate results
        if get_dicc:
            return {
                "baseline": base_labels,
                "deceleration": decelerations_labels,
                "variability": variability_labels,
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
        Compute and classify the FHR baseline using FIGO thresholds.

        This method calculates the baseline of the FHR signal using a rolling mean,
        then classifies each time point based on FIGO guidelines:
            - NORMAL: 110–160 bpm
            - SUSPICIOUS: <110 bpm or >160 bpm
            - PATHOLOGICAL: <100 bpm

        Optionally, it can display plots of the FHR signal with classification overlays.

        Args:
            baseline_window_min_size (int): Minimum window size (in seconds) used to compute the baseline.
            graph (bool): Whether to display the baseline and classification graphs.
            center (bool): Whether the rolling mean should be centered.

        Returns:
            pd.Series: A pandas Series containing the classification labels for each time point.
        """

        # Initialize a label array with NaNs, same shape as FHR
        labels = np.nan * np.ones_like(self.fhr)

        # Compute the baseline (rolling mean of FHR)
        baseline = self._get_baseline(window_min_size=window_min_size, center=center)

        # TODO: Por encima de 160 no indica hipoxia (Asegurarme y cambiarlo)
        # Apply FIGO classification thresholds
        labels[(baseline >= 110.0) & (baseline <= 160.0)] = self.NORMAL
        labels[(baseline < 110.0) | (baseline > 160.0)] = self.SUSPICIOUS
        labels[(baseline < 100)] = self.PATHOLOGICAL

        # Store baseline and labels as class attributes
        self._baseline = baseline
        self._baseline_labels = labels

        # Optionally plot the results
        if graph:
            self._baseline_graph(show=1, grid=None)
            self._get_baseline_graph()

        return pd.Series(labels)

    def _get_decelerations_labels(
        self,
        center: bool = False,
        # -- baseline info (Only check) --
        window_time_baseline: int = 10,
        # -- dec condition --
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        # -- rep dec cond --
        time_in_rep_dec: int = 30,
        correlation_threshold: float = -0.4,
        # -- late_dec_cond --
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        # -- prolonged dec cond --
        time_in_prolonged_dec: int = 3,
        # -- rep dec + red variab --
        time_in_rep_dec_for_red_var: int = 20,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        # -- time for pathological prolonged dec --
        time_patholog_prolonged_dec: int = 5,
        # -- graph --
        graph: bool = False,
    ) -> pd.Series:
        """
        Classify fetal heart rate (FHR) decelerations using FIGO criteria into:
            - Normal
            - Suspicious
            - Pathological

        Combines multiple deceleration types based on:
            • Amplitude and duration,
            • Temporal relationship with uterine contractions,
            • Repetitiveness and variability.

        Args:
            center (bool): Whether to center rolling window computations.
            window_time_baseline (int): Minutes used to compute baseline FHR.
            amplitude_dec_cond (int): Minimum drop in bpm to qualify as a deceleration.
            time_in_deceleration (int): Minimum duration (in seconds) of a deceleration.
            time_in_rep_dec (int): Duration (in minutes) for detecting repetitive decelerations.
            correlation_threshold (float): Threshold for correlation with contractions.
            time_in_late_dec_cond (int): Minimum time to consider a deceleration as 'late'.
            contraction_time_shift (int): Seconds to shift contraction signal for alignment.
            contraction_value_threshold (int): Intensity threshold for a contraction.
            contraction_duration_threshold (int): Duration threshold to qualify as contraction.
            contraction_baseline_duration (int): Baseline window for contraction detection.
            time_in_prolonged_dec (int): Duration (in minutes) for prolonged deceleration.
            time_in_rep_dec_for_red_var (int): Duration for combining rep dec + red variability.
            red_var_duration_baseline (int): Duration for reduced variability at baseline.
            red_var_duration_deceleration (int): Duration for reduced variability in deceleration.
            red_var_bandwith (int): Variability threshold (in bpm).
            time_patholog_prolonged_dec (int): Duration (in minutes) for pathological prolonged deceleration.
            graph (bool): Whether to display the deceleration graph.

        Returns:
            pd.Series: Deceleration labels (NORMAL, SUSPICIOUS, PATHOLOGICAL).
        """

        # Return if FHR is entirely NaN
        if self.fhr.isna().all():
            self._decelerations_labels = pd.Series(
                np.nan, index=self.fhr.index, dtype="float"
            )
            return self._decelerations_labels

        # Compute baseline if not already available
        if not hasattr(self, "_baseline"):
            self._baseline = self._get_baseline(
                window_min_size=window_time_baseline, center=center
            )

        # Step 1: Detect simple decelerations (based on amplitude + duration)
        dec_cond = self._get_deceleration_condition(
            amplitude_deceleration=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
        )

        # Initialize label series with NaNs
        dec_label = pd.Series(np.nan, index=self.fhr.index, dtype="float")

        # Mark suspicious decelerations and normal zones
        dec_label[dec_cond == True] = self.SUSPICIOUS  # hay una desaceleración
        dec_label[dec_cond == False] = self.NORMAL

        # Step 2: Identify pathological patterns

        # 2.1 Repetitive decelerations (late or prolonged) > 30 min
        rep_dec_30_min_cond = self._get_repetitive_deceleration_condition(
            time_in_rep_dec=time_in_rep_dec,
            correlation_threshold=-correlation_threshold,
            center=center,
        )

        late_dec_cond = self._get_late_deceleration_condition(
            center=center,
            dec_cond=dec_cond,
            time_in_late_dec_cond=time_in_late_dec_cond,
            contraction_time_shift=contraction_time_shift,
            contraction_value_threshold=contraction_value_threshold,
            contraction_duration_threshold=contraction_duration_threshold,
            contraction_baseline_duration=contraction_baseline_duration,
        )

        prol_dec_3_min_cond = self._get_prolonged_deceleration_condition(
            dec_cond, time_in_prolonged_dec=time_in_prolonged_dec, center=center
        )

        dec_label[
            (rep_dec_30_min_cond == True)
            & ((late_dec_cond == True) | (prol_dec_3_min_cond == True))
        ] = self.PATHOLOGICAL

        # 2.2 Repetitive decelerations > 20 min with reduced variability
        rep_dec_20_min_cond = self._get_repetitive_deceleration_condition(
            time_in_rep_dec=time_in_rep_dec_for_red_var,
            correlation_threshold=correlation_threshold,
            center=center,
        )

        red_var_cond = self._get_reduced_variability_condition(
            center=center,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
        )

        # TODO: (Hacer comentario) ->
        # NOTA: No se entra nunca
        dec_label[
            (rep_dec_20_min_cond == True)
            & (red_var_cond == True)
            & ((late_dec_cond == True) | (prol_dec_3_min_cond == True))
        ] = self.PATHOLOGICAL

        # 2.3 A single prolonged deceleration (>5 minutes) is pathological
        prol_dec_5_min_cond = self._get_prolonged_deceleration_condition(
            dec_cond, time_in_prolonged_dec=time_patholog_prolonged_dec, center=center
        )

        dec_label[prol_dec_5_min_cond == True] = self.PATHOLOGICAL

        # Store the result
        self._decelerations_labels = dec_label

        # Optionally plot the results
        if graph:
            self._decelerations_graph()

        # TODO 1: BORRAR
        # if time_in_rep_dec == 30 and any(dec_label.fillna(False)):
        #     print("Se encontró un valor True. Deteniendo ejecución. id:", self.id)
        #     sys.exit()

        # BORRAR
        # dec_label = pd.Series(np.nan, index=self.fhr.index, dtype="float")
        # dec_label[
        #     (rep_dec_30_min_cond == True)
        #     & ((late_dec_cond == True) | (prol_dec_3_min_cond == True))
        # ] = self.PATHOLOGICAL

        # TODO 1 dec_le
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
        """
        Classify fetal heart rate (FHR) variability into NORMAL, SUSPICIOUS, and PATHOLOGICAL categories.

        This method computes rolling 1-minute windows to estimate variability amplitude (bandwidth),
        then applies thresholds to label segments of the signal. It also incorporates
        pathological conditions detected by reduced or increased variability functions.

        Args:
            graph (bool): Whether to display the variability classification graphs.
            center (bool): Whether the rolling window should be centered.
            window_var (int): Window size multiplier (in minutes) for rolling calculations.
            amplitude_dec_cond (int): Amplitude threshold for deceleration conditions.
            time_in_deceleration (int): Duration threshold for deceleration detection.
            incr_var_min_time (int): Minimum time to consider for increased variability.
            incr_var_bandwidth (int): Bandwidth threshold for increased variability detection.
            red_var_duration_baseline (int): Duration for reduced variability at baseline.
            red_var_duration_deceleration (int): Duration for reduced variability during deceleration.
            red_var_bandwith (int): Bandwidth threshold for reduced variability.

        Returns:
            pd.Series: Series with variability classification labels (NORMAL, SUSPICIOUS, PATHOLOGICAL).
        """

        FHR = self.fhr
        freq = self.frequency

        # Compute 1-minute rolling maximum and minimum FHR
        max_FHR_1_min = FHR.rolling(
            window=freq * 60 * window_var,
            min_periods=int(self.NOT_NAN_PERC * freq * 60 * window_var),
            center=center,
        ).max()
        min_FHR_1_min = FHR.rolling(
            window=freq * 60 * window_var,
            min_periods=int(self.NOT_NAN_PERC * freq * 60 * window_var),
            center=center,
        ).min()

        # Calculate FHR bandwidth (variability amplitude)
        FHR_bandwith_1_min = max_FHR_1_min - min_FHR_1_min

        # Initialize output labels as NaNs
        var_labels = np.nan * np.ones_like(FHR)

        # Label NORMAL variability: 5–25 bpm amplitude
        # TODO: Poner estos intervalos como param
        var_labels[(FHR_bandwith_1_min >= 5) & (FHR_bandwith_1_min <= 25)] = self.NORMAL

        # Label SUSPICIOUS variability: amplitude < 5 or > 25 bpm
        # Suspicious if < 5 or > 25
        var_labels[(FHR_bandwith_1_min < 5) | (FHR_bandwith_1_min > 25)] = (
            self.SUSPICIOUS
        )

        # Label PATHOLOGICAL variability based on:
        # 1. Reduced variability
        red_var_cond = self._get_reduced_variability_condition(
            center=center,
            amplitude_dec_cond=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
            red_var_duration_baseline=red_var_duration_baseline,
            red_var_duration_deceleration=red_var_duration_deceleration,
            red_var_bandwith=red_var_bandwith,
        )
        var_labels[(red_var_cond == True)] = self.PATHOLOGICAL  # CAMBIO: == -> = XD

        # 2. Increased variability
        inc_var_cond = self._get_increased_variability_condition(
            center=center,
            window_var=window_var,
            min_time=incr_var_min_time,
            bandwidth_min=incr_var_bandwidth,
        )
        var_labels[(inc_var_cond == True)] = self.PATHOLOGICAL

        # Store the result in an instance variable
        self._variability_labels = var_labels

        # Optionally display graph of variability classification
        if graph:
            self._variability_graph()
            self._get_variability_graph(center=center)

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
            (baseline_labels == self.SUSPICIOUS)
            | (decelerations_labels == self.SUSPICIOUS)
            | (variability_labels == self.SUSPICIOUS)
        ] = self.SUSPICIOUS

        # Assign "NORMAL" only if all three components are normal
        conclusion_labels[
            (baseline_labels == self.NORMAL)
            & (decelerations_labels == self.NORMAL)
            & (variability_labels == self.NORMAL)
        ] = self.NORMAL

        # Assign "PATHOLOGICAL" if any of the components is pathological
        conclusion_labels[
            (baseline_labels == self.PATHOLOGICAL)
            | (decelerations_labels == self.PATHOLOGICAL)
            | (variability_labels == self.PATHOLOGICAL)
        ] = self.PATHOLOGICAL

        # Store result in instance variable
        self._conclusion_labels = conclusion_labels

        # Optionally display graph
        if graph:
            self._conclusion_graph()

        return pd.Series(conclusion_labels)

    ## -----------------------------------------------------
    ## --------------- GRAPHS FUNCTIONS --------------------

    def _baseline_graph(self, show: int = 0, grid=None) -> None:
        """
        Plot the FHR signal with background regions based on baseline classification.

        This method displays the Fetal Heart Rate (FHR) signal and highlights time
        intervals classified into different baseline categories:
            - 0: Normal (green)
            - 1: Suspicious (orange)
            - 2: Pathological (red)

        Optionally integrates into a provided subplot grid and can display the plot immediately.

        Args:
            show (int, optional): If 0, the plot is shown immediately. Defaults to 0.
            grid (matplotlib.gridspec.SubplotSpec, optional): Grid position for subplot.
                If None, a standalone figure is created.

        Returns:
            None
        """

        t = self._time
        baseline_labels = self._baseline_labels

        # Set up the figure and axis for plotting
        if grid == None:
            _, ax = plt.subplots(figsize=(8, 3))
        else:
            ax = plt.subplot(grid)

        # Plot the FHR signal
        ax.set_title("BASELINE")
        ax.plot(t, self.fhr, color="black")

        # Highlight different baseline classification zones
        ax.fill_between(
            t, 0, 250, where=(baseline_labels == 0), color="lime", alpha=0.5
        )  # Normal
        ax.fill_between(
            t, -1, 250, where=(baseline_labels == 1), color="darkorange", alpha=0.5
        )  # Suspicious
        ax.fill_between(
            t, -1, 250, where=(baseline_labels == 2), color="red", alpha=0.5
        )  # Pathological

        # Configure axis ticks and labels
        plt.xticks(np.linspace(t.min(), t.max(), 5))
        plt.ylabel("FHR (bpm)")
        plt.xlabel("Time (sec)")
        plt.ylim(-1, 250)

        # Show the plot if requested
        if show == 0:
            plt.show()

    def _decelerations_graph(self, show: int = 0, grid=None) -> None:
        """
        Plot the fetal heart rate (FHR) signal and highlight deceleration classifications.

        This method visualizes decelerations using color-coded regions:
            - Green: Normal (label 0)
            - Orange: Suspicious (label 1)
            - Red: Pathological (label 2)

        Optionally supports integration into subplot grids.

        Args:
            show (int): If 0, display the plot immediately using plt.show().
            grid (matplotlib.gridspec.SubplotSpec or None): Optional subplot configuration.

        Returns:
            None
        """

        t = self._time
        s1 = self.fhr
        s2 = self._decelerations_labels

        UC = self.uc  # TODO: Quitar

        # Create new figure and axis if no grid is specified
        if grid == None:
            # _, ax = plt.subplots(figsize=(8, 3))
            _, ax = plt.subplots(figsize=(15, 5))
        else:
            ax = plt.subplot(grid)

        ax.set_title("DECELERATIONS")
        ax.plot(t, s1, color="black")

        # if UC is not None:
        #     ax.plot(t, UC, color="grey")

        # Highlight different deceleration categories
        ax.fill_between(t, 0, 250, where=(s2 == 0), color="lime", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 1), color="darkorange", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 2), color="red", alpha=0.5)

        plt.xticks(np.linspace(t.min(), t.max(), 5))

        plt.ylabel("FHR (bpm)")
        plt.xlabel("Time (sec)")
        plt.ylim(-1, 250)

        # Display plot if requested
        if show == 0:
            plt.show()

    def _variability_graph(
        self,
        show: int = 0,
        grid=None,
    ) -> None:
        """
        Plot the fetal heart rate (FHR) signal with background regions based on variability classification.

        This method displays the FHR signal and highlights time intervals classified into different variability categories:
            - 0: Normal (green)
            - 1: Suspicious (orange)
            - 2: Pathological (red)

        Optionally integrates into a provided subplot grid and can display the plot immediately.

        Args:
            show (int, optional): If 0, display the plot immediately. Defaults to 0.
            grid (matplotlib.gridspec.SubplotSpec or None, optional): Grid position for subplot.
                If None, a standalone figure is created.

        Returns:
            None
        """

        s2 = self._variability_labels
        t = self._time

        # Setup figure and axis depending on grid argument
        if grid == None:
            # _, ax = plt.subplots(figsize=(8, 3))
            _, ax = plt.subplots(figsize=(15, 5))
        else:
            ax = plt.subplot(grid)

        ax.set_title("VARIABILITY")

        # Plot the FHR signal in black
        ax.plot(t, self.fhr, color="black")

        # Highlight regions based on variability classification
        ax.fill_between(t, 0, 250, where=(s2 == 0), color="lime", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 1), color="darkorange", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 2), color="red", alpha=0.5)

        # Configure axes ticks and labels
        plt.xticks(np.linspace(t.min(), t.max(), 5))
        plt.ylabel("FHR (bpm)")
        plt.xlabel("Time (sec)")
        plt.ylim(-1, 250)

        # TODO 1: Borrar
        plt.savefig("inc_var_cond.png", dpi=300, bbox_inches="tight")

        # Show plot immediately if requested
        if show == 0:
            plt.show()

    def _conclusion_graph(self) -> None:
        """
        Plot a comprehensive figure showing baseline, decelerations, variability, and the final conclusion.

        The plot is arranged in a 3x2 grid with:
            - Baseline classification plot (top-left)
            - Decelerations classification plot (middle-left)
            - Variability classification plot (bottom-left)
            - Final conclusion plot (middle-right)

        The final conclusion plot overlays the FHR signal with colored regions indicating:
            - 0: Normal (green)
            - 1: Suspicious (orange)
            - 2: Pathological (red)

        Returns:
            None
        """

        the_grid = GridSpec(3, 2)
        grid_b = the_grid[0, 0]
        grid_d = the_grid[1, 0]
        grid_v = the_grid[2, 0]
        show = 1

        plt.figure(figsize=(14, 12))

        # Plot baseline, decelerations, and variability in respective subplot positions
        self._baseline_graph(show, grid_b)
        self._decelerations_graph(show, grid_d)
        self._variability_graph(show, grid_v)

        t = self._time
        s1 = self.fhr
        s2 = self._conclusion_labels

        # Create subplot for final conclusion
        ax = plt.subplot(the_grid[1, 1])
        ax.set_title("CONCLUSION")

        # Plot FHR signal
        ax.plot(t, s1, color="black")

        # Fill background with color-coded conclusion labels
        ax.fill_between(t, 0, 250, where=(s2 == 0), color="lime", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 1), color="darkorange", alpha=0.5)
        ax.fill_between(t, -1, 250, where=(s2 == 2), color="red", alpha=0.5)

        # Configure ticks and labels
        plt.xticks(np.linspace(t.min(), t.max(), 5))
        plt.ylabel("FHR (bpm)")
        plt.xlabel("Time (sec)")
        plt.ylim(-1, 250)

        # Adjust vertical spacing between subplots
        plt.subplots_adjust(hspace=0.3)

        # Show the full figure
        plt.show()

    ## -----------------------------------------------------
    ## --------------- AUXILIARY FUNCTIONS -----------------

    def _get_baseline(self, window_min_size: float, center: bool = False) -> pd.Series:
        """
        Compute the rolling baseline of the fetal heart rate (FHR) signal.

        The baseline is calculated as the rolling mean over a specified window in minutes.
        It smooths the FHR signal to identify the underlying baseline trend.

        Args:
            window_min_size (float): Window size in minutes to compute the rolling mean.
            center (bool): If True, the rolling window is centered. If False (default),
                        the window is right-aligned.

        Returns:
            pd.Series: Rolling baseline values for the FHR signal, with NaNs where insufficient data.
        """

        window_size = int(window_min_size * 60 * self.frequency)
        min_valid = int(window_size * self.NOT_NAN_PERC)

        # Compute rolling mean (baseline) over the defined time window
        return self.fhr.rolling(
            window=window_size,
            min_periods=min_valid,
            center=center,
        ).mean()

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
        self, fhr: pd.Series, uc: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Removes trailing NaN values from the end of two aligned pandas Series (fhr and uc).

        Args:
            fhr (pd.Series): Fetal heart rate Series that may contain trailing NaNs.
            uc (pd.Series): Uterine contraction Series, assumed to be aligned with fhr.

        Returns:
            Tuple[pd.Series, pd.Series]: New Series with trailing NaNs removed from both fhr and uc.
        """
        if not isinstance(fhr, pd.Series):
            raise TypeError("La entrada debe ser un pandas.Series")

        # Get the index of the last non-NaN value
        last_valid_index = fhr.last_valid_index()

        if last_valid_index is None:
            # If all values are NaN, return an empty Series with same dtype
            return fhr.iloc[0:0], uc.loc[0:0]

        # Slice both Series up to the last valid index (inclusive)
        return fhr.loc[:last_valid_index], uc.loc[:last_valid_index]

    @staticmethod
    def _replace_gaps(signal: pd.Series, threshold: int) -> pd.Series:
        """
        Replace small gaps (NaN segments) in a signal by interpolation.

        This method identifies continuous NaN segments (gaps) in the given signal.
        If the length of a gap is less than or equal to the specified threshold,
        the gap is interpolated. Otherwise, the gap is left as NaN.

        Args:
            signal (pd.Series): Time series signal containing NaN values to be processed.
            threshold (int): Maximum length of a gap that can be interpolated.

        Returns:
            pd.Series: Signal with small gaps interpolated and larger ones left as NaN.
        """

        # Retrieve the gaps (NaN segments) from the signal and iterate over them
        gaps = CTG._get_nan_series_from_series(signal)  # DEBUG

        for _, row in CTG._get_nan_series_from_series(signal).iterrows():

            # If the gap length is less than or equal to the threshold, interpolate it
            # Otherwise, leave the gap as NaN
            if row["Length"] <= threshold:

                # Define the start and end points of the gap with a buffer
                # for better interpolation
                start = row["Start Index"] - 0.25
                end = start + (row["Length"] * 0.25)

                # Interpolate the values within the defined range
                interpolated_part = signal.loc[start:end].interpolate()

                # Replace the original gap with the interpolated values
                signal[start:end] = interpolated_part

        return signal

    def _get_deceleration_condition(
        self,
        amplitude_deceleration: int = 15,
        time_in_deceleration: int = 15,
    ) -> pd.Series:
        """
        Detect fetal heart rate (FHR) decelerations based on amplitude and duration criteria.

        A deceleration is defined as a drop in FHR below the baseline with:
            - Amplitude greater than `amplitude_deceleration` bpm (default 15 bpm)
            - Duration longer than `time_in_deceleration` seconds (default 15 seconds)

        The function uses a moving baseline and identifies sequences where the FHR
        stays below the threshold for the required time. It also excludes false positives
        where the heart rate does not actually decrease during the interval.

        Args:
            amplitude_deceleration (int, optional): Minimum amplitude drop in bpm to qualify (default 15).
            time_in_deceleration (int, optional): Minimum duration in seconds below baseline to qualify (default 15).

        Returns:
            pd.Series: Boolean mask indexed as the original FHR signal,
                    with True values indicating detected decelerations.
        """

        # The difference between the baseline and the FHR is calculated; this difference must be greater than 15.
        # This baseline is calculated as a moving average with 10-minute windows.
        is_deceleration = self._baseline - self.fhr > amplitude_deceleration

        # Groups consecutive sequences with the same value in is_deceleration, assigning a unique ID to each group
        group_id = (is_deceleration != is_deceleration.shift()).cumsum()
        grouped = is_deceleration.groupby(group_id)

        # Minimum time that deceleration must take
        min_samples = int(time_in_deceleration * self.frequency)
        mask = grouped.transform(
            lambda x: x.sum() if x.all() and len(x) > min_samples else 0
        ).astype(bool)

        # -- Prevent it from detecting accelerations as decelerations --

        # Select only the True values from the mask
        true_serie = mask[mask]

        # If there are no decelerations, return the original mask
        if len(true_serie) == 0:
            return mask

        # Calculate differences between consecutive indices
        diffs = np.diff(true_serie.index)  # resto los índices

        # Identify new groups where the time gap between indices
        # exceeds threshold (e.g., 0.25s if frequency is 4 Hz)
        # (if it does not exceed 1/freq they are consecutive positions)
        new_group = np.insert(diffs > (1 / self.frequency), 0, True).cumsum()

        # Group the True values into consecutive blocks
        grouped = true_serie.groupby(new_group)

        # Iterate through each group of consecutive True values
        for _, group in grouped:

            # If the group does not start at index 0, go 0.25s back to find the deceleration start
            if group.index[0] != 0:
                start = group.index[0] - (0.25)
            else:
                start = group.index[0]

            # The end is the last index of the group
            end = group.index[-1]

            # If heart rate did not drop (not a true deceleration), remove this block from the mask
            if self.fhr.loc[start] - self.fhr.loc[end] < 0:
                mask.loc[start:end] = False

        return mask

    @staticmethod
    def _extend_condition_back(condition: pd.Series, window: int) -> pd.Series:
        """
        Extends True values backwards in a boolean pandas Series condition.

        For each new True event in the Series, the function sets to True
        all values in the preceding `window` length interval before the event.

        Args:
            condition (pd.Series): Boolean Series where True values mark events.
            window (int): Number of samples to extend backwards.

        Returns:
            pd.Series: Modified Series with True values extended backwards.
        """

        for i in range(1, len(condition)):
            # Check if current value is True and previous is False or NaN (start of a new True event)
            if condition.iloc[i] == True and (
                condition.iloc[i - 1] == False
                or np.isnan(condition.iloc[i - 1]) == True
            ):

                # Calculate start index for extension
                start_window = i - window + 1

                if start_window < 0:
                    # CAMBIO: condition.iloc[0:i] = np.ones(len(range(i)))
                    # If window goes beyond start, set all values from 0 up to i to True
                    condition.iloc[0:i] = True

                else:
                    # CAMBIO: condition.iloc[start_window:i] = np.ones(window-1)
                    # Otherwise, set values from start_window to i-1 to True
                    condition.iloc[start_window:i] = True

        return condition

    def _get_repetitive_deceleration_condition(
        self,
        time_in_rep_dec: int,
        correlation_threshold: float = -0.4,
        center: bool = False,
    ) -> np.ndarray:
        """
        Identify periods of repetitive decelerations based on negative correlation
        between FHR and uterine contractions (UC).

        This method computes a rolling correlation between the FHR signal and UC signal.
        When the correlation is below a specified negative threshold (e.g., -0.4), it is
        considered indicative of decelerations linked to contractions, which may suggest
        fetal compromise.

        The result is extended backwards in time over the specified window to capture
        the full temporal context of the condition.

        Args:
            time_in_rep_dec (int): Length of the rolling window in minutes for correlation analysis.
            correlation_threshold (float): Threshold for negative correlation to define a deceleration.
            center (bool): Whether to center the rolling window or align it to the right.

        Returns:
            np.ndarray: Boolean array indicating periods with repetitive deceleration conditions.
                        NaN where correlation could not be calculated due to missing data.
        """

        # Safety check: if there is no uterine contraction data, return an array of NaNs
        if self.uc is None:
            return np.nan * np.zeros(len(self.fhr))

        # Create rolling windows on FHR signal for correlation calculation
        fhr_rolling = self.fhr.rolling(
            window=time_in_rep_dec * self.frequency * 60,
            min_periods=int(self.NOT_NAN_PERC * time_in_rep_dec * self.frequency * 60),
            center=center,
        )

        # Calculate rolling correlation of FHR with the full UC signal
        corr_FHR_UC_X_min = fhr_rolling.corr(self.uc)

        # Condition is true where correlation drops below the threshold
        repetitive_deceleration_condition = corr_FHR_UC_X_min < correlation_threshold

        # Preserve NaNs where correlation could not be computed
        repetitive_deceleration_condition[np.isnan(corr_FHR_UC_X_min)] = np.nan

        # Extend the condition backward in time to cover the whole window
        repetitive_deceleration_condition = self._extend_condition_back(
            repetitive_deceleration_condition,
            window=60 * time_in_rep_dec * self.frequency,
        )

        # TODO 1: Borrar
        # if (
        #     time_in_rep_dec == 30
        #     and any(repetitive_deceleration_condition.fillna(False))
        #     and self.id != 1220
        # ):
        #     print("Se encontró un valor True. Deteniendo ejecución. id:", self.id)
        #     sys.exit()

        return repetitive_deceleration_condition

    def _get_late_deceleration_condition(
        self,
        dec_cond: np.ndarray,  # DUDA: Seguro que es un numpy array ??
        center: bool = False,
        time_in_late_dec_cond: int = 30,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
    ) -> np.ndarray:
        """
        Detect late decelerations based on duration and timing relative to uterine contractions.

        A late deceleration is defined as a deceleration that:
        - Lasts at least `time_in_late_dec_cond` seconds,
        - Overlaps with a uterine contraction (shifted in time),
        - Starts after the onset of the contraction (via time shift).

        This method uses a binary deceleration condition (`dec_cond`) and
        identifies where sustained deceleration aligns with contraction patterns.

        Args:
            dec_cond (np.ndarray): Boolean array indicating deceleration periods.
            center (bool): Whether to center rolling windows.
            time_in_late_dec_cond (int): Minimum duration (in seconds) of a deceleration to qualify as "late".
            contraction_time_shift (int): Delay (in seconds) to align UC signal to physiological response.
            contraction_value_threshold (int): Minimum amplitude over UC baseline to detect a contraction.
            contraction_duration_threshold (int): Duration (in seconds) for contraction detection.
            contraction_baseline_duration (int): Duration (in seconds) to compute dynamic UC baseline.

        Returns:
            np.ndarray: Boolean array indicating periods of late decelerations.
                        NaN values indicate insufficient data for reliable computation.
        """

        # Return zeros if there is no uterine contraction data
        if self.uc is None:
            return np.zeros(len(self.fhr))

        # Check if decelerations persist for at least `time_in_late_dec_cond` seconds
        max_dec = (
            (1 * dec_cond)
            .rolling(
                window=self.frequency * time_in_late_dec_cond,
                min_periods=int(
                    self.NOT_NAN_PERC * self.frequency * time_in_late_dec_cond
                ),
                center=center,
            )
            .max()
        )
        min_dec = (
            (1 * dec_cond)
            .rolling(
                window=self.frequency * time_in_late_dec_cond,
                min_periods=int(
                    self.NOT_NAN_PERC * self.frequency * time_in_late_dec_cond
                ),
                center=center,
            )
            .min()
        )

        # True if the entire 60-second window is part of a deceleration (value stays 1)
        # (We have a deceleration that lasts one minute)
        aux_late_deceleration_condition = (max_dec == 1) & (min_dec == 1)

        # Obtain the information from the contractions
        shift_contr_condition = self._get_contraction_condition(
            time_shift=contraction_time_shift,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=center,
        )

        # Late deceleration = long deceleration that overlaps with contraction (and starts after)
        late_deceleration_condition = (aux_late_deceleration_condition == True) & (
            shift_contr_condition == True
        )

        # Keep NaNs where the deceleration duration could not be reliably computed
        late_deceleration_condition[np.isnan(min_dec)] = np.nan

        # Extend the condition backward to cover the full deceleration period
        late_deceleration_condition = self._extend_condition_back(
            condition=late_deceleration_condition,
            window=time_in_late_dec_cond * self.frequency,
        )

        return late_deceleration_condition

    def _get_contraction_condition(
        self,
        center: bool = False,
        time_shift: int = 20,
        value_threshold: int = 10,
        duration_threshold: int = 20,
        baseline_duration: int = 120,
    ) -> np.ndarray:
        """
        Detect uterine contractions based on a sustained rise in UC signal above a dynamic baseline.

        This method analyzes the uterine contraction (UC) signal by:
        - Shifting the signal to account for physiological delay,
        - Calculating a rolling baseline over a long window,
        - Detecting short-duration rises above this baseline (contractions),
        - Applying a value and duration threshold,
        - Optionally extending detected contraction periods backward in time.

        Args:
            center (bool): Whether to center the rolling windows (affects alignment).
            time_shift (int): Time delay (in seconds) to shift the UC signal to account for physiological lag.
            value_threshold (int): Minimum increase over the baseline to be considered a contraction.
            duration_threshold (int): Minimum duration (in seconds) that the increase must persist.
            baseline_duration (int): Duration (in seconds) for calculating the dynamic UC baseline.

        Returns:
            np.ndarray: Boolean array indicating periods of uterine contractions.
                        NaN values indicate insufficient data for computation.
        """

        UC = self.uc

        # Safety check: if UC signal is unavailable, return an array of NaNs
        if UC is None:
            return np.nan * np.zeros(len(self.fhr))

        freq = self.frequency

        # Shift the UC signal forward to account for delay between contraction and FHR response
        UC = self.uc.shift(time_shift * freq)

        # Compute a rolling minimum over a long window to serve as a dynamic UC baseline
        base_min_UC = UC.rolling(
            window=baseline_duration * freq,
            min_periods=int(self.NOT_NAN_PERC * baseline_duration * freq),
            center=center,
        ).min()

        # Subtract the dynamic baseline to get the relative UC increase
        diff_UC_base = UC - base_min_UC

        # Check if the relative increase is sustained over a short duration (i.e., potential contraction)
        min_diff_uc = diff_UC_base.rolling(
            window=int(duration_threshold * freq),
            min_periods=int(self.NOT_NAN_PERC * duration_threshold * freq),
            center=center,
        ).min()

        # Condition is True where sustained rise above baseline exceeds the value threshold
        contraction_condition = min_diff_uc > value_threshold

        # Preserve NaNs where contraction could not be evaluated
        contraction_condition[np.isnan(min_diff_uc)] = np.nan

        # Extend the detected contraction period backward in time by the duration threshold
        contraction_condition = self._extend_condition_back(
            condition=contraction_condition, window=int(duration_threshold * freq)
        )

        return contraction_condition

    def _get_prolonged_deceleration_condition(
        self,
        dec_cond: pd.Series,
        time_in_prolonged_dec: int = 3,
        center: bool = False,
    ) -> np.ndarray:
        """
        Identify prolonged decelerations based on sustained deceleration over a specified duration.

        A prolonged deceleration is defined as a continuous deceleration that lasts
        at least `time_in_prolonged_dec` minutes. This method uses a rolling window to
        verify if the deceleration condition remains True throughout the entire window.

        Args:
            dec_cond (np.ndarray): Boolean array indicating deceleration events (1 = deceleration, 0 = no deceleration).
            time_in_prolonged_dec (int): Minimum duration (in minutes) for a deceleration to be considered prolonged.
            center (bool): Whether to center the rolling window or align it to the right.

        Returns:
            np.ndarray: Boolean array indicating periods of prolonged decelerations.
                        NaNs indicate periods where insufficient data prevented evaluation.
        """

        freq = self.frequency

        # Compute rolling max and min over the window to detect continuous deceleration
        max_dec_X_min = dec_cond.rolling(
            time_in_prolonged_dec * 60 * freq,
            min_periods=int(self.NOT_NAN_PERC * time_in_prolonged_dec * 60 * freq),
            center=center,
        ).max()
        min_dec_X_min = dec_cond.rolling(
            time_in_prolonged_dec * 60 * freq,
            min_periods=int(self.NOT_NAN_PERC * time_in_prolonged_dec * 60 * freq),
            center=center,
        ).min()

        # A prolonged deceleration is one where the full window is continuously True (1)
        prolonged_deceleration_condition = (max_dec_X_min == 1) & (min_dec_X_min == 1)

        # Preserve NaNs where the window did not meet the minimum data requirement
        prolonged_deceleration_condition[np.isnan(min_dec_X_min)] = np.nan

        # Extend the condition backward to cover the duration of the prolonged deceleration
        prolonged_deceleration_condition = self._extend_condition_back(
            prolonged_deceleration_condition, window=time_in_prolonged_dec * 60 * freq
        )

        # TODO 1: Borrar
        # if time_in_prolonged_dec == 5 and any(
        #     prolonged_deceleration_condition.fillna(False)
        # ):
        #     print("Se encontró un valor True. Deteniendo ejecución. id:", self.id)
        #     sys.exit()

        return prolonged_deceleration_condition

    def _get_reduced_variability_condition(
        self,
        center: bool = False,
        amplitude_dec_cond: int = 15,
        time_in_deceleration: int = 15,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
    ) -> np.ndarray:
        """
        Detect periods of reduced fetal heart rate (FHR) variability, either during baseline
        or associated with decelerations.

        Reduced variability is a potential indicator of fetal distress. This method:
        - Computes short-term variability using a 1-minute window (max - min),
        - Detects significant reduction in variability sustained over:
            • Long baseline periods (e.g. 50 min),
            • Short deceleration periods (e.g. 3 min),
        - Combines both sources of reduced variability for final classification.

        Args:
            center (bool): Whether to center the rolling windows.
            amplitude_dec_cond (int): Minimum amplitude to define a deceleration.
            time_in_deceleration (int): Minimum duration (in seconds) for a deceleration.
            red_var_duration_baseline (int): Duration (in minutes) for evaluating reduced variability at baseline.
            red_var_duration_deceleration (int): Duration (in minutes) for reduced variability during decelerations.
            red_var_bandwith (int): Threshold for defining reduced variability (in bpm).

        Returns:
            np.ndarray: Boolean array indicating reduced variability periods.
        """

        FHR = self.fhr
        freq = self.frequency

        # Calculate 1-minute rolling max and min to define short-term FHR variability
        max_FHR_1_min = FHR.rolling(
            window=60 * freq,
            min_periods=int(self.NOT_NAN_PERC * 60 * freq),
            center=center,
        ).max()
        min_FHR_1_min = FHR.rolling(
            window=60 * freq,
            min_periods=int(self.NOT_NAN_PERC * 60 * freq),
            center=center,
        ).min()

        # Variability (bandwidth) over 1-minute windows
        FHR_bandwith_1_min = max_FHR_1_min - min_FHR_1_min

        # Get deceleration periods
        dec_cond = self._get_deceleration_condition(
            amplitude_deceleration=amplitude_dec_cond,
            time_in_deceleration=time_in_deceleration,
        )

        # Compute max variability over long durations to detect prolonged reduction
        max_FHR_bandwith_50_min = FHR_bandwith_1_min.rolling(
            window=red_var_duration_baseline * 60 * freq,
            min_periods=int(self.NOT_NAN_PERC * red_var_duration_baseline * 60 * freq),
            center=center,
        ).max()

        # Maximum variability during shorter deceleration windows
        max_FHR_bandwith_3_min = FHR_bandwith_1_min.rolling(
            window=red_var_duration_deceleration * 60 * freq,
            min_periods=int(
                self.NOT_NAN_PERC * red_var_duration_deceleration * 60 * freq
            ),
            center=center,
        ).max()

        # Reduced variability in baseline segments (long duration)
        red_var_base = max_FHR_bandwith_50_min < red_var_bandwith
        red_var_base[np.isnan(max_FHR_bandwith_50_min)] = np.nan
        red_var_base = self._extend_condition_back(
            red_var_base, window=red_var_duration_baseline * 60 * freq
        )

        # Reduced variability during decelerations (short duration)
        red_var_dec = (max_FHR_bandwith_3_min < red_var_bandwith) & (dec_cond == True)
        red_var_base[np.isnan(max_FHR_bandwith_3_min)] = np.nan
        red_var_dec = self._extend_condition_back(
            red_var_dec, window=red_var_duration_deceleration * 60 * freq
        )

        # Combine both baseline and deceleration-related reduced variability
        reduced_variability_condition = (red_var_base == True) | (red_var_dec == True)

        return reduced_variability_condition

    def _get_increased_variability_condition(
        self,
        window_var: int = 1,
        center: bool = False,
        min_time: int = 30,
        bandwidth_min: float = 25,
    ) -> pd.Series:
        """
        Detect periods of increased fetal heart rate (FHR) variability,
        characterized by wide oscillations in FHR sustained over time.

        This function computes the short-term variability (1-minute bandwidth)
        and identifies when this variability remains consistently above a threshold
        for a given duration.

        Args:
            window_var (int): Size (in minutes) of the rolling window used to compute 1-minute variability.
            center (bool): Whether the rolling window should be centered.
            min_time (int): Minimum duration (in minutes) the high variability must be sustained.
            bandwidth_min (float): Threshold in bpm for defining "increased" variability.

        Returns:
            pd.Series: Boolean series indicating when increased variability is present.
        """

        freq = self.frequency

        # Calculate 1-minute rolling max and min of the FHR signal
        max_FHR_1_min = self.fhr.rolling(
            window=60 * freq * window_var,
            min_periods=int(self.NOT_NAN_PERC * 60 * freq * window_var),
            center=center,
        ).max()

        min_FHR_1_min = self.fhr.rolling(
            window=60 * freq * window_var,
            min_periods=int(self.NOT_NAN_PERC * 60 * freq * window_var),
            center=center,
        ).min()

        # Compute variability (bandwidth = max - min) over 1-minute windows
        FHR_bandwith_1_min = max_FHR_1_min - min_FHR_1_min

        # Compute the *minimum* bandwidth over a 30-minute window
        # This detects sustained periods of wide variability
        min_FHR_bandwith_30_min = FHR_bandwith_1_min.rolling(
            window=min_time * 60 * freq,
            min_periods=int(self.NOT_NAN_PERC * min_time * 60 * freq),
            center=center,
        ).min()

        # Identify periods where even the minimum variability in the window exceeds the threshold
        increased_variability_condition = min_FHR_bandwith_30_min > bandwidth_min

        # Set NaN where rolling window results are unreliable (not enough data)
        increased_variability_condition[np.isnan(min_FHR_bandwith_30_min)] = np.nan

        # Extend the condition backward to cover the full 30-minute window
        increased_variability_condition = self._extend_condition_back(
            increased_variability_condition, window=min_time * 60 * freq
        )

        return increased_variability_condition

    def _get_variable_deceleration_condition(self, center: bool = False) -> pd.Series:
        """
        Detect variable decelerations in the fetal heart rate (FHR) signal.

        Variable decelerations are V-shaped decelerations characterized by a rapid drop
        (onset to nadir in <30s) and rapid recovery, often occurring within 60 seconds.

        This method detects such events by:
            - Looking for decelerations that start and end within 60 seconds,
            - Excluding those that are prolonged, late, or early decelerations.

        Args:
            center (bool): Whether the rolling windows are centered.

        Returns:
            pd.Series: Boolean series indicating variable deceleration periods.
                    NaN values indicate insufficient data for evaluation.
        """

        FHR = self.fhr
        freq = self.frequency

        # Variable Decelerations: V-shaped decelerations that exhibit a rapid drop (onset to nadir in <30s)
        # followed by a rapid recovery to the baseline
        # we assume that in 60s there are a rapid drop and a rapid recovery, so it is a deceleration <60s

        # Get general deceleration condition
        dec_cond = self._get_deceleration_condition()

        # Rolling max and min over 60 seconds of deceleration condition
        max_dec_60_sec = (
            (1 * dec_cond)
            .rolling(
                window=60 * freq,
                min_periods=int(self.NOT_NAN_PERC * 60 * freq),
                center=center,
            )
            .max()
        )
        min_dec_60_sec = (
            (1 * dec_cond)
            .rolling(
                window=60 * freq,
                min_periods=int(self.NOT_NAN_PERC * 60 * freq),
                center=center,
            )
            .min()
        )
        dec_cond_bandwith_60_sec = max_dec_60_sec - min_dec_60_sec
        # maximum is 1 (could be nan), minimum is 0 -> there are a rapid drop and recovery in 60s

        # Deceleration must rise and fall within 60s (i.e., variability in condition within that window)
        variable_deceleration_condition = (
            (max_dec_60_sec == 1) & (min_dec_60_sec == 0) & (dec_cond == True)
        )

        # Remove decelerations that are classified as late, prolonged, or early
        late_dec_cond = self._get_late_deceleration_condition()
        prol_dec_3_min_cond = self._get_prolonged_deceleration_condition(minutes=3)
        early_dec_cond = self._get_early_deceleration_condition()

        variable_deceleration_condition = (
            (variable_deceleration_condition == True)
            & (late_dec_cond != True)
            & (prol_dec_3_min_cond != True)
            & (early_dec_cond != True)
        )

        # Set NaN where rolling statistics were not computed
        variable_deceleration_condition[np.isnan(min_dec_60_sec)] = np.nan

        # variable_deceleration_condition = extend_condition_back(variable_deceleration_condition, window=minutes*60*freq)

        return variable_deceleration_condition

    def _get_early_deceleration_condition(
        self,
        contraction_time_shift: int = 20,
        contraction_value_threshold: int = 10,
        contraction_duration_threshold: int = 20,
        contraction_baseline_duration: int = 120,
        center: bool = False,
    ) -> pd.Series:
        """
        Detect early decelerations in the fetal heart rate (FHR) signal.

        Early decelerations are gradual decreases in FHR (onset to nadir ≥ 30s)
        that return to baseline, and the nadir occurs simultaneously with the
        peak of a uterine contraction.

        This method assumes early decelerations last longer than 60 seconds
        and overlap in time with uterine contractions.

        Args:
            contraction_time_shift (int): Time (in seconds) to shift UC signal to align with expected FHR response.
            contraction_value_threshold (int): Minimum amplitude above UC baseline to define a contraction.
            contraction_duration_threshold (int): Minimum duration (in seconds) of a contraction.
            contraction_baseline_duration (int): Time (in seconds) over which to compute the UC baseline.
            center (bool): Whether to center rolling windows.

        Returns:
            pd.Series: Boolean series indicating early deceleration periods.
                    NaN values represent regions where the condition could not be evaluated.
        """

        FHR = self.fhr
        UC = self.uc
        freq = self.frequency

        # Early Decelerations: : Decelerations that are gradual (onset to nadir ≥30s) that return to the baseline.
        # The nadir occurs with the peak of a contraction
        # we assume that it is a deceleration >60s

        # Return NaNs if uterine contraction data is unavailable
        if UC is None:
            return np.nan * np.zeros(len(FHR))

        # Get deceleration condition mask (boolean)
        dec_cond = self._get_deceleration_condition()

        # Rolling max and min over 60 seconds to assess deceleration persistence
        max_dec_60_sec = (
            (1 * dec_cond)
            .rolling(
                window=60 * freq,
                min_periods=int(self.NOT_NAN_PERC * 60 * freq),
                center=center,
            )
            .max()
        )
        min_dec_60_sec = (
            (1 * dec_cond)
            .rolling(
                window=60 * freq,
                min_periods=int(self.NOT_NAN_PERC * 60 * freq),
                center=center,
            )
            .min()
        )
        dec_cond_bandwith_60_sec = max_dec_60_sec - min_dec_60_sec

        # there are only 1s, maximum is 1 (could be nan) -> duration >60s
        # Deceleration condition must be continuously True over 60s (bandwidth = 0)
        aux_early_deceleration_condition = (max_dec_60_sec == 1) & (min_dec_60_sec == 1)

        # TODO: Comprobar si estas contraciones deberian estar desplazadas también
        # Get contraction mask (True when contractions are detected)
        contr_condition = self._get_contraction_condition(
            time_shift=contraction_time_shift,
            value_threshold=contraction_value_threshold,
            duration_threshold=contraction_duration_threshold,
            baseline_duration=contraction_baseline_duration,
            center=center,
        )

        # Early deceleration = prolonged deceleration that overlaps with contraction
        early_deceleration_condition = (aux_early_deceleration_condition == True) & (
            contr_condition == True
        )

        # print(len(early_deceleration_condition), len(min_dec_60_sec))
        # Assign NaNs to preserve locations where min_dec_60_sec is invalid
        early_deceleration_condition[np.isnan(min_dec_60_sec)] = np.nan

        # Extend the condition backward to capture full duration of deceleration
        early_deceleration_condition = self._extend_condition_back(
            early_deceleration_condition, window=60 * freq
        )

        return early_deceleration_condition
