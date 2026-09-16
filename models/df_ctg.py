"""Dataset-level utilities for CTG analysis.

This module manages collections of CTG recordings, applies preprocessing and
rule-based analysis, summarizes classification labels, and provides evaluation
and visualization utilities used in the project.
"""

import sys
import copy
import commentjson
import numpy as np
import pandas as pd
from .ctg import CTG
import seaborn as sns
import tensorflow as tf
from collections import Counter
from config.config_methods import AppConfig
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Union, Optional, Dict, Tuple
from sklearn.feature_selection import mutual_info_classif

import matplotlib.pyplot as plt


class DF_CTG:

    def __init__(
        self,
        config_file: str,
        additional_config={},
    ):

        self.ctgs = []

        self.additional_config = additional_config

        try:
            with open(config_file, "r", encoding="utf-8") as file:
                config = AppConfig(commentjson.load(file))
                self.config = config

        except Exception as e:
            print("ERROR! It was not possible to read the config file")
            print(f"Error details: {e}")
            self.config = None

        # Rule-analysis results are stored after evaluation.

    def add_ctg(self, ctg: CTG) -> None:
        """
        Add a CTG (Cardiotocography) object to the DF_CTG group.
        """
        ctg.group = self  # Ahora forma parte de una familia 🥲
        self.ctgs.append(ctg)

    def read_ctgs_from_files(self) -> None:
        # Track records skipped or successfully loaded.
        num_files_no_exist = 0
        num_ctg_created = 0
        max_num_data = (
            self.config.freq * self.config.read.cut_min_read * 60
        )  # 4 datos/sec * 60 sec/min * x min

        fhr_df = pd.read_csv(
            self.config.data_folder_path + "fhr.csv", index_col=0, compression="gzip"
        )

        uc_df = pd.read_csv(
            self.config.data_folder_path + "uc.csv", index_col=0, compression="gzip"
        )

        clinical_df = pd.read_csv(
            self.config.data_folder_path + "clinical.csv",
            index_col=0,
            compression="gzip",
        )

        for index in fhr_df.index:
            missing_sources = []
            if index not in uc_df.index:
                missing_sources.append("uc")
            if index not in clinical_df.index:
                missing_sources.append("clinical information")

            if missing_sources:
                num_files_no_exist += 1
                missing_str = " and the ".join(missing_sources)
                print(
                    f"{num_files_no_exist}.- The {missing_str} for the ctg ({index}) does not exist. Skipping iteration."
                )
                continue

            fhr = fhr_df.loc[index].iloc[-max_num_data:]
            uc = uc_df.loc[index].iloc[-max_num_data:]
            clinical_data = clinical_df.loc[index]

            ph = float(clinical_data["ph"])

            self.new_ctg(fhr, uc, clinical_data, ph, index)
            num_ctg_created += 1

        print("Reading Completed: ", num_ctg_created, "ctg created")

    def new_ctg(
        self,
        fhr: pd.Series,
        uc: pd.Series,
        clinical_data: pd.Series,
        ph: float,
        id: int,
    ) -> None:
        """
        Create a new CTG (Cardiotocography) object with given FHR and UC signals, and append it to the CTG list.

        This method verifies that the lengths of the fetal heart rate (FHR) and uterine contraction (UC) signals are equal.
        If they differ, it raises a ValueError. Upon successful validation, it creates a CTG object using the provided data
        and internal parameters, then stores the CTG and its pH value.

        Args:
            fhr (list): Fetal Heart Rate signal values.
            uc (list): Uterine Contraction signal values.
            ph (float): pH value associated with this CTG.
            id (int): Identifier for the CTG record.

        Returns:
            None: This method does not return anything.
        """
        # Check that the length of FHR and UC signals are equal
        if len(fhr) != len(uc):
            raise ValueError("ERROR: The size of the FHR and UC must be equal")

        # Create a new CTG instance with given data and class parameters
        ctg = CTG(
            fhr=fhr.to_numpy(),  # We're switching to NumPy because the indices will be changed.
            uc=uc.to_numpy(),
            clinical_data=clinical_data,
            ph=ph,
            id=id,
            group=self,
        )

        # Append the new CTG and its pH value to the lists
        self.ctgs.append(ctg)

    def copy(self):
        return copy.deepcopy(self)

    # ---------------------------------------------------------------------
    # DATASET INSPECTION
    # ---------------------------------------------------------------------

    def len_ctgs(self) -> None:
        """
        Print the distribution of FHR (Fetal Heart Rate) signal lengths among the stored CTGs.

        This method counts how many CTGs have FHR signals of each length and prints the results.

        Args:
            None

        Returns:
            None: This method only prints information and does not return anything.
        """
        # Count the frequency of each FHR length across all CTGs
        lengths = Counter(len(ctg.fhr) for ctg in self.ctgs)

        # Print how many CTGs have each specific FHR length
        for length, amount in lengths.items():
            print(f"{amount} CTGs have an FHR length of {length}")

    def get_ctg_by_id(self, id: str) -> Optional[CTG]:
        """
        Retrieve a CTG object by its identifier.

        This method searches the stored CTGs for one matching the given id.
        If found, it returns the CTG object; otherwise, it prints "NOT FOUND" and returns None.

        Args:
            id (int): The identifier of the CTG to retrieve.

        Returns:
            Optional[CTG]: The CTG object with the matching id, or None if not found.
        """

        for i in range(len(self.ctgs)):
            if self.ctgs[i].id == id:
                return self.ctgs[i]
        print("NOT FOUND")

    # ---------------------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------------------

    def preprocess_ctgs(self, prep_type=None) -> None:
        """
        Preprocess all stored CTG recordings using the configured strategy.

        Recordings that cannot be preprocessed successfully are excluded from
        the collection. If ``prep_type`` is not provided explicitly, the value
        defined in the configuration is used.

        Parameters
        ----------
        prep_type : str or None
            Preprocessing strategy to apply. If None, the configured strategy
            is used.

        Returns
        -------
        None
            The CTG collection is modified in place.
        """

        if "prep_type" in self.additional_config.keys():
            prep_type = self.additional_config["prep_type"]

        else:
            prep_type = self.config.preprocessing.prep_type

        new_ctgs_list = []

        num_error = 0
        id_error = []
        original_size = len(self.get_fhr())

        # Validate the configured preprocessing strategy.
        correct_rules_type = ["FIGO", "FB_FOURIER", "SPLINES"]

        # Reject unsupported preprocessing strategies.
        if prep_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # Preprocess each CTG and retain only valid recordings.
        for ctg in self.ctgs:
            error_code = ctg.preprocess_signal(prep_type=prep_type)

            if error_code == 0:
                new_ctgs_list.append(ctg)
            else:
                num_error = num_error + 1
                id_error.append(ctg.id)

        self.ctgs = new_ctgs_list

        print(
            f"{prep_type} preprocessing completed: {num_error}/{original_size} skipped as invalid (id= {id_error})"
        )

    # ---------------------------------------------------------------------
    # RULE-BASED ANALYSIS
    # ---------------------------------------------------------------------

    def get_labels_rules(
        self,
        # -- rules param --
        rules_type: str,
        center: bool = False,
        preprocess: bool = False,
        prep_cut_time: int = 60,
        prep_max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
        window_time_baseline: int = 10,
        baseline_graph: bool = False,
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
        window_var: int = 1,
        variability_graph: bool = False,
        incr_var_min_time: int = 30,
        incr_var_bandwidth: int = 25,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        conclusion_graph: bool = False,
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """
        Apply labeling rules to each stored CTG and retrieve categorized labels.

        This method applies specified diagnostic rules to all CTG records stored in the instance,
        allowing for preprocessing and multiple configurable parameters related to contraction detection,
        deceleration criteria, variability, and graphical output options.

        Args:
            rules_type (str): The type of rules to apply for labeling.
            center (bool, optional): Whether to center the signal during analysis. Defaults to False.
            get_dicc (bool, optional): If True, returns a dictionary with all category labels; otherwise returns only conclusions. Defaults to False.
            preprocess (bool, optional): Whether to preprocess CTGs before applying rules. Defaults to False.
            prep_cut_time (int, optional): Cut time parameter for preprocessing. Defaults to 60.
            prep_max_size_gaps (int, optional): Maximum size of gaps allowed in preprocessing. Defaults to 15.
            rm_tail_nan (bool, optional): Whether to remove trailing NaN values during preprocessing. Defaults to False.
            window_time_baseline (int, optional): Time window for baseline calculation. Defaults to 10.
            baseline_graph (bool, optional): Whether to generate a graph of the baseline. Defaults to False.
            amplitude_dec_cond (int, optional): Amplitude threshold for deceleration condition. Defaults to 15.
            time_in_deceleration (int, optional): Time in seconds to detect deceleration. Defaults to 15.
            time_in_rep_dec (int, optional): Time for repetitive deceleration detection. Defaults to 30.
            correlation_threshold (float, optional): Threshold for correlation detection. Defaults to -0.4.
            time_in_late_dec_cond (int, optional): Time threshold for late deceleration condition. Defaults to 30.
            contraction_time_shift (int, optional): Time shift applied to contraction signals. Defaults to 20.
            contraction_value_threshold (int, optional): Minimum amplitude to define contraction. Defaults to 10.
            contraction_duration_threshold (int, optional): Minimum duration to define contraction. Defaults to 20.
            contraction_baseline_duration (int, optional): Duration for contraction baseline calculation. Defaults to 120.
            time_in_prolonged_dec (int, optional): Time threshold for prolonged deceleration. Defaults to 3.
            time_in_rep_dec_for_red_var (int, optional): Time for repetitive deceleration impacting reduced variability. Defaults to 20.
            time_patholog_prolonged_dec (int, optional): Time for pathological prolonged deceleration. Defaults to 5.
            deceleration_graph (bool, optional): Whether to generate deceleration graphs. Defaults to False.
            window_var (int, optional): Window size for variability analysis. Defaults to 1.
            variability_graph (bool, optional): Whether to generate variability graphs. Defaults to False.
            incr_var_min_time (int, optional): Minimum time for increased variability detection. Defaults to 30.
            incr_var_bandwidth (int, optional): Bandwidth for increased variability detection. Defaults to 25.
            red_var_duration_baseline (int, optional): Duration for reduced variability in baseline. Defaults to 50.
            red_var_duration_deceleration (int, optional): Duration for reduced variability during deceleration. Defaults to 3.
            red_var_bandwith (int, optional): Bandwidth for reduced variability. Defaults to 5.
            conclusion_graph (bool, optional): Whether to generate conclusion graphs. Defaults to True.

        Returns:
            Union[pd.Series, Dict[str, pd.Series]]: Returns a list of conclusions labels by default.
                                    If get_dicc is True, returns a dictionary with labels
                                    for baseline, deceleration, variability, and conclusion.
        """

        # Apply the rule-based analysis to each stored CTG.
        N = len(self.ctgs)

        base_labels_list = []
        dec_labels_list = []
        var_labels_list = []
        conc_labels_list = []

        # Process each CTG independently.
        for i in range(N):
            # Report progress in place.
            sys.stdout.write("\r" + str(i + 1) + "/" + str(N) + " ")

            # Compute the rule-based labels for the current CTG.
            ctg = self.ctgs[i]
            labels_rules_ctg = ctg.apply_rules(
                rules_type,
                center=center,
                get_dicc=True,
                preprocess=preprocess,
                prep_cut_time=prep_cut_time,
                prep_max_size_gaps=prep_max_size_gaps,
                rm_tail_nan=rm_tail_nan,
                window_time_baseline=window_time_baseline,
                baseline_graph=baseline_graph,
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
                window_var=window_var,
                variability_graph=variability_graph,
                incr_var_min_time=incr_var_min_time,
                incr_var_bandwidth=incr_var_bandwidth,
                red_var_duration_baseline=red_var_duration_baseline,
                red_var_duration_deceleration=red_var_duration_deceleration,
                red_var_bandwith=red_var_bandwith,
                conclusion_graph=conclusion_graph,
            )

            # Store the component-wise labels.
            base_labels_list.append(labels_rules_ctg["baseline"])
            dec_labels_list.append(labels_rules_ctg["deceleration"])
            var_labels_list.append(labels_rules_ctg["variability"])
            conc_labels_list.append(labels_rules_ctg["conclusion"])

        # Cache the labels for later evaluation.
        self._base_labels = base_labels_list
        self._dec_labels = dec_labels_list
        self._var_labels = var_labels_list
        self._conc_labels = conc_labels_list

    def get_roc_curves(
        self,
        conc_method=None,
        ph_limit=7.2,
        title="database_ctu-chb: FIGO 15 ph=" + str(7.20),
        ax=None,
        show_roc_curve=True,
        only_conclusion: bool = False,
        # -- rules param --
        rules_type: str = "FIGO",
        center: bool = False,
        preprocess: bool = False,
        prep_cut_time: int = 60,
        prep_max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
        window_time_baseline: int = 10,
        baseline_graph: bool = False,
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
        window_var: int = 1,
        variability_graph: bool = False,
        incr_var_min_time: int = 30,
        incr_var_bandwidth: int = 25,
        red_var_duration_baseline: int = 50,
        red_var_duration_deceleration: int = 3,
        red_var_bandwith: int = 5,
        conclusion_graph: bool = False,
        # -- rules param --
    ) -> None:
        """
        Plot ROC curves for CTG labels within a specified pH interval,
        applying diagnostic rules and categorizing labels by severity.

        This method checks if rule-based labels are already applied;
        if not, it applies the rules with given parameters. It then computes
        label percentages for baseline, variability, deceleration, and conclusion
        categories, separates them into normal, suspicious, and pathological groups,
        and finally plots the ROC curves based on these categorizations.

        Args:
            title (str, optional): Title for the ROC plot. Defaults to "database_ctu-chb: FIGO 15 ph=7.20".
            conc_method (Optional[str], optional): Method for conclusion label processing. Defaults to None.
            ph_limit (float, optional): pH threshold limit to consider. Defaults to 7.2.
            only_conclusion (bool, optional): Whether to paint only the conclusions. The default is False.
            rules_type (str, optional): Type of rules to apply for labeling. Defaults to "FIGO".
            center (bool, optional): Whether to center signal during rule application. Defaults to False.
            preprocess (bool, optional): Whether to preprocess CTGs before labeling. Defaults to False.
            prep_cut_time (int, optional): Cut time parameter for preprocessing. Defaults to 60.
            prep_max_size_gaps (int, optional): Maximum size of gaps allowed during preprocessing. Defaults to 15.
            rm_tail_nan (bool, optional): Whether to remove trailing NaN values during preprocessing. Defaults to False.
            window_time_baseline (int, optional): Window duration for baseline calculation. Defaults to 10.
            baseline_graph (bool, optional): Whether to plot baseline graph. Defaults to False.
            amplitude_dec_cond (int, optional): Amplitude threshold for deceleration condition. Defaults to 15.
            time_in_deceleration (int, optional): Duration threshold for deceleration detection. Defaults to 15.
            time_in_rep_dec (int, optional): Duration threshold for repetitive decelerations. Defaults to 30.
            correlation_threshold (float, optional): Correlation threshold for deceleration analysis. Defaults to -0.4.
            time_in_late_dec_cond (int, optional): Duration threshold for late deceleration condition. Defaults to 30.
            contraction_time_shift (int, optional): Time shift applied to contractions signal. Defaults to 20.
            contraction_value_threshold (int, optional): Minimum amplitude to define contraction. Defaults to 10.
            contraction_duration_threshold (int, optional): Minimum duration for contraction detection. Defaults to 20.
            contraction_baseline_duration (int, optional): Duration for contraction baseline calculation. Defaults to 120.
            time_in_prolonged_dec (int, optional): Duration threshold for prolonged deceleration. Defaults to 3.
            time_in_rep_dec_for_red_var (int, optional): Duration for repetitive deceleration affecting reduced variability. Defaults to 20.
            time_patholog_prolonged_dec (int, optional): Duration for pathological prolonged deceleration. Defaults to 5.
            deceleration_graph (bool, optional): Whether to plot deceleration graphs. Defaults to False.
            window_var (int, optional): Window size for variability analysis. Defaults to 1.
            variability_graph (bool, optional): Whether to plot variability graphs. Defaults to False.
            incr_var_min_time (int, optional): Minimum duration for increased variability detection. Defaults to 30.
            incr_var_bandwidth (int, optional): Bandwidth parameter for increased variability. Defaults to 25.
            red_var_duration_baseline (int, optional): Duration for reduced variability baseline. Defaults to 50.
            red_var_duration_deceleration (int, optional): Duration for reduced variability during deceleration. Defaults to 3.
            red_var_bandwith (int, optional): Bandwidth for reduced variability detection. Defaults to 5.
            conclusion_graph (bool, optional): Whether to plot conclusion graphs. Defaults to True.
            normal_vs_rest :

        Returns:
            None: This method generates plots but does not return a value.
        """

        # Compute rule-based labels if they are not already available.
        if not hasattr(self, "_base_labels"):
            print("Applying rule-based analysis: ", rules_type)
            self.get_labels_rules(
                rules_type,
                center=center,
                preprocess=preprocess,
                prep_cut_time=prep_cut_time,
                prep_max_size_gaps=prep_max_size_gaps,
                rm_tail_nan=rm_tail_nan,
                window_time_baseline=window_time_baseline,
                baseline_graph=baseline_graph,
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
                window_var=window_var,
                variability_graph=variability_graph,
                incr_var_min_time=incr_var_min_time,
                incr_var_bandwidth=incr_var_bandwidth,
                red_var_duration_baseline=red_var_duration_baseline,
                red_var_duration_deceleration=red_var_duration_deceleration,
                red_var_bandwith=red_var_bandwith,
                conclusion_graph=conclusion_graph,
            )

        N = len(self._base_labels)

        # Allocate per-recording proportions for each classification category.
        base_normal = np.zeros(N)
        base_suspicious = np.zeros(N)
        base_pathological = np.zeros(N)

        var_normal = np.zeros(N)
        var_suspicious = np.zeros(N)
        var_pathological = np.zeros(N)

        dec_normal = np.zeros(N)
        dec_suspicious = np.zeros(N)
        dec_pathological = np.zeros(N)

        conc_normal = np.zeros(N)
        conc_suspicious = np.zeros(N)
        conc_pathological = np.zeros(N)

        # Convert sample-wise labels into per-recording proportions.
        for i in range(N):
            # Report progress in place.
            sys.stdout.write("\r" + str(i + 1) + "/" + str(N) + " ")

            # Compute category proportions for each feature.
            base_normal[i], base_suspicious[i], base_pathological[i] = (
                self._get_labels_percentage(self._base_labels[i])
            )
            var_normal[i], var_suspicious[i], var_pathological[i] = (
                self._get_labels_percentage(self._var_labels[i])
            )
            dec_normal[i], dec_suspicious[i], dec_pathological[i] = (
                self._get_labels_percentage(self._dec_labels[i])
            )

            # Derive the conclusion score using the selected aggregation method.
            if conc_method != "percentage":
                # Average the three feature-level proportions.
                conc_normal[i] = (base_normal[i] + var_normal[i] + dec_normal[i]) / 3.0
                conc_suspicious[i] = (
                    base_suspicious[i] + var_suspicious[i] + dec_suspicious[i]
                ) / 3.0
                conc_pathological[i] = (
                    base_pathological[i] + var_pathological[i] + dec_pathological[i]
                ) / 3.0

            else:
                # Use the explicit conclusion labels when requested.
                conc_normal[i], conc_suspicious[i], conc_pathological[i] = (
                    self._get_labels_percentage(self._conc_labels[i])
                )

        base_perc = base_pathological + (0.5 * base_suspicious)
        var_perc = var_pathological + (0.5 * var_suspicious)
        dec_perc = dec_pathological + (0.5 * dec_suspicious)
        conc_perc = conc_pathological + (0.5 * conc_suspicious)

        if only_conclusion:
            data_perc = {
                "conclusion": conc_perc,
            }
        else:
            data_perc = {
                "baseline": base_perc,
                "variability": var_perc,
                "deceleration": dec_perc,
                "conclusion": conc_perc,
            }

        # Plot the ROC curves using the derived per-recording scores.
        self._plot_roc_graph(
            data_perc,
            ph_limit=ph_limit,
            title=title,
            only_conclusion=only_conclusion,
            ax=ax,
            show=show_roc_curve,
        )

    def _get_labels_percentage(self, labels_array) -> Tuple[float, float, float]:
        """
        Compute the proportions of normal, suspicious, and pathological labels.

        Samples that do not match any of the configured classification labels
        are excluded from the denominator.

        Parameters
        ----------
        labels_array : array-like
            Sample-wise classification labels.

        Returns
        -------
        tuple of float
            Proportions of normal, suspicious, and pathological labels.

        Raises
        ------
        ValueError
            If the input array is empty.
        """

        # Convert to NumPy for vectorized counting.
        labels_array = np.array(labels_array)

        # Reject empty inputs explicitly.
        if len(labels_array) == 0:
            raise ValueError("ERROR: labels_array is empty")

        # Count the three recognized FIGO-based categories.
        normal = np.count_nonzero(labels_array == self.config.labels.normal)
        suspicious = np.count_nonzero(labels_array == self.config.labels.suspicious)
        pathological = np.count_nonzero(labels_array == self.config.labels.pathological)

        # Ignore samples that do not belong to a recognized category.
        total = normal + suspicious + pathological

        # Return zero proportions when no valid category is present.
        if total == 0:
            return 0, 0, 0

        # Convert counts to proportions.
        perc_normal = float(normal / total)
        perc_suspicious = float(suspicious / total)
        perc_pathological = float(pathological / total)

        return perc_normal, perc_suspicious, perc_pathological

    def _plot_roc_graph(
        self,
        data_perc,
        ph_limit,
        title,
        only_conclusion=False,
        ax=None,
        show=True,
    ):
        """
        Plot ROC curves comparing fetal hypoxia classification based on
        umbilical artery pH against the calculated CTG rule percentages.

        Parameters
        ----------
        data_perc : dict
            Dictionary containing scores for baseline, variability,
            deceleration and conclusion.
        ph_limit : float
            pH threshold used to define fetal hypoxia.
        title : str or None
            Plot title. If None, no title is displayed.
        only_conclusion : bool
            If True, only the ROC curve corresponding to the final
            CTG conclusion is plotted.
        ax : matplotlib.axes.Axes or None
            Axis where the ROC curve will be plotted.
            If None, a new figure is created.
        show : bool
            If True, displays and saves the figure.
            Set False when combining this ROC with other curves.

        Returns
        -------
        dict
            Dictionary containing FPR, TPR, thresholds and AUC
            for each plotted ROC curve.
        """

        # Colors used for the ROC curves.
        colors = {
            "baseline": "#4C78A8",  # Muted blue
            "variability": "#E45756",  # Soft red
            "deceleration": "#72B7B2",  # Teal
            "conclusion": "#F2A541",  # Warm orange
        }

        # Figure formatting.
        TITLE_FONTSIZE = 14
        LABEL_FONTSIZE = 12
        TICK_FONTSIZE = 10
        LEGEND_FONTSIZE = 10

        # Reuse an existing axis when curves are combined externally.
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5.5))
        else:
            fig = ax.figure

        # Define the positive class from the umbilical artery pH threshold.
        phs_class = np.where(
            self.get_pH() < ph_limit,
            1.0,
            0.0,
        )

        roc_results = {}

        # Compute one ROC curve for each requested score.
        for key, value in data_perc.items():

            if only_conclusion and key != "conclusion":
                continue

            y_true = np.asarray(
                phs_class,
                dtype=float,
            )

            y_score = np.asarray(
                value,
                dtype=float,
            )

            # Remove missing values while preserving sample alignment.
            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_score)

            y_true_clean = y_true[valid_mask]
            y_score_clean = y_score[valid_mask]

            # ROC estimation requires valid samples from both classes.
            if len(y_true_clean) == 0:
                print(
                    f"[WARN - {key}] "
                    "No valid pH or CTG percentage data found "
                    "for ROC. Skipping..."
                )
                continue

            unique_classes = np.unique(y_true_clean)

            if len(unique_classes) < 2:
                print(
                    f"[WARN - {key}] "
                    "ROC requires both classes (0 and 1). "
                    f"Only found {unique_classes}. Skipping..."
                )
                continue

            # Compute the ROC curve and area under the curve.
            fpr, tpr, thresholds = roc_curve(
                y_true_clean,
                y_score_clean,
                pos_label=1,
            )

            roc_auc = auc(
                fpr,
                tpr,
            )

            roc_results[key] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": roc_auc,
            }

            curve_color = colors.get(
                key,
                "#757575",
            )

            # Plot the ROC curve.
            ax.plot(
                fpr,
                tpr,
                color=curve_color,
                linewidth=2.2,
                label=(f"{key.capitalize()} " f"(AUC = {roc_auc:.2f})"),
                zorder=3,
            )

        # Configure axes and labels.
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xticks(np.arange(0, 1.01, 0.2))

        ax.set_yticks(np.arange(0, 1.01, 0.2))

        ax.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE,
            length=4,
            width=0.8,
        )

        ax.set_xlabel(
            "False Positive Rate",
            fontsize=LABEL_FONTSIZE,
            labelpad=7,
        )

        ax.set_ylabel(
            "True Positive Rate",
            fontsize=LABEL_FONTSIZE,
            labelpad=7,
        )

        if title is not None:
            ax.set_title(
                title,
                fontsize=TITLE_FONTSIZE,
                pad=18,
            )

        # Remove non-essential plot borders.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.grid(False)

        # Add the reference classifier and export when shown standalone.
        if show:

            # Random-classifier reference.
            ax.plot(
                [0, 1],
                [0, 1],
                color="black",
                linewidth=1.1,
                linestyle="--",
                alpha=0.65,
                label="Random",
                zorder=1,
            )

            ax.legend(
                loc="lower right",
                fontsize=LEGEND_FONTSIZE,
                frameon=False,
            )

            fig.tight_layout()

            fig.savefig(
                "fig/roc_curve_model.pdf",
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.05,
            )

            plt.show()

        return roc_results

    # ---------------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------------

    def ph_histogram(self):
        plt.figure(figsize=(12, 8))

        intervalos_columnas = np.arange(6.9, 7.51, 0.02)

        plt.hist(
            self.phs,
            bins=intervalos_columnas,
            color="#2849C1",
            edgecolor="white",
            alpha=0.6,
        )

        plt.axvline(
            x=7.2,
            color="#E04A3F",
            linestyle="--",
            linewidth=2,
            label="pH = 7.2",
        )

        plt.xlim(6.9, 7.5)
        plt.xticks(np.arange(7, 7.41, 0.1))

        plt.xlabel("pH values", fontsize=25)
        plt.ylabel("Frequency", fontsize=25)

        plt.legend(loc="upper right", fontsize=25)
        plt.tick_params(axis="both", labelsize=20)

        plt.tight_layout()

        plt.savefig("fig/pH_histogram.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def clasificar_ph(self):

        df = pd.DataFrame({"pH": self.phs})

        # Define pH intervals for descriptive visualization.
        condiciones = [
            df["pH"] < 7.15,
            (df["pH"] >= 7.15) & (df["pH"] <= 7.20),
            df["pH"] > 7.20,
        ]

        # Assign a category to each pH interval.
        etiquetas = [
            "Pathological",
            "Suspicious",
            "Normal",
        ]

        # Create the categorical pH variable.
        df["Category"] = np.select(condiciones, etiquetas, default="Normal")

        # Use a consistent category order in the plot.
        orden_categorias = [
            "Normal",
            "Suspicious",
            "Pathological",
        ]

        plt.figure(figsize=(12, 8))

        # Plot the pH distribution by category.
        sns.boxplot(
            data=df,
            x="Category",
            y="pH",
            order=orden_categorias,
            color="#2849C1",
            width=0.5,
            fliersize=4,
            boxprops=dict(alpha=0.6),
        )

        # Display the pH thresholds used for categorization.
        plt.axhline(
            y=7.15,
            color="#E04A3F",
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
            label="pH = 7.15",
        )
        plt.axhline(
            y=7.20,
            color="#F58220",
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
            label="pH = 7.20",
        )

        # Configure publication-oriented plot formatting.
        plt.xlabel("", fontsize=14)
        plt.ylabel("pH Values", fontsize=25)

        plt.legend(loc="upper right", fontsize=25, frameon=True)

        plt.tight_layout()

        plt.tick_params(axis="both", labelsize=25)

        plt.savefig("fig/ph_categories_boxplot.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def get_fhr(self):
        fhr = []

        for ctg in self.ctgs:
            fhr.append(ctg.fhr)

        return np.array(fhr)

    def get_pH(self):
        ph = []
        for ctg in self.ctgs:
            ph.append(ctg.ph)

        return np.array(ph)
