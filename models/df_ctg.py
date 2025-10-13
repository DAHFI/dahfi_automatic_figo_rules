import sys
from .ctg import CTG
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Union, Optional, Dict, Tuple

# TODO: Repasar los comentarios de los docstring


class DF_CTG:

    NORMAL = 0
    SUSPICIOUS = 1
    PATHOLOGICAL = 2

    def __init__(
        self,
        freq,
        MAX_TIME_FHR=250,
        MIN_TIME_FHR=0,
        MAX_TIME_UC=150,
        MIN_TIME_UC=0,
    ):

        self.ctgs = []
        self.phs = []
        self.freq = freq

        self.MAX_TIME_FHR = MAX_TIME_FHR
        self.MIN_TIME_FHR = MIN_TIME_FHR
        self.MAX_TIME_UC = MAX_TIME_UC
        self.MIN_TIME_UC = MIN_TIME_UC

        # self._base_labels
        # self._dec_labels
        # self._var_labels
        # self._conc_labels

    def add_ctg(self, ctg: CTG) -> None:
        """
        Add a CTG (Cardiotocography) object to the DF_CTG container.

        This method checks that the frequency of the input CTG matches the internal
        frequency of the DF_CTG instance before appending it and its corresponding PH value.
        If the frequencies do not match, it raises a ValueError.

        Args:
            ctg (CTG): A CTG object containing FHR, UC, PH, and frequency information.

        Returns:
            None: This method does not return anything.
        """

        # Check that the frequency of the input CTG matches the instance frequency
        if self.freq == ctg.frequency:

            # Append the CTG object and its corresponding PH value
            self.ctgs.append(ctg)
            self.phs.append(ctg.ph)

        # Raise an error if frequencies do not match
        else:
            raise ValueError(
                "ERROR: The frequency of the CTG is not equal to the frequency of DF_CTG (id: ",
                ctg.id,
                ")",
            )

    def new_ctg(self, fhr: pd.Series, uc: pd.Series, ph: float, id: int) -> None:
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
            fhr=fhr,
            uc=uc,
            freq=self.freq,
            ph=ph,
            id=id,
            MAX_TIME_FHR=self.MAX_TIME_FHR,
            MIN_TIME_FHR=self.MIN_TIME_FHR,
            MAX_TIME_UC=self.MAX_TIME_UC,
            MIN_TIME_UC=self.MIN_TIME_UC,
        )

        # Append the new CTG and its pH value to the lists
        self.ctgs.append(ctg)
        self.phs.append(ctg.ph)

    ## ----------------------------------------------
    ## --------------- TEST FUNCTIONS ---------------

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
            print(f"{amount} CTGs tienen fhr de longitud {length}")

    def get_ctg_by_id(self, id: int) -> Optional[CTG]:
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

    ## ---------------------------------------------
    ## --------------- PREPROCESSING ---------------

    def preprocess_ctgs(
        self,
        rules_type: str,
        cut_time: int = 60,
        max_size_gaps: int = 15,
        rm_tail_nan: bool = False,
    ) -> None:
        """
        Preprocess stored CTG records based on specified rules.

        This method supports different preprocessing rule types.
        Currently, only the "FIGO" rule type is supported, which applies FIGO-specific preprocessing
        to each CTG in the collection.

        Args:
            rules_type (str): Type of preprocessing rules to apply (supported: "FIGO").
            cut_time (int, optional): Time threshold in seconds for cutting the signal. Defaults to 60.
            max_size_gaps (int, optional): Maximum allowed gap size in the signal. Defaults to 15.
            rm_tail_nan (bool, optional): Whether to remove trailing NaN values. Defaults to True.

        Returns:
            None: This method modifies CTGs in place and does not return anything.
        """

        # Define supported rule types
        correct_rules_type = ["FIGO"]

        # Validate input rule type
        if rules_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # Apply FIGO-specific preprocessing to each CTG
        if rules_type == "FIGO":
            for ctg in self.ctgs:
                ctg.preprocess_rules_figo(
                    cut_time=cut_time,
                    max_size_gaps=max_size_gaps,
                    rm_tail_nan=rm_tail_nan,
                )

    ## ---------------------------------------------
    ## --------------- RULES METODS ----------------

    # TODO: Añadir la opción de guardar en algún lado los resultados
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

        # Number of stored CTGs
        N = len(self.ctgs)

        base_labels_list = []
        dec_labels_list = []
        var_labels_list = []
        conc_labels_list = []

        # Iterate over each CTG and apply the rules
        for i in range(N):
            # Show progress message
            sys.stdout.write("\r" + str(i + 1) + "/" + str(N) + " ")

            # Apply the rules
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

            # Store scores for each category
            base_labels_list.append(labels_rules_ctg["baseline"])
            dec_labels_list.append(labels_rules_ctg["deceleration"])
            var_labels_list.append(labels_rules_ctg["variability"])
            conc_labels_list.append(labels_rules_ctg["conclusion"])

        # Save results in the instance
        self._base_labels = base_labels_list
        self._dec_labels = dec_labels_list
        self._var_labels = var_labels_list
        self._conc_labels = conc_labels_list

    def get_roc_curves(
        self,
        conc_method=None,
        ph_limit=7.2,
        title="database_ctu-chb: FIGO 15 ph=" + str(7.20),
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
        # normal_vs_rest=True, TODO: Quitarlo del esquema
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

        # Check if rules are already applied; if not, apply them
        if not hasattr(self, "_base_labels"):
            print("The rules will be applied following the guidelines: ", rules_type)
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

        # Number of results; should match number of CTGs #TODO: Comprobar que eso es verdad
        N = len(self._base_labels)

        # Initialize arrays for label categories
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

        # Iterate through each data point and calculate label percentages
        for i in range(N):
            # Print progress on same line: "current_index/total"
            sys.stdout.write("\r" + str(i + 1) + "/" + str(N) + " ")

            # Get percentages of normal, suspicious, pathological for baseline
            base_normal[i], base_suspicious[i], base_pathological[i] = (
                self._get_labels_percentage(self._base_labels[i])
            )
            var_normal[i], var_suspicious[i], var_pathological[i] = (
                self._get_labels_percentage(self._var_labels[i])
            )
            dec_normal[i], dec_suspicious[i], dec_pathological[i] = (
                self._get_labels_percentage(self._dec_labels[i])
            )

            # Calculate conclusion percentages differently depending on method
            if conc_method != "percentage":
                # Average percentages from baseline, variability, deceleration for conclusion
                conc_normal[i] = (base_normal[i] + var_normal[i] + dec_normal[i]) / 3.0
                conc_suspicious[i] = (
                    base_suspicious[i] + var_suspicious[i] + dec_suspicious[i]
                ) / 3.0
                conc_pathological[i] = (
                    base_pathological[i] + var_pathological[i] + dec_pathological[i]
                ) / 3.0

            else:
                # Directly get percentages from conclusion labels if method is "percentage"
                conc_normal[i], conc_suspicious[i], conc_pathological[i] = (
                    self._get_labels_percentage(self._conc_labels[i])
                )

        base_perc = base_pathological + (0.5 * base_suspicious)
        var_perc = var_pathological + (0.5 * var_suspicious)
        dec_perc = dec_pathological + (0.5 * dec_suspicious)
        conc_perc = conc_pathological + (0.5 * conc_suspicious)

        # # Prepare lists of arrays by category for plotting
        # if normal_vs_rest:
        #     base_perc = [base_normal, base_suspicious + base_pathological]
        #     var_perc = [var_normal, var_suspicious + var_pathological]
        #     dec_perc = [dec_normal, dec_suspicious + dec_pathological]
        #     conc_perc = [conc_normal, conc_suspicious + conc_pathological]

        #     title = " A : Conservative Case "
        # else:
        #     base_perc = [base_normal + base_suspicious, base_pathological]
        #     var_perc = [var_normal + var_suspicious, var_pathological]
        #     dec_perc = [dec_normal + dec_suspicious, dec_pathological]
        #     conc_perc = [conc_normal + conc_suspicious, conc_pathological]

        #     title = " B : Liberal Case "

        # TODO: CREAR LA OTRA, SOLO PATH

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

        # Call plotting function with all prepared data and parameters
        self._plot_roc_graph(
            data_perc, ph_limit=ph_limit, title=title, only_conclusion=only_conclusion
        )

    def _get_labels_percentage(self, labels_array) -> Tuple[float, float, float]:
        """
        Calculate the percentage of normal, suspicious, and pathological labels in an array.

        Args:
            labels_array (pd.Series[int]): Array of labels to be analyzed. Each label should be one of
                                            self.NORMAL, self.SUSPICIOUS, or self.PATHOLOGICAL.

        Returns:
            Tuple[float, float, float]: Percentages of normal, suspicious, and pathological labels respectively.

        Raises:
            ValueError: If the input labels_array is empty.
        """

        # Convert input to numpy array for efficient computation
        labels_array = np.array(labels_array)

        # Check for empty array and raise an error if empty
        if len(labels_array) == 0:
            raise ValueError("ERROR: labels_array is empty")

        # Count the occurrences of each label type
        normal = np.count_nonzero(labels_array == self.NORMAL)
        suspicious = np.count_nonzero(labels_array == self.SUSPICIOUS)
        pathological = np.count_nonzero(labels_array == self.PATHOLOGICAL)

        # Sum counts to get total number of labels counted
        total = normal + suspicious + pathological

        # Edge case: if no recognized labels found, return zeros
        if total == 0:
            return 0, 0, 0

        # Calculate the proportion of each label type
        perc_normal = float(normal / total)
        perc_suspicious = float(suspicious / total)
        perc_pathological = float(pathological / total)

        return perc_normal, perc_suspicious, perc_pathological

    def _plot_roc_graph(
        self,
        data_perc: dict,
        ph_limit: float = 7.20,
        title: str = None,
        only_conclusion: bool = False,
    ) -> None:
        """
        Plot ROC curves comparing label percentages against hypoxia presence (ph_limit threshold).

        Args:
            data_perc (dict):
            ph_limit (float): Threshold of pH to define hypoxia presence.
            title (Optional[str]): Title for the plots.

        Returns:
            None: This function displays ROC curve plots and does not return a value.
        """

        # Create boolean array: True where CTG pH is less or equal to ph_limit (indicating hypoxia)
        phs_class = [
            x <= ph_limit for x in self.phs
        ]  # BORRAR: Si el ph es menor que 7.2 (Patológico) lo guardamos como 1

        plt.figure(figsize=(7, 6))  # TODO: Quitar

        plt.plot([0, 1], [0, 1], "k--", label="Random")

        if only_conclusion:
            fpr, tpr, _ = roc_curve(phs_class, data_perc["conclusion"])

            # Calculate area under the curve (AUC) metric
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve with label showing category and AUC score
            plt.plot(
                fpr,
                tpr,
                color="palevioletred",
                # label="ROC curve %s (AUC = %0.2f)" % ("conclusion", roc_auc),
                label="ROC Automatic Model (AUC = %0.2f)" % roc_auc,
            )
        else:

            color = {
                "baseline": "palevioletred",
                "variability": "yellowgreen",
                "deceleration": "sandybrown",
                "conclusion": "steelblue",
            }

            for key, value in data_perc.items():
                # Unpack suspicious and pathological percentages, ignoring normal here
                # _, perc_sick = value

                # Compute ROC curve values comparing hypoxia label to suspicious+pathological combined
                fpr, tpr, _ = roc_curve(phs_class, value)

                # Calculate area under the curve (AUC) metric
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve with label showing category and AUC score
                plt.plot(
                    fpr,
                    tpr,
                    color=color[key],
                    label="ROC curve %s (AUC = %0.2f)" % (key, roc_auc),
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(title)
        plt.legend(loc="lower right")

        save_name = "curve_model.png"

        plt.savefig(save_name, dpi=300, bbox_inches="tight")
