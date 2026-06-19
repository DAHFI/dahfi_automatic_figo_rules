import sys
from .ctg import CTG
from collections import Counter
import commentjson
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Union, Optional, Dict, Tuple
from config_methods import AppConfig
from utils.features import extract_all_signal_features, extract_clinical_features


class DF_CTG:

    def __init__(
        self,
        config_file: str,
    ):

        self.ctgs = []
        # self.phs = []  # TODO: BORRAR
        # self.clinical_data = []  # TODO: BORRAR

        try:
            with open("config.jsonc", "r", encoding="utf-8") as file:
                config = AppConfig(commentjson.load(file))
                self.config = config

        except Exception as e:
            print("ERROR! It was not possible to read the config file")
            print(f"Error details: {e}")
            self.config = None

        ## --- VAR FOR SAVING RESULTS ---
        # self._base_labels
        # self._dec_labels
        # self._var_labels
        # self._conc_labels

    def add_ctg(self, ctg: CTG) -> None:
        """
        Add a CTG (Cardiotocography) object to the DF_CTG group.
        """
        ctg.group = self  # Ahora forma parte de una familia 🥲
        self.ctgs.append(ctg)

    def read_ctgs_from_files(self) -> None:
        # Variables to provide information on the progress of data reading
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

            fhr = fhr_df.loc[index][:max_num_data]
            uc = uc_df.loc[index][:max_num_data]
            clinical_data = clinical_df.loc[index]

            ph = float(clinical_data["ph"])

            self.new_ctg(fhr, uc, clinical_data, ph, index)
            num_ctg_created += 1
            print(f"{index} created! Total success: {num_ctg_created}")

        print(num_ctg_created, "ctg created!")

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

    ## ----------------------------------------------
    ## ------------ GET DATA TF STYLE ---------------

    def get_ph_array(self) -> np.ndarray:
        """
        Get the list of pH values for all stored CTGs.

        Returns:
            list: A list containing the pH values of all CTGs in the instance.
        """
        return np.array([ctg.ph for ctg in self.ctgs])

    def get_corr_curves_and_ph(self):
        print("Para obtener estas curvas antes hay que ejecutar get_corr()")

        corr_curves = []
        ph = []

        for ctg in self.ctgs:

            if np.isnan(ctg.corr_fhr_uc).all():
                continue

            print(ctg.id)
            corr_interp = np.interp(
                np.arange(len(ctg.corr_fhr_uc)),
                np.where(~np.isnan(ctg.corr_fhr_uc))[0],
                ctg.corr_fhr_uc[~np.isnan(ctg.corr_fhr_uc)],
            )

            corr_curves.append(corr_interp)
            ph.append(ctg.ph)

        return np.array(corr_curves), np.array([1 if x < 7.2 else 0 for x in ph])

    def get_training_features(self):
        """
        Get the training features for all stored CTGs.

        Returns:
            list: A list containing the training features of all CTGs in the instance.
        """
        return [ctg.get_features() for ctg in self.ctgs]

    def get_all_features(self, get_names=True):
        if hasattr(self.ctgs[0], "_preprocess_type"):
            print(
                f"Extrayendo las características usando el preprocesado: {self.ctgs[0]._preprocess_type}"
            )
        else:
            print(
                "WARNING!! : Se están extrayendo las características usando señales sin preprocesar..."
            )

        if not hasattr(self.ctgs[0], "corr_fhr_uc") or not hasattr(
            self.ctgs[0], "corr_fhr_prima_uc"
        ):
            raise RuntimeError("Debe de ejecutarse antes .get_corr()")

        # FHR
        print("1 -> Extrayendo características de FHR")
        signals_fhr = [ctg.fhr for ctg in self.ctgs]
        names, feat_fhr = extract_all_signal_features(signals_fhr, get_names=get_names)

        name_fhr = [f"{name}_FHR" for name in names]

        # FHR'
        print("2 -> Extrayendo características de FHR'")
        signals_fhr_prima = [np.diff(ctg.fhr) for ctg in self.ctgs]
        names, feat_fhr_prima = extract_all_signal_features(
            signals_fhr_prima, get_names=get_names
        )

        name_fhr_prima = [f"{name}_FHR_prima" for name in names]

        # UC
        print("3 -> Extrayendo características de UC")
        signals_uc = [ctg.uc for ctg in self.ctgs]
        names, feat_uc = extract_all_signal_features(signals_uc, get_names=get_names)

        name_uc = [f"{name}_UC" for name in names]

        # Corr
        print("4 -> Extrayendo características de Corr (FHR y UC)")
        signals_corr = [ctg.corr_fhr_uc for ctg in self.ctgs]
        names, feat_corr = extract_all_signal_features(
            signals_corr, get_names=get_names
        )

        name_corr = [f"{name}_CORR" for name in names]

        # Corr Prima
        print("5 -> Extrayendo características de Corr (FHR' y UC)")
        signals_corr_prima = [ctg.corr_fhr_prima_uc for ctg in self.ctgs]
        names, feat_corr_prima = extract_all_signal_features(
            signals_corr_prima, get_names=get_names
        )

        name_corr_prima = [f"{name}_CORR_PRIMA" for name in names]

        # Clinicas
        print("6 -> Extrayendo características Clínicas")
        ids = [ctg.id for ctg in self.ctgs]
        name_clinicas, feat_clinicas = extract_clinical_features(
            ids, get_names=get_names
        )

        # Resultados finales
        names = (
            name_fhr
            + name_fhr_prima
            + name_uc
            + name_corr
            + name_corr_prima
            + name_clinicas
        )
        feat = np.hstack(
            (
                feat_fhr,
                feat_fhr_prima,
                feat_uc,
                feat_corr,
                feat_corr_prima,
                feat_clinicas,
            )
        )

        return names, feat

    def prepare_model_data(self):
        """
        Obtener los vectores (tensorflow) para entrenar modelos.
        La forma de los vectores es :
            X -> TensorShape([552, 2, 7200])
            y -> TensorShape([552])

        """

        # pH values
        phs = self.get_ph_list()
        y = tf.where(tf.convert_to_tensor(phs) > 7.2, 0, 1)

        # get the signals
        X_list = []

        for ctg in self.ctgs:
            df_combined = pd.concat([ctg.fhr, ctg.uc], axis=1)
            X_list.append(df_combined.values)

        X_padded = tf.keras.utils.pad_sequences(
            X_list,
            dtype="float32",
            padding="pre",
            value=0.0,
        )

        X = tf.transpose(
            tf.convert_to_tensor(X_padded, dtype=tf.float32), perm=[0, 2, 1]
        )

        return X, y

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

    ## ---------------------------------------------
    ## --------------- PREPROCESSING ---------------

    def preprocess_ctgs(
        self,
    ) -> None:
        """
        Preprocess stored CTG records based on specified rules.

        This method supports different preprocessing rule types.
        Currently, only the "FIGO" rule type is supported, which applies FIGO-specific preprocessing
        to each CTG in the collection.

        Args:
            rules_type (str): Type of preprocessing rules to apply (supported: "FIGO").
            cut_time (int, optional): Time threshold in MINUTES for cutting the signal. Defaults to 60.
            max_size_gaps (int, optional): Maximum allowed gap size in the signal. Defaults to 15.
            rm_tail_nan (bool, optional): Whether to remove trailing NaN values. Defaults to True.

        Returns:
            None: This method modifies CTGs in place and does not return anything.
        """

        prep_type = self.config.preprocessing.prep_type

        new_ctgs_list = []

        # Define supported rule types
        correct_rules_type = ["FIGO", "FB_FOURIER"]

        # Validate input rule type
        if prep_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # Apply FIGO-specific preprocessing to each CTG
        for ctg in self.ctgs:
            error_code = ctg.preprocess_signal()

            if error_code == 0:
                new_ctgs_list.append(ctg)

        self.ctgs = new_ctgs_list

    ## ---------------------------------------------
    ## -------------- CORR FUNCTION ----------------

    def get_corr(self, max_desplazamiento=5, graph=False):
        # TODO: one_ctg_analisis borrar
        # TODO: incluir graph

        max_desplazamiento = self.config.correlation.max_sec_displacement
        max_desplazamiento = 60  # borrar

        contador_error = 0

        ctg_with_corr = []

        # Creamos una figura con 2 filas y 1 columna. sharex=True hace que compartan el eje X.
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Configuración del gráfico de ARRIBA (UC vs FHR)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlim(-max_desplazamiento, max_desplazamiento)
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_title("Correlación de la FHR y UC")
        ax1.set_ylabel("Correlación (FHR)")
        ax1.grid(True, alpha=0.3)

        # Configuración del gráfico de ABAJO (UC vs FHR')
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_title("Correlación de la Derivada FHR' y UC")
        ax2.set_xlabel("Desplazamiento en minutos")
        ax2.set_ylabel("Correlación (FHR')")
        ax2.grid(True, alpha=0.3)

        # Pasamos ambos ejes a la función interna para que dibuje en los dos
        for ctg in self.ctgs:
            res = ctg.get_corr_fun(
                ax1,
                ax2,
                contador_error,
                one_ctg_analisis=False,
            )

            if res == -1:
                contador_error = contador_error + 1
            else:
                ctg_with_corr.append(ctg)

        # Ajusta el diseño para que los títulos y etiquetas no se solapen
        fig.tight_layout()
        fig.show()

        print("Señales saltadas: ", contador_error, " / ", len(self.ctgs))

        self.ctgs = ctg_with_corr

    ## ---------------------------------------------
    ## --------------- SMOOTHE METODS --------------

    def smoothe(
        self,
        type_smoothe: str = "Fourier",
    ):
        """
        Suavizamos las curvas FHR de las ctg. Para ellos se
        utiliza la interpolación de Fourier.
        """

        if type_smoothe == "Fourier":
            for ctg in self.ctgs:
                ctg.smoothe(type_smoothe=type_smoothe)

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

        plt.figure(figsize=(12, 12))  # TODO: Quitar

        plt.plot([0, 1], [0, 1], "k--", label="Random")

        if only_conclusion:
            fpr, tpr, _ = roc_curve(phs_class, data_perc["conclusion"])

            # Calculate area under the curve (AUC) metric
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve with label showing category and AUC score
            plt.plot(
                fpr,
                tpr,
                color="#2849C1",
                # label="ROC curve %s (AUC = %0.2f)" % ("conclusion", roc_auc),
                label="ROC Automatic Model (AUC = %0.2f)" % roc_auc,
            )

            plt.fill_between(fpr, 0, tpr, color="#2849C1", alpha=0.3)

        else:

            color = {
                "baseline": "#2849C1",
                "variability": "#E04A3F",
                "deceleration": "#388E3C",
                "conclusion": "#F58220",
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
                    linewidth=1.5,
                    label="ROC curve %s (AUC = %0.2f)" % (key, roc_auc),
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xticks(fontsize=25)
        plt.yticks(np.arange(0.2, 1.2, 0.2), fontsize=25)

        plt.xlabel("False Positive Rate", fontsize=25)
        plt.ylabel("True Positive Rate", fontsize=25)

        plt.title(title, fontsize=25)
        # plt.legend(loc="lower right", fontsize=15)

        # Leyenda debajo de la gráfica
        # plt.legend(
        #     loc="upper center",  # posición de referencia
        #     bbox_to_anchor=(0.5, -0.15),  # 0.5 = centro horizontal, -0.15 = fuera abajo
        #     ncol=1,  # número de columnas en la leyenda
        #     fontsize=14,
        # )

        plt.legend(
            loc="lower right",  # Posición fija dentro del recuadro
            ncol=1,  # Una sola columna
            fontsize=20,
        )

        save_name = "curve_model.pdf"

        # Save plot
        plt.savefig(save_name, format="pdf", bbox_inches="tight")

    ## ---------------------------------------------
    ## ------------- GRAPH FUNCTION ----------------

    def ph_histogram(self):
        # 2. Configurar el tamaño de la imagen alargada (como en tus gráficas anteriores)
        # Nota: Volví a poner (12, 4) por si querías mantener el formato alargado anterior,
        # si lo quieres totalmente cuadrado puedes regresar a (12, 12).
        plt.figure(figsize=(12, 8))

        intervalos_columnas = np.arange(6.9, 7.51, 0.02)

        # 3. Dibujar el histograma de la LÍSTA pasando los intervalos manuales
        plt.hist(
            self.phs,
            bins=intervalos_columnas,
            color="#2849C1",
            edgecolor="white",
            alpha=0.6,
        )

        # Línea vertical clavada en el borde de la columna
        plt.axvline(
            x=7.2,
            color="#E04A3F",
            linestyle="--",
            linewidth=2,
            label="pH = 7.2",
        )
        # ============================================================

        # 4. Forzar el intervalo del eje X
        plt.xlim(6.9, 7.5)
        plt.xticks(np.arange(7, 7.41, 0.1))

        # 5. Nombres de los ejes y títulos grandes
        plt.xlabel("pH values", fontsize=25)
        plt.ylabel("Frequency", fontsize=25)

        # Añadir la leyenda para que se vea qué significa la línea roja
        plt.legend(loc="upper right", fontsize=25)
        plt.tick_params(axis="both", labelsize=20)

        # 7. Detalles visuales y cuadrícula de fondo
        plt.tight_layout()

        # 8. Guardar opcionalmente para tu LaTeX o mostrar
        plt.savefig("pH_histogram.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    # 2. Función para clasificar cada pH según tus reglas clínicas
    def clasificar_ph(self):

        # 1. Crear el DataFrame con tu lista de phs
        df = pd.DataFrame({"pH": self.phs})

        # 2. Definir las condiciones utilizando las columnas del DataFrame
        condiciones = [
            df["pH"] < 7.15,
            (df["pH"] >= 7.15) & (df["pH"] <= 7.20),
            df["pH"] > 7.20,
        ]

        # 3. Definir las etiquetas que corresponden a cada condición en el mismo orden
        etiquetas = [
            "Pathological",
            "Suspicious",
            "Normal",
        ]

        # 4. Crear la columna 'Category' sin usar funciones intermedias
        df["Category"] = np.select(condiciones, etiquetas, default="Normal")

        # Ordenar las categorías para que aparezcan en un orden clínico lógico en el gráfico
        orden_categorias = [
            "Normal",
            "Suspicious",
            "Pathological",
        ]

        # Definir tu paleta de colores acoplada exacta
        # colores_paleta = {
        #     "Normal": "#2849C1",  # Tu Azul
        #     "Suspicious": "#F58220",  # Tu Naranja
        #     "Pathological": "#E04A3F",  # Tu Rojo
        # }

        # 4. Configurar el tamaño de la imagen (Formato estándar/cuadrado para boxplots)
        plt.figure(figsize=(12, 8))

        # 5. Dibujar el gráfico de cajas con Seaborn
        sns.boxplot(
            data=df,
            x="Category",
            y="pH",
            order=orden_categorias,
            color="#2849C1",
            width=0.5,
            fliersize=4,  # Tamaño de los puntos atípicos (outliers)
            boxprops=dict(alpha=0.6),
        )

        # 6. Añadir las líneas de umbral horizontales en el fondo para validar visualmente
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

        # 7. Personalizar nombres de los ejes y tamaños grandes para publicaciones
        plt.xlabel(
            "", fontsize=14
        )  # Dejamos el eje X vacío porque las etiquetas de las cajas ya lo explican
        plt.ylabel("pH Values", fontsize=25)
        # plt.title(
        #     "Distribution of pH Recordings by Clinical Category",
        #     fontsize=15,
        #     pad=15,
        #     fontweight="bold",
        # )

        plt.legend(loc="upper right", fontsize=25, frameon=True)

        # Estilo de cuadrícula sutil
        # plt.grid(True, linestyle="--", alpha=0.4, axis="y")
        plt.tight_layout()

        plt.tick_params(axis="both", labelsize=25)

        # 8. Guardar listo para tu documento de LaTeX
        plt.savefig("ph_categories_boxplot.pdf", format="pdf", bbox_inches="tight")
        plt.show()
