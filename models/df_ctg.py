import sys
import copy
import commentjson
import numpy as np
import pandas as pd
from .ctg import CTG
import seaborn as sns
import tensorflow as tf
from mrmr import mrmr_classif
from collections import Counter
from config_methods import AppConfig
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Union, Optional, Dict, Tuple
from sklearn.feature_selection import mutual_info_classif

from utils.features import extract_all_signal_features, extract_clinical_features


class DF_CTG:

    def __init__(
        self,
        config_file: str,
        additional_config={},
    ):

        self.ctgs = []
        # self.phs = []  # TODO: BORRAR
        # self.clinical_data = []  # TODO: BORRAR

        self.additional_config = additional_config

        try:
            with open(config_file, "r", encoding="utf-8") as file:
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

            fhr = fhr_df.loc[index].iloc[-max_num_data:]
            uc = uc_df.loc[index].iloc[-max_num_data:]
            clinical_data = clinical_df.loc[index]

            ph = float(clinical_data["ph"])

            self.new_ctg(fhr, uc, clinical_data, ph, index)
            num_ctg_created += 1
            # print(f"{index} created! Total success: {num_ctg_created}")

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

    ## ----------------------------------------------
    ## ------------ GET DATA TF STYLE ---------------

    def get_labels_array(self) -> np.ndarray:

        ph = np.array([ctg.ph for ctg in self.ctgs])

        return (ph < 7.2).astype(int)

    def get_corr_curves_and_ph(self):
        if not hasattr(self.ctgs[0], "corr_fhr_uc"):
            print("Para obtener estas curvas antes hay que ejecutar get_corr()")
            self.get_corr()

        corr_curves = []
        ph = []

        for ctg in self.ctgs:

            if np.isnan(ctg.corr_fhr_uc).all():
                continue

            # print(ctg.id)
            corr_interp = np.interp(
                np.arange(len(ctg.corr_fhr_uc)),
                np.where(~np.isnan(ctg.corr_fhr_uc))[0],
                ctg.corr_fhr_uc[~np.isnan(ctg.corr_fhr_uc)],
            )

            corr_curves.append(corr_interp)
            ph.append(ctg.ph)

        return np.array(corr_curves), np.array([1 if x < 7.2 else 0 for x in ph])

    def get_fhr_features(self):
        signals_fhr = [ctg.fhr for ctg in self.ctgs]
        names, feat_fhr = extract_all_signal_features(
            signals_fhr, get_names=True, fhr=True
        )

        return names, feat_fhr

    def get_fhr_prima_features(self):
        signals_fhr_prima = [np.diff(ctg.fhr) for ctg in self.ctgs]
        names, feat_fhr_prima = extract_all_signal_features(
            signals_fhr_prima, get_names=True, fhr=False
        )

        return names, feat_fhr_prima

    def get_uc_features(self):
        signals_uc = [ctg.uc for ctg in self.ctgs]
        names, feat_uc = extract_all_signal_features(
            signals_uc, get_names=True, fhr=False
        )

        return names, feat_uc

    def get_corr_features(self):
        signals_corr = [ctg.corr_fhr_uc for ctg in self.ctgs]
        names, feat_corr = extract_all_signal_features(
            signals_corr, get_names=True, fhr=False
        )

        return names, feat_corr

    def get_clinic_features(self):
        ids = [ctg.id for ctg in self.ctgs]
        name_clinicas, feat_clinicas = extract_clinical_features(ids, get_names=True)
        return name_clinicas, feat_clinicas

    def get_all_features(self, get_names=True):
        if not hasattr(self.ctgs[0], "_preprocess_type"):
            print(
                "WARNING!! : Se están extrayendo las características usando señales sin preprocesar..."
            )

        if not hasattr(self.ctgs[0], "corr_fhr_uc") or not hasattr(
            self.ctgs[0], "corr_fhr_prima_uc"
        ):
            print("Extrayendo las curvas de correlaciones!!")
            self.get_corr()
            # raise RuntimeError("Debe de ejecutarse antes .get_corr()")

        # FHR
        signals_fhr = [ctg.fhr for ctg in self.ctgs]
        names, feat_fhr = extract_all_signal_features(
            signals_fhr, get_names=get_names, fhr=True
        )

        name_fhr = [f"{name}_FHR" for name in names]

        # FHR'
        signals_fhr_prima = [np.diff(ctg.fhr) for ctg in self.ctgs]
        names, feat_fhr_prima = extract_all_signal_features(
            signals_fhr_prima, get_names=get_names, fhr=False
        )

        name_fhr_prima = [f"{name}_FHR_PRIMA" for name in names]

        # UC
        signals_uc = [ctg.uc for ctg in self.ctgs]
        names, feat_uc = extract_all_signal_features(
            signals_uc, get_names=get_names, fhr=False
        )

        name_uc = [f"{name}_UC" for name in names]

        # Corr
        signals_corr = [ctg.corr_fhr_uc for ctg in self.ctgs]
        names, feat_corr = extract_all_signal_features(
            signals_corr, get_names=get_names, fhr=False
        )

        name_corr = [f"{name}_CORR" for name in names]

        # Corr Prima
        signals_corr_prima = [ctg.corr_fhr_prima_uc for ctg in self.ctgs]
        names, feat_corr_prima = extract_all_signal_features(
            signals_corr_prima, get_names=get_names, fhr=False
        )

        name_corr_prima = [f"{name}_CORR_PRIMA" for name in names]

        # Clinicas
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

        self.names_features = names
        self.features = feat

        print(f"Features calculated: {len(names)}")

        return np.array(names), feat

    def get_mrmr_features(self, num_features=30, get_names=False):
        # if not hasattr(self, "features"):
        # self.get_all_features()

        df_featues = pd.DataFrame(self.features, columns=self.names_features)
        labels = self.get_labels_array()

        features_seleccionadas = mrmr_classif(
            X=df_featues, y=labels, K=num_features, show_progress=False
        )

        # print(
        #     f"=== CARACTERÍSTICAS SELECCIONADAS POR mRMR (num_feat = {num_features}) ==="
        # )
        # for i, feat in enumerate(features_seleccionadas, start=1):
        #     print(f"{i}. {feat}")

        # print("\n")

        if get_names:
            return features_seleccionadas

        return df_featues[features_seleccionadas].values

    def get_signal_and_ph(self):
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

    def get_fhr(self):
        fhr = []

        for ctg in self.ctgs:
            fhr.append(ctg.fhr)

        return np.array(fhr)

    def get_uc(self):
        uc = []
        for ctg in self.ctgs:
            uc.append(ctg.uc)

        return np.array(uc)

    def get_fhr_prima(self):
        fhr_prima = []

        for ctg in self.ctgs:
            fhr_prima.append(ctg.get_fhr_prima())

        return np.array(fhr_prima)

    def get_corr_curves(self):
        corr_fhr_uc = []
        for ctg in self.ctgs:
            corr_fhr_uc.append(ctg.corr_fhr_uc)

        return np.array(corr_fhr_uc)

    def get_corr_prima_curves(self):
        corr_fhr_uc = []
        for ctg in self.ctgs:
            corr_fhr_uc.append(ctg.corr_fhr_prima_uc)

        return np.array(corr_fhr_uc)

    def get_pH(self):
        ph = []
        for ctg in self.ctgs:
            ph.append(ctg.ph)

        return np.array(ph)

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

    def preprocess_ctgs(self, prep_type=None) -> None:
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

        if "prep_type" in self.additional_config.keys():
            prep_type = self.additional_config["prep_type"]

        else:
            prep_type = self.config.preprocessing.prep_type

        # print(f"Preprocesando según: {prep_type}")

        new_ctgs_list = []

        num_error = 0
        id_error = []
        original_size = len(self.get_fhr())

        # Define supported rule types
        correct_rules_type = ["FIGO", "FB_FOURIER", "SPLINES"]

        # Validate input rule type
        if prep_type not in correct_rules_type:
            raise ValueError("ERROR: It is not a valid preprocessing")

        # Apply FIGO-specific preprocessing to each CTG
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

    ## ---------------------------------------------
    ## -------------- CORR FUNCTION ----------------

    def get_corr(self):
        # TODO: one_ctg_analisis borrar
        # TODO: incluir graph

        if hasattr(self.ctgs[0], "corr_fhr_uc"):
            print("The correlation curves have already been previously calculated...")
            return None

        max_desplazamiento = self.config.correlation.max_sec_displacement

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
        ax2.set_xlabel("Desplazamiento en segundos")
        ax2.set_ylabel("Correlación (FHR')")
        ax2.grid(True, alpha=0.3)

        ids_ctg_error = []
        original_size = len(self.get_fhr())

        # Pasamos ambos ejes a la función interna para que dibuje en los dos
        for ctg in self.ctgs:
            res = ctg.get_corr_fun(
                ax1,
                ax2,
                contador_error=contador_error,
                one_ctg_analisis=False,
                plot_graph=True,
            )

            if res == -1:
                contador_error = contador_error + 1

                ids_ctg_error.append(ctg.id)
            else:
                ctg_with_corr.append(ctg)

        # Ajusta el diseño para que los títulos y etiquetas no se solapen
        fig.tight_layout()
        # fig.show()
        plt.close(fig)

        # print("Señales saltadas: ", contador_error, " / ", len(self.ctgs))

        self.ctgs = ctg_with_corr

        print(
            f"Calculated correlation curves: {contador_error}/{original_size} skipped as invalid (id= {ids_ctg_error})"
        )

    def plot_histogram_corr(self):

        if not hasattr(self.ctgs[0], "corr_fhr_uc"):
            print("Ejecuta antes get_corr()")
            return None

        lags_sano = []
        lags_patolog = []

        for ctg in self.ctgs:
            if ctg.ph < self.config.ph_limit:
                lags_patolog.append(ctg.lags_corr / 4)
            else:
                lags_sano.append(ctg.lags_corr / 4)

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # 1. Pintamos el lote Sano (Verde) con sus argumentos directos
        ax.hist(
            lags_sano,
            bins=50,
            range=(-60, 60),
            density=True,
            alpha=0.5,
            color="green",
            label="Sano",
            edgecolor="darkgreen",
        )

        # 2. Pintamos el lote Patológico (Rojo) con sus argumentos directos
        ax.hist(
            lags_patolog,
            bins=50,
            range=(-60, 60),
            density=True,
            alpha=0.5,
            color="red",
            label="Patológico",
            edgecolor="darkred",
        )

        # 3. Decoración del gráfico
        ax.set_title(
            "Histograma Correlación FHR y UC",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Desplazamiento en segundos")
        ax.set_ylabel("Frecuencia")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        plt.tight_layout()

        plt.savefig("corr_hist_fhr.pdf", format="pdf", bbox_inches="tight")

        plt.show()

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

    def _plot_roc_graph(self, data_perc, ph_limit, title, only_conclusion=False):
        """Plot the ROC curve comparing fetal hypoxia classification based on umbilical
        artery pH (phs_class) against the calculated CTG rule percentages.
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        import numpy as np

        color = {
            "baseline": "#2849C1",
            "variability": "#E04A3F",
            "deceleration": "#388E3C",
            "conclusion": "#F58220",
        }

        # Retrieve and binarize the ground truth pH labels based on the ph_limit threshold
        # (Assuming self.phs contains the pH values of the dataset)
        phs_class = np.where(self.get_pH() < ph_limit, 1.0, 0.0)

        plt.figure(figsize=(8, 6))

        for key, value in data_perc.items():
            if only_conclusion and key != "conclusion":
                continue

            # Convert to numpy arrays to guarantee alignment and safe indexing
            y_true = np.array(phs_class, dtype=float)
            y_score = np.array(value, dtype=float)

            # 1. Align and filter out NaNs from both arrays simultaneously
            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_score)
            y_true_clean = y_true[valid_mask]
            y_score_clean = y_score[valid_mask]

            # 2. Critical Validation: Ensure we have samples and both classes are present (0 and 1)
            unique_classes = np.unique(y_true_clean)

            if len(y_true_clean) == 0:
                print(
                    f"[WARN - {key}] No valid pH or CTG percentage data found for ROC. Skipping..."
                )
                continue

            if len(unique_classes) < 2:
                print(
                    f"[WARN - {key}] ROC requires both classes (0 and 1). Only found {unique_classes}. Skipping..."
                )
                continue

            # 3. Compute ROC curve safely
            fpr, tpr, _ = roc_curve(y_true_clean, y_score_clean, pos_label=1)

            # Calculate Area Under the Curve (AUC)
            roc_auc = auc(fpr, tpr)

            # Plot individual feature curve
            plt.plot(
                fpr,
                tpr,
                color=color.get(key, "#757575"),
                lw=2,
                label=f"{key.capitalize()} (AUC = {roc_auc:.2f})",
            )

        # Draw the diagonal random-guess reference line
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"{title} (pH Limit: {ph_limit})", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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

        plt.legend(loc="upper right", fontsize=25, frameon=True)

        # Estilo de cuadrícula sutil
        # plt.grid(True, linestyle="--", alpha=0.4, axis="y")
        plt.tight_layout()

        plt.tick_params(axis="both", labelsize=25)

        # 8. Guardar listo para tu documento de LaTeX
        plt.savefig("ph_categories_boxplot.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    ## ---------------------------------------------
    ## --------------- MI FUNCTION -----------------

    def _get_mutual_information(self):

        # if not hasattr(self, ".features"):
        # self.get_all_features()

        nombres_features = self.names_features
        X = self.features
        y = self.get_labels_array()

        # 2. Calcular la Información Mutua
        # random_state=42 asegura que el cálculo (que usa estimación por vecinos cercanos) sea reproducible
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # 3. Crear un Pandas Series para ordenar y visualizar fácilmente los resultados
        mi_series = pd.Series(mi_scores, index=nombres_features)
        mi_series = mi_series.sort_values(ascending=False)

        print("=== PUNTAJES DE INFORMACIÓN MUTUA ===")
        print(mi_series)

        return mi_scores

    def mi_features_labels_graph(self):

        mi_scores = self._get_mutual_information()

        nombres_features = self.names_features

        df_mi = pd.DataFrame({"Feature": nombres_features, "MI": mi_scores})
        df_mi = df_mi.sort_values(by="MI", ascending=False).reset_index(drop=True)

        df_mi["Cumulative_MI"] = df_mi["MI"].cumsum()

        _, ax1 = plt.subplots(figsize=(50, 10))

        ax1.bar(
            df_mi["Feature"],
            df_mi["MI"],
            alpha=0.3,
            color="gray",
            label="MI Individual",
            width=0.4,
        )

        ax1.plot(
            df_mi["Feature"],
            df_mi["Cumulative_MI"],
            marker="o",
            linestyle="-",
            color="#17a2b8",
            linewidth=2.5,
            label="MI Acumulada",
        )

        ax1.set_ylabel("Valor de Información Mutua", fontsize=12)
        ax1.set_xlabel("Características", fontsize=12, labelpad=10)
        ax1.set_xticklabels(df_mi["Feature"], rotation=45, ha="right", fontsize=10)

        max_acumulado = df_mi["Cumulative_MI"].max()
        ax1.set_ylim(0, max_acumulado * 1.1)

        # Añadir leyenda para distinguir línea y barras
        ax1.legend(loc="upper left")

        plt.title(
            "Información Mutua Individual y Acumulada",
            fontsize=14,
            pad=15,
            fontweight="bold",
        )
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plt.savefig("curva_informacion_mutua_acumulada.pdf", format="pdf")

        return 0
