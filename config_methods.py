# BaseModel se asegura de que el formato en el que recibimos los datos es el correcto
# de no ser así lo transforma (si se puede), o envia un error.

# Así se evita que surgan errores más adelante

from pydantic import BaseModel


class SignalLimitsConfig(BaseModel):
    max_value_fhr: int
    min_value_fhr: int
    max_value_uc: int
    min_value_uc: int


class ReadDataConfig(BaseModel):
    data_folder_path: str
    cut_min_read: int


class PreprocessingConfig(BaseModel):
    prep_type: str
    cut_time: int
    max_sec_gaps: int
    rm_tail_nan: bool
    fourier_num_bases: int


class CorrelationConfig(BaseModel):
    max_sec_displacement: int
    not_nans_threshold_porc: float


class GuidelinesRulesConfig(BaseModel):
    guidelines_rules: str
    center_window: bool
    preprocess: bool
    window_min_baseline: int
    amplitude_dec_cond: int
    time_in_deceleration: int
    time_in_rep_dec: int
    correlation_threshold: float
    time_in_late_dec_cond: int
    contraction_time_shift: int
    contraction_value_threshold: int
    contraction_duration_threshold: int
    contraction_baseline_duration: int
    time_in_prolonged_dec: int
    time_in_rep_dec_for_red_var: int
    time_patholog_prolonged_dec: int
    deceleration_graph: bool
    window_var: int
    incr_var_min_time: int
    incr_var_bandwidth: int
    red_var_duration_baseline: int
    red_var_duration_deceleration: int
    red_var_bandwith: int


class FeaturesModelConfig(BaseModel):
    morphological_features: list
    linear_features: list
    nolinear_features: list
    IBTF_features: list
    clinical_features: list
    clinical_features_binary: list
    clinical_features_numeric: list


class AppConfig:
    def __init__(self, raw_dict: dict):
        self.data_folder_path = raw_dict["data_folder_path"]
        self.get_graphs = raw_dict["get_graphs"]
        self.ph_limit = raw_dict["ph_limit"]
        self.freq = raw_dict["freq"]

        self.limit = SignalLimitsConfig(**raw_dict)
        self.read = ReadDataConfig(**raw_dict)
        self.preprocessing = PreprocessingConfig(**raw_dict)
        self.correlation = CorrelationConfig(**raw_dict)
        self.rules = GuidelinesRulesConfig(**raw_dict)
        self.features = FeaturesModelConfig(**raw_dict)
