from collections import OrderedDict

# Default pretty names for various keys. Can be overridden by config.
PRETTY_NAMES: dict[str, str] = {
    "P@1": "P@1", "P@5": "P@5", "pairwise_f_value": "CS", "pccf": "CSCF", "gsr_score": "GSR",
    "weighted_purity": "Purity (W)", "num_clusters_found": "Clusters", "csr_score": "CSR",
    "accuracy_mean": "Accuracy", "top_5_accuracy_mean": "Top-5 Acc.",
    "cosine": "C", "euclidean": "E", "spearman": "S",
    "all": "Overall", "avian_perception": "Avian Perc.", "mouse_strain": "Mouse Strain", "mouse_identity": "Mouse ID",
    "BS1": "BS1", "BS2": "BS2", "BS3": "BS3", "BS4": "BS4", "BS5": "BS5", "BC": "BC", "ES1": "ES1",
    "HP": "HP", "HS1": "HS1", "HS2": "HS2", "HU1": "HU1", "HU2": "HU2", "HU3": "HU3", "HU4": "HU4",
    "HW1": "HW1", "HW2": "HW2", "HW3": "HW3", "HW4": "HW4", "OC1": "OC1",
}

# Column order for the large VocSim appendix tables
VOCSIM_APPENDIX_S_COLUMN_ORDER = [
    "BC", "BS1", "BS2", "BS3", "BS4", "BS5", "ES1", "HP", "HS1", "HS2", "HU1", "HU2",
    "HU3", "HU4", "HW1", "HW2", "HW3", "HW4", "OC1", "Avg", "Avg (Blind)",
]

# Subsets reserved for blind testing
BLIND_TEST_SUBSETS = ["HU3", "HU4", "HW3", "HW4"]

# Configuration for metrics to be included in the correlation analysis
METRICS_FOR_CORRELATION = OrderedDict([
    ('GSR', {'benchmark': 'GlobalSeparationRate', 'column': 'gsr_score', 'transform': lambda x: x}),
    ('Silhouette', {'benchmark': 'SilhouetteBenchmark', 'column': 'silhouette_score', 'transform': lambda x: x}),
    ('P@1', {'benchmark': 'PrecisionAtK', 'column': 'P@1', 'transform': lambda x: x}),
    ('P@5', {'benchmark': 'PrecisionAtK', 'column': 'P@5', 'transform': lambda x: x}),
    ('CSR', {'benchmark': 'ClassSeparationRatio', 'column': 'csr_score', 'transform': lambda x: x}),
    ('CS', {'benchmark': 'FValueBenchmark', 'column': 'pairwise_f_value', 'transform': lambda x: 1 - x}), 
    ('CSCF', {'benchmark': 'CSCFBenchmark', 'column': 'pccf', 'transform': lambda x: 1 - x}),
])

# Reference data from Goffinet et al. (2021)
GOFFINET_FEATURES = OrderedDict([("Spectrogram D=10", "Spectrogram D=10*"), ("Spectrogram D=30", "Spectrogram D=30*"), ("Spectrogram D=100", "Spectrogram D=100*"), ("MUPET D=9", "MUPET D=9*"), ("DeepSqueak D=10", "DeepSqueak D=10*"), ("Latent D=7", "Latent D=7*"), ("Latent D=8", "Latent D=8*")])
GOFFINET_STRAIN_DATA = { "k-NN (k=3)": {"Spectrogram D=10": "68.1 (0.2)", "Spectrogram D=30": "76.4 (0.3)", "Spectrogram D=100": "82.3 (0.5)", "MUPET D=9": "86.1 (0.2)", "DeepSqueak D=10": "79.0 (0.3)", "Latent D=7": "89.8 (0.2)"}, "k-NN (k=10)": {"Spectrogram D=10": "71.0 (0.3)", "Spectrogram D=30": "78.2 (0.1)", "Spectrogram D=100": "82.7 (0.6)", "MUPET D=9": "87.0 (0.1)", "DeepSqueak D=10": "80.7 (0.3)", "Latent D=7": "90.7 (0.4)"}, "k-NN (k=30)": {"Spectrogram D=10": "72.8 (0.3)", "Spectrogram D=30": "78.5 (0.2)", "Spectrogram D=100": "81.3 (0.5)", "MUPET D=9": "86.8 (0.2)", "DeepSqueak D=10": "81.0 (0.2)", "Latent D=7": "90.3 (0.4)"}, "RF (depth=10)": {"Spectrogram D=10": "72.8 (0.2)", "Spectrogram D=30": "76.6 (0.2)", "Spectrogram D=100": "79.1 (0.3)", "MUPET D=9": "87.4 (0.5)", "DeepSqueak D=10": "81.2 (0.4)", "Latent D=7": "88.1 (0.5)"}, "RF (depth=15)": {"Spectrogram D=10": "73.1 (0.3)", "Spectrogram D=30": "78.0 (0.3)", "Spectrogram D=100": "80.5 (0.2)", "MUPET D=9": "87.9 (0.4)", "DeepSqueak D=10": "82.1 (0.3)", "Latent D=7": "89.6 (0.4)"}, "RF (depth=20)": {"Spectrogram D=10": "73.2 (0.2)", "Spectrogram D=30": "78.3 (0.2)", "Spectrogram D=100": "80.7 (0.3)", "MUPET D=9": "87.9 (0.4)", "DeepSqueak D=10": "81.9 (0.3)", "Latent D=7": "89.6 (0.4)"}, "MLP (α=0.1)": {"Spectrogram D=10": "72.4 (0.3)", "Spectrogram D=30": "79.1 (0.4)", "Spectrogram D=100": "84.5 (0.3)", "MUPET D=9": "87.8 (0.2)", "DeepSqueak D=10": "82.1 (0.4)", "Latent D=7": "90.1 (0.3)"}, "MLP (α=0.01)": {"Spectrogram D=10": "72.3 (0.4)", "Spectrogram D=30": "78.6 (0.3)", "Spectrogram D=100": "82.9 (0.4)", "MUPET D=9": "88.1 (0.3)", "DeepSqueak D=10": "82.4 (0.4)", "Latent D=7": "90.0 (0.4)"}, "MLP (α=0.001)": {"Spectrogram D=10": "72.4 (0.4)", "Spectrogram D=30": "78.5 (0.8)", "Spectrogram D=100": "82.8 (0.1)", "MUPET D=9": "87.9 (0.2)", "DeepSqueak D=10": "81.0 (0.2)", "Latent D=7": "90.4 (0.3)"}}
GOFFINET_IDENTITY_DATA = {"Top-1 accuracy": {"MLP (α=0.01)": {"Spectrogram D=10": "9.9 (0.2)", "Spectrogram D=30": "14.9 (0.2)", "Spectrogram D=100": "20.4 (0.4)", "MUPET D=9": "14.7 (0.2)", "Latent D=8": "17.0 (0.3)"}, "MLP (α=0.001)": {"Spectrogram D=10": "10.8 (0.1)", "Spectrogram D=30": "17.3 (0.4)", "Spectrogram D=100": "25.3 (0.3)", "MUPET D=9": "19.0 (0.3)", "Latent D=8": "22.7 (0.5)"}, "MLP (α=0.0001)": {"Spectrogram D=10": "10.7 (0.2)", "Spectrogram D=30": "17.3 (0.3)", "Spectrogram D=100": "25.1 (0.3)", "MUPET D=9": "20.6 (0.4)", "Latent D=8": "24.0 (0.2)"}}, "Top-5 accuracy": {"MLP (α=0.01)": {"Spectrogram D=10": "36.6 (0.4)", "Spectrogram D=30": "45.1 (0.5)", "Spectrogram D=100": "55.0 (0.3)", "MUPET D=9": "46.5 (0.3)", "Latent D=8": "49.9 (0.4)"}, "MLP (α=0.001)": {"Spectrogram D=10": "38.6 (0.2)", "Spectrogram D=30": "50.7 (0.6)", "Spectrogram D=100": "62.9 (0.4)", "MUPET D=9": "54.0 (0.2)", "Latent D=8": "59.2 (0.6)"}, "MLP (α=0.0001)": {"Spectrogram D=10": "38.7 (0.5)", "Spectrogram D=30": "50.8 (0.3)", "Spectrogram D=100": "63.2 (0.4)", "MUPET D=9": "57.3 (0.4)", "Latent D=8": "61.6 (0.4)"}}}

# Reference data from Zandberg et al. (2024)
ZANDBERG_RESULTS = {"EMB-LUA (Zandberg et al.)": 0.727, "Luscinia-U (Zandberg et al.)": 0.698, "Luscinia (Zandberg et al.)": 0.66, "SAP (Zandberg et al.)": 0.64, "Raven (Zandberg et al.)": 0.57}

# Definitions for matching classifier results from benchmark strings
CLASSIFIERS = OrderedDict([("k-NN", OrderedDict([("k=3", {"type_match": "knn", "params_to_match": {"n_neighbors": 3}}), ("k=10", {"type_match": "knn", "params_to_match": {"n_neighbors": 10}}), ("k=30", {"type_match": "knn", "params_to_match": {"n_neighbors": 30}})])), ("RF", OrderedDict([("depth=10", {"type_match": "rf", "params_to_match": {"max_depth": 10, "class_weight": "balanced"}}), ("depth=15", {"type_match": "rf", "params_to_match": {"max_depth": 15, "class_weight": "balanced"}}), ("depth=20", {"type_match": "rf", "params_to_match": {"max_depth": 20, "class_weight": "balanced"}})])), ("MLP", OrderedDict([("α=0.1", {"type_match": "mlp", "params_to_match": {"alpha": 0.1}}), ("α=0.01", {"type_match": "mlp", "params_to_match": {"alpha": 0.01}}), ("α=0.001", {"type_match": "mlp", "params_to_match": {"alpha": 0.001}})]))])
MLP_CONFIGS = OrderedDict([("MLP (α=0.01)", {"alpha": 0.01}), ("MLP (α=0.001)", {"alpha": 0.001}), ("MLP (α=0.0001)", {"alpha": 0.0001})])