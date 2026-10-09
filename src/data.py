import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

RANDOM_STATE = 42    
N_SAMPLES = 40_000    
CAL_SIZE = 0.2        
TEST_SIZE = 0.2

ADULT_NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                  "hours-per-week"]
ADULT_CAT_COLS = ["workclass", "education", "marital-status", "occupation",
                  "relationship", "race", "sex", "native-country"]
COVTYPE_NUM_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]
COVTYPE_CAT_COLS = ["Wilderness_Area", "Soil_Type"]

def _check_labels(name, y):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(f"{name}: target is not encoded in {{0, 1}}")
    return y.astype(int)

def load_higgs():
    path = DATA_DIR / "higgs.csv.gz"
    if not path.exists():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            raw = fetch_openml(data_id=23512, data_home=tmp, as_frame=True).frame
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(path, index=False)

    df = pd.read_csv(path).dropna()   
    return df.drop(columns="class"), _check_labels("higgs", df["class"])


def load_covertype():
    path = DATA_DIR / "covertype.csv.gz"
    if not path.exists():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            raw = fetch_covtype(data_home=tmp, as_frame=True).frame
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(path, index=False)

    df = pd.read_csv(path)
    X = df[COVTYPE_NUM_COLS].copy()
    for cat in COVTYPE_CAT_COLS:   
        block = df.filter(regex=f"^{cat}_")
        if not (block.sum(axis=1) == 1).all():
            raise RuntimeError(f"covertype: {cat} columns are not a valid one-hot encoding")
        X[cat] = block.idxmax(axis=1).str.removeprefix(f"{cat}_")
    y = (df["Cover_Type"] == 2).astype(int)
    return X, _check_labels("covertype", y)


def load_adult():
    path = DATA_DIR / "adult.csv.gz"
    if not path.exists():
        data = fetch_ucirepo(id=2).data
        DATA_DIR.mkdir(exist_ok=True)
        pd.concat([data.features, data.targets], axis=1).to_csv(path, index=False)

    df = pd.read_csv(path)
    str_cols = ADULT_CAT_COLS + ["income"]
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    df = df.replace("?", np.nan).dropna()
    y = df["income"].str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    return df[ADULT_NUM_COLS + ADULT_CAT_COLS], _check_labels("adult", y)

DATASETS = {"higgs": load_higgs, "covertype": load_covertype, "adult": load_adult}

def split_data(X, y, n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    if len(y) < n_samples:
        raise RuntimeError(f"{len(y)} rows available, {n_samples} required")
    if len(y) > n_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=random_state)

    holdout = CAL_SIZE + TEST_SIZE
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout, stratify=y, random_state=random_state)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_hold, y_hold, test_size=TEST_SIZE / holdout, stratify=y_hold,
        random_state=random_state)
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def encode_and_scale(X_train, X_cal, X_test):
    num_cols = list(X_train.select_dtypes("number").columns)
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    scaler = StandardScaler().fit(X_train[num_cols])
    if cat_cols:
        dummy_cols = pd.get_dummies(X_train[cat_cols], columns=cat_cols).columns

    def transform(X):
        num = scaler.transform(X[num_cols])
        if not cat_cols:
            return num
        cat = pd.get_dummies(X[cat_cols], columns=cat_cols).reindex(columns=dummy_cols, fill_value=0)
        return np.hstack([num, cat.to_numpy(dtype=float)])

    return transform(X_train), transform(X_cal), transform(X_test)


def get_dataset(name, n_samples=N_SAMPLES):
    X, y = DATASETS[name]()
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(X, y, n_samples)
    X_train, X_cal, X_test = encode_and_scale(X_train, X_cal, X_test)
    return X_train, X_cal, X_test, y_train, y_cal, y_test