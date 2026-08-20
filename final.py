
import warnings, random, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

try:
    from lightgbm import LGBMClassifier
    import xgboost as xgb
    HAS_BOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_BOOST = False
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)

TRAIN_FILE = "DataSetbearing-failure.csv"
TEST_FILE = "Star_test.csv"
OUT_FILE = "submission_Star_test.csv"

OC = ["Vel, Rms (RMS)", "Acc, Rms (RMS)", "Crest (RMS)", "Kurt (RMS)",
      "Vel, Peak (RMS)", "Vel, Peak to peak (RMS)"]
MATCH_COLS = ["COMP_NAME"] + OC + ["MP_LOC"]
eps = 1e-9

def load(path):
    try: return pd.read_csv(path)
    except Exception: return pd.read_csv(path, encoding="utf-16", sep="\t")

train_df = load(TRAIN_FILE).dropna(subset=["Label"]).reset_index(drop=True)
train_df["Label"] = train_df["Label"].astype(int)
test_df = load(TEST_FILE).reset_index(drop=True)

# ---------- 1. exact lookup ----------
lookup = {}
for _, row in train_df.iterrows():
    key = tuple(row[c] for c in MATCH_COLS)
    lbl = int(row["Label"])
    if key not in lookup:
        lookup[key] = lbl
    elif lookup[key] != lbl:
        lookup[key] = None

test_keys = test_df[MATCH_COLS].apply(tuple, axis=1)
exact_labels = np.array([lookup.get(k, -1) if lookup.get(k) is not None else -1 for k in test_keys])
exact_mask = np.array([k in lookup and lookup[k] is not None for k in test_keys])
print(f"Exact-match: {exact_mask.sum()} / {len(test_df)}")

def _base_feats(d):
    d = d.copy()
    d["peak_to_rms"] = d["Vel, Peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["pp_to_rms"] = d["Vel, Peak to peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["acc_vel_ratio"] = d["Acc, Rms (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["severity_index"] = d["Vel, Rms (RMS)"] * d["Acc, Rms (RMS)"] * d["Crest (RMS)"]
    d["kurt_crest"] = d["Kurt (RMS)"] * d["Crest (RMS)"]
    return d

_feat_cols = OC + ["peak_to_rms","pp_to_rms","acc_vel_ratio","severity_index","kurt_crest"]
_X = _base_feats(train_df)
_dup_key = train_df[MATCH_COLS].apply(tuple, axis=1)
_row_group = pd.factorize(_dup_key)[0]
_gkf = GroupKFold(n_splits=5)
_cv_scores = []
for _tr, _te in _gkf.split(_X, y_tr, groups=_row_group):
    _Xtr_df, _Xte_df = _X.iloc[_tr], _X.iloc[_te]
    _ytr, _yte = y_tr[_tr], y_tr[_te]
    _scaler = RobustScaler().fit(_Xtr_df[_feat_cols])
    _Xtr_s, _Xte_s = _scaler.transform(_Xtr_df[_feat_cols]), _scaler.transform(_Xte_df[_feat_cols])
    if HAS_BOOST:
        _m1 = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, max_depth=5,
                              min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                              class_weight="balanced", random_state=SEED, verbose=-1)
        _m2 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.8,
                                 colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss")
    else:
        _m1 = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.04, max_iter=300, random_state=SEED)
        _m2 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    _m1.fit(_Xtr_s, _ytr); _m2.fit(_Xtr_s, _ytr)
    _p_model = 0.5*_m1.predict_proba(_Xte_s) + 0.5*_m2.predict_proba(_Xte_s)

    _nn_scaler = StandardScaler().fit(train_df.iloc[_tr][OC])
    _Xtr_raw = _nn_scaler.transform(train_df.iloc[_tr][OC])
    _Xte_raw = _nn_scaler.transform(train_df.iloc[_te][OC])
    _comp_tr, _loc_tr = train_df.iloc[_tr]["COMP_NAME"].values, train_df.iloc[_tr]["MP_LOC"].values
    _comp_te, _loc_te = train_df.iloc[_te]["COMP_NAME"].values, train_df.iloc[_te]["MP_LOC"].values

    _final_probs = np.zeros((len(_te), 3))
    for i in range(len(_te)):
        _idx = np.where((_comp_tr==_comp_te[i]) & (_loc_tr==_loc_te[i]))[0]
        if len(_idx)==0: _idx = np.where(_comp_tr==_comp_te[i])[0]
        if len(_idx)==0: _idx = np.arange(len(_tr))
        _Xsub = _Xtr_raw[_idx]
        _d = np.linalg.norm(_Xsub - _Xte_raw[[i]], axis=1)
        _k = min(15, len(_idx))
        _nn_idx = np.argsort(_d)[:_k]
        _nn_d, _nn_y = _d[_nn_idx], _ytr[_idx][_nn_idx]
        _w = 1.0/(_nn_d+1e-5); _w /= _w.sum()
        _knn_prob = np.zeros(3)
        for _lbl, _wt in zip(_nn_y, _w): _knn_prob[_lbl] += _wt
        _min_d = _nn_d[0]
        _alpha = 0.80 if _min_d<0.5 else 0.55 if _min_d<1.5 else 0.30 if _min_d<3.0 else 0.10
        _final_probs[i] = _alpha*_knn_prob + (1-_alpha)*_p_model[i]
    _pred = np.argmax(_final_probs, axis=1)
    _cv_scores.append(f1_score(_yte, _pred, average="macro"))

print(f"F1-Macro (CV, duplicate-safe GroupKFold, model+KNN layer only): "
      f"{np.mean(_cv_scores):.4f} +/- {np.std(_cv_scores):.4f}")
print(f"  Fold scores: {[round(s,4) for s in _cv_scores]}")
print("  NOTE: this measures the fallback layer's real generalization.")
print("  The exact-lookup layer cannot be honestly measured via CV since")
print("  duplicate rows are kept together across folds by construction.")

# ---------- 0. Honest CV of the model+KNN layer ----------
def _base_feats(d):
    d = d.copy()
    d["peak_to_rms"] = d["Vel, Peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["pp_to_rms"] = d["Vel, Peak to peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["acc_vel_ratio"] = d["Acc, Rms (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["severity_index"] = d["Vel, Rms (RMS)"] * d["Acc, Rms (RMS)"] * d["Crest (RMS)"]
    d["kurt_crest"] = d["Kurt (RMS)"] * d["Crest (RMS)"]
    return d

_feat_cols = OC + ["peak_to_rms","pp_to_rms","acc_vel_ratio","severity_index","kurt_crest"]
_X = _base_feats(train_df)
_dup_key = train_df[MATCH_COLS].apply(tuple, axis=1)
_row_group = pd.factorize(_dup_key)[0]
_y_tr = train_df["Label"].values 

_gkf = GroupKFold(n_splits=5)
_cv_scores = []
for _tr, _te in _gkf.split(_X, _y_tr, groups=_row_group):  # استفاده از _y_tr
    _Xtr_df, _Xte_df = _X.iloc[_tr], _X.iloc[_te]
    _ytr, _yte = _y_tr[_tr], _y_tr[_te]  # استفاده از _y_tr
    _scaler = RobustScaler().fit(_Xtr_df[_feat_cols])
    _Xtr_s, _Xte_s = _scaler.transform(_Xtr_df[_feat_cols]), _scaler.transform(_Xte_df[_feat_cols])
    
    if HAS_BOOST:
        _m1 = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, max_depth=5,
                              min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                              class_weight="balanced", random_state=SEED, verbose=-1)
        _m2 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.8,
                                 colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss")
    else:
        _m1 = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.04, max_iter=300, random_state=SEED)
        _m2 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    
    _m1.fit(_Xtr_s, _ytr)
    _m2.fit(_Xtr_s, _ytr)
    _p_model = 0.5*_m1.predict_proba(_Xte_s) + 0.5*_m2.predict_proba(_Xte_s)

    _nn_scaler = StandardScaler().fit(train_df.iloc[_tr][OC])
    _Xtr_raw = _nn_scaler.transform(train_df.iloc[_tr][OC])
    _Xte_raw = _nn_scaler.transform(train_df.iloc[_te][OC])
    _comp_tr, _loc_tr = train_df.iloc[_tr]["COMP_NAME"].values, train_df.iloc[_tr]["MP_LOC"].values
    _comp_te, _loc_te = train_df.iloc[_te]["COMP_NAME"].values, train_df.iloc[_te]["MP_LOC"].values

    _final_probs = np.zeros((len(_te), 3))
    for i in range(len(_te)):
        _idx = np.where((_comp_tr==_comp_te[i]) & (_loc_tr==_loc_te[i]))[0]
        if len(_idx)==0: 
            _idx = np.where(_comp_tr==_comp_te[i])[0]
        if len(_idx)==0: 
            _idx = np.arange(len(_tr))
        _Xsub = _Xtr_raw[_idx]
        _d = np.linalg.norm(_Xsub - _Xte_raw[[i]], axis=1)
        _k = min(15, len(_idx))
        _nn_idx = np.argsort(_d)[:_k]
        _nn_d, _nn_y = _d[_nn_idx], _ytr[_idx][_nn_idx]
        _w = 1.0/(_nn_d+1e-5)
        _w /= _w.sum()
        _knn_prob = np.zeros(3)
        for _lbl, _wt in zip(_nn_y, _w): 
            _knn_prob[_lbl] += _wt
        _min_d = _nn_d[0]
        _alpha = 0.80 if _min_d<0.5 else 0.55 if _min_d<1.5 else 0.30 if _min_d<3.0 else 0.10
        _final_probs[i] = _alpha*_knn_prob + (1-_alpha)*_p_model[i]
    
    _pred = np.argmax(_final_probs, axis=1)
    _cv_scores.append(f1_score(_yte, _pred, average="macro"))

print(f"F1-Macro (CV, duplicate-safe GroupKFold, model+KNN layer only): "
      f"{np.mean(_cv_scores):.4f} +/- {np.std(_cv_scores):.4f}")


# ---------- 2 & 3. trained model + distance-weighted neighbor blend ----------
def base_features(d):
    d = d.copy()
    d["peak_to_rms"] = d["Vel, Peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["pp_to_rms"] = d["Vel, Peak to peak (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["acc_vel_ratio"] = d["Acc, Rms (RMS)"] / (d["Vel, Rms (RMS)"] + eps)
    d["severity_index"] = d["Vel, Rms (RMS)"] * d["Acc, Rms (RMS)"] * d["Crest (RMS)"]
    d["kurt_crest"] = d["Kurt (RMS)"] * d["Crest (RMS)"]
    return d

X_tr = base_features(train_df)
X_te = base_features(test_df)
feat_cols = OC + ["peak_to_rms","pp_to_rms","acc_vel_ratio","severity_index","kurt_crest"]
scaler = RobustScaler().fit(X_tr[feat_cols])
Xtr_s, Xte_s = scaler.transform(X_tr[feat_cols]), scaler.transform(X_te[feat_cols])
y_tr = train_df["Label"].values

if HAS_BOOST:
    m1 = LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, max_depth=5,
                         min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                         class_weight="balanced", random_state=SEED, verbose=-1)
    m2 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.04, subsample=0.8,
                            colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss")
else:
    m1 = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.04, max_iter=300, random_state=SEED)
    m2 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
m1.fit(Xtr_s, y_tr); m2.fit(Xtr_s, y_tr)
p_model = 0.5*m1.predict_proba(Xte_s) + 0.5*m2.predict_proba(Xte_s)

nn_scaler = StandardScaler().fit(train_df[OC])
Xtr_raw = nn_scaler.transform(train_df[OC])
Xte_raw = nn_scaler.transform(test_df[OC])

final_probs = np.zeros((len(test_df), 3))
for i in range(len(test_df)):
    c, loc = test_df.iloc[i]["COMP_NAME"], test_df.iloc[i]["MP_LOC"]
    idx = np.where((train_df["COMP_NAME"]==c)&(train_df["MP_LOC"]==loc))[0]
    if len(idx)==0: idx = np.where(train_df["COMP_NAME"]==c)[0]
    if len(idx)==0: idx = np.arange(len(train_df))
    Xsub = Xtr_raw[idx]
    d = np.linalg.norm(Xsub - Xte_raw[[i]], axis=1)
    k = min(15, len(idx))
    nn_idx = np.argsort(d)[:k]
    nn_d, nn_y = d[nn_idx], y_tr[idx][nn_idx]
    w = 1.0/(nn_d+1e-5); w /= w.sum()
    knn_prob = np.zeros(3)
    for lbl, wt in zip(nn_y, w): knn_prob[lbl]+=wt
    min_d = nn_d[0]
    alpha = 0.80 if min_d<0.5 else 0.55 if min_d<1.5 else 0.30 if min_d<3.0 else 0.10
    final_probs[i] = alpha*knn_prob + (1-alpha)*p_model[i]

y_pred = np.argmax(final_probs, axis=1)
y_pred[exact_mask] = exact_labels[exact_mask]

print(f"Class 0: {(y_pred==0).sum()}  Class 1: {(y_pred==1).sum()}  Class 2: {(y_pred==2).sum()}")

submission = test_df.copy()
submission["Label"] = y_pred.astype(int)
submission.to_csv(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE}")