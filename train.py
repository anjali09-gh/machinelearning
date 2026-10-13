import json, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from prep import *

BASE = ["d_hp","d_attack","d_defense","d_sp_atk","d_sp_def","d_speed","d_bst","speed_a","speed_b","a_speed_adv","d_legendary",
        "eff_ab","eff_ba","eff_diff","a_off_vs_b_def","b_off_vs_a_def","offence_ratio_log","bulk_ratio_log","d_dualtype"]
GROUPS = {"Speed": ["d_speed","speed_a","speed_b","a_speed_adv","a_speed","b_speed"],
          "Attack power": ["d_attack","d_sp_atk","a_off_vs_b_def","b_off_vs_a_def","offence_ratio_log","a_attack","b_attack","a_sp_atk","b_sp_atk"],
          "Bulk (HP/Def)": ["d_hp","d_defense","d_sp_def","bulk_ratio_log","a_hp","b_hp","a_defense","b_defense","a_sp_def","b_sp_def"],
          "Type matchup": ["eff_ab","eff_ba","eff_diff","d_dualtype"],
          "Overall / legendary": ["d_bst","d_legendary"]}

def raw_cols(A, B):
    return pd.DataFrame({f"a_{s}": A[s].values for s in STATS} | {f"b_{s}": B[s].values for s in STATS})

def features(A, B):
    F = pair_features(A, B)[BASE].copy()
    return pd.concat([F, raw_cols(A, B)], axis=1)

# ---------------- generic antisymmetric wrapper (logit space) ----------------
class Anti:
    def __init__(self, make): self.make = make
    def fit(self, XA, XB, y):
        self.m = self.make(); self.m.fit(np.vstack([XA, XB]), np.r_[y, 1 - y]); return self
    def margin(self, X):
        if hasattr(self.m, "get_booster"): return self.m.get_booster().predict(xgb.DMatrix(X), output_margin=True)
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
    def predict(self, XA, XB):
        return 1 / (1 + np.exp(-(self.margin(XA) - self.margin(XB)) / 2))

MODELS = {
 "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
 "Neural net (MLP 64-32)": lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0)),
 "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0),
 "XGBoost": lambda: xgb.XGBClassifier(n_estimators=500, learning_rate=.04, max_depth=5, subsample=.8, colsample_bytree=.8, min_child_weight=3,
                                      reg_lambda=2.0, tree_method="hist", n_jobs=4, random_state=0, verbosity=0),
}

def metrics(y, p):
    return dict(acc=accuracy_score(y, p > .5), auc=roc_auc_score(y, p), logloss=log_loss(y, p), brier=brier_score_loss(y, p))

if __name__ == "__main__":
    t0 = time.time()
    bp = load_battle_pokemon(); c = pd.read_csv(UP + "combats.csv"); c, A, B = build_battles(bp, c); y = c.y.values
    XA, XB = features(A, B), features(B, A); cols = list(XA.columns)
    out = {"features": cols}
    # ---- protocol 1: hold out 20% of Pokemon, 5 repeats
    res = {n: [] for n in MODELS}; res_both = {n: [] for n in MODELS}
    for seed in range(5):
        r = np.random.default_rng(seed); held = set(r.choice(bp.kid.values, 160, replace=False))
        ha = c.First_pokemon.isin(held).values; hb = c.Second_pokemon.isin(held).values
        tr = np.where(~ha & ~hb)[0]; te = np.where(ha | hb)[0]; tb = np.where(ha & hb)[0]
        for n, mk in MODELS.items():
            m = Anti(mk).fit(XA.iloc[tr].values, XB.iloc[tr].values, y[tr])
            p = m.predict(XA.iloc[te].values, XB.iloc[te].values)
            res[n].append(metrics(y[te], p))
            pb = pd.Series(p, index=te).loc[tb].values; res_both[n].append(metrics(y[tb], pb))
        print("seed", seed, "done", round(time.time() - t0), "s", flush=True)
    agg = lambda L: {k: [float(np.mean([d[k] for d in L])), float(np.std([d[k] for d in L]))] for k in L[0]}
    out["holdout_pokemon"] = {n: agg(v) for n, v in res.items()}
    out["holdout_pokemon_both_unseen"] = {n: agg(v) for n, v in res_both.items()}
    # ---- protocol 2: random battle split (optimistic)
    perm = np.random.default_rng(1).permutation(len(c)); te, tr = perm[:10000], perm[10000:]
    out["random_split"] = {}
    for n, mk in MODELS.items():
        m = Anti(mk).fit(XA.iloc[tr].values, XB.iloc[tr].values, y[tr]); out["random_split"][n] = metrics(y[te], m.predict(XA.iloc[te].values, XB.iloc[te].values))
    # ---- protocol 3: train on Gen1-4 Pokemon, test on matchups where both are Gen5-6 (generalisation to a new generation)
    ga, gb = A.generation.values, B.generation.values
    tr = np.where((ga <= 4) & (gb <= 4))[0]; te = np.where((ga >= 5) & (gb >= 5))[0]
    out["new_generation"] = {"n_train": int(len(tr)), "n_test": int(len(te))}
    for n, mk in MODELS.items():
        m = Anti(mk).fit(XA.iloc[tr].values, XB.iloc[tr].values, y[tr]); out["new_generation"][n] = metrics(y[te], m.predict(XA.iloc[te].values, XB.iloc[te].values))
    # ---- baselines
    sa, sb = A.speed.values, B.speed.values
    out["baseline_faster_wins_acc"] = float(((sa > sb) == y)[sa != sb].mean()); out["tie_share"] = float((sa == sb).mean())
    json.dump(out, open("artifacts/metrics.json", "w"), indent=1)
    for k in ["holdout_pokemon", "holdout_pokemon_both_unseen", "random_split", "new_generation"]:
        print("\n", k)
        for n in MODELS:
            d = out[k][n]; g = lambda x: x[0] if isinstance(x, list) else x
            print(f"  {n:24s} acc={g(d['acc']):.4f} auc={g(d['auc']):.4f} logloss={g(d['logloss']):.4f} brier={g(d['brier']):.4f}")
    print("\nfaster-wins baseline:", out["baseline_faster_wins_acc"], " total", round(time.time() - t0), "s")
