import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from prep import *

bp = load_battle_pokemon(); c = pd.read_csv(UP+"combats.csv")
c, A, B = build_battles(bp, c)
FA = pair_features(A, B); FB = pair_features(B, A)       # B-vs-A view for augmentation / symmetrisation
y = c.y.values

# ---- label noise ceiling: unordered pairs that were fought more than once
key = c.apply(lambda r: tuple(sorted((r.First_pokemon, r.Second_pokemon))), axis=1)
w = np.where(c.Winner == c.First_pokemon, c.First_pokemon, c.Second_pokemon)
d = pd.DataFrame({"k": key, "w": c.Winner.values})
g = d.groupby("k").w.agg(["size", lambda s: s.value_counts().iloc[0]])
g.columns = ["n", "top"]; rep = g[g.n > 1]
print(f"repeated pairs: {len(rep)}  | agreement of repeats on same winner: {rep.top.sum()/rep.n.sum():.3f}  "
      f"| fully consistent pairs: {(rep.top==rep.n).mean():.3f}")
print("unique unordered pairs:", len(g), "of", 800*799//2, "possible")

def fit_predict(model, tr, te):
    Xtr = pd.concat([FA.iloc[tr], FB.iloc[tr]]).values; ytr = np.r_[y[tr], 1 - y[tr]]
    model.fit(Xtr, ytr)
    p = (model.predict_proba(FA.iloc[te].values)[:, 1] + 1 - model.predict_proba(FB.iloc[te].values)[:, 1]) / 2
    return p

def report(name, yt, p):
    print(f"{name:34s} acc={accuracy_score(yt,p>.5):.4f} auc={roc_auc_score(yt,p):.4f} "
          f"logloss={log_loss(yt,p):.4f} brier={brier_score_loss(yt,p):.4f}  n={len(yt)}")

rng = np.random.default_rng(0)
mk = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
      "hist-gbm": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)}

# 1) random battle split (what most notebooks report)
perm = rng.permutation(len(c)); te = perm[:10000]; tr = perm[10000:]
print("\n== random battle split (optimistic) ==")
for n, f in mk.items(): report(n, y[te], fit_predict(f(), tr, te))

# 2) hold out Pokemon: 20% of species never seen in training
ids = bp.kid.values
res = {n: {"one": [], "both": []} for n in mk}
for seed in range(5):
    r = np.random.default_rng(seed); held = set(r.choice(ids, size=160, replace=False))
    ha = c.First_pokemon.isin(held).values; hb = c.Second_pokemon.isin(held).values
    tr = np.where(~ha & ~hb)[0]; te_any = np.where(ha | hb)[0]; te_both = np.where(ha & hb)[0]
    for n, f in mk.items():
        m = f(); p = fit_predict(m, tr, te_any)
        res[n]["one"].append((accuracy_score(y[te_any], p > .5), log_loss(y[te_any], p)))
        pb = pd.Series(p, index=te_any).loc[te_both].values
        res[n]["both"].append((accuracy_score(y[te_both], pb > .5), log_loss(y[te_both], pb)))
print("\n== hold out 20% of Pokemon (5 repeats; train only on battles between seen Pokemon) ==")
for n in mk:
    for k, lab in [("one", "≥1 unseen"), ("both", "both unseen")]:
        a = np.array(res[n][k]); print(f"{n:10s} {lab:12s} acc={a[:,0].mean():.4f}±{a[:,0].std():.4f} logloss={a[:,1].mean():.4f}")
