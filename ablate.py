import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from prep import *
bp = load_battle_pokemon(); c = pd.read_csv(UP+"combats.csv"); c, A, B = build_battles(bp, c); y = c.y.values
FA = pair_features(A, B); FB = pair_features(B, A)
def raw(X, Y):
    r = pd.DataFrame({f"a_{s}": X[s].values for s in STATS} | {f"b_{s}": Y[s].values for s in STATS})
    ta = np.array([type_vec(a, b) for a, b in zip(X.type1, X.type2)]); tb = np.array([type_vec(a, b) for a, b in zip(Y.type1, Y.type2)])
    for i, t in enumerate(TYPES): r[f"a_T_{t}"] = ta[:, i]; r[f"b_T_{t}"] = tb[:, i]
    return r
RA = raw(A, B); RB = raw(B, A)
sets = {"speed only": ["d_speed","speed_a","speed_b","a_speed_adv"],
        "+ stat diffs": ["d_hp","d_attack","d_defense","d_sp_atk","d_sp_def","d_speed","d_bst","speed_a","speed_b","a_speed_adv","d_legendary"],
        "+ type effectiveness/ratios": list(FA.columns)}
r = np.random.default_rng(0); held = set(r.choice(bp.kid.values, 160, replace=False))
ha = c.First_pokemon.isin(held).values; hb = c.Second_pokemon.isin(held).values
tr = np.where(~ha & ~hb)[0]; te = np.where(ha | hb)[0]
def run(Xa, Xb, name):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)
    m.fit(np.vstack([Xa.iloc[tr].values, Xb.iloc[tr].values]), np.r_[y[tr], 1 - y[tr]])
    p = (m.predict_proba(Xa.iloc[te].values)[:, 1] + 1 - m.predict_proba(Xb.iloc[te].values)[:, 1]) / 2
    print(f"{name:34s} acc={accuracy_score(y[te],p>.5):.4f} logloss={log_loss(y[te],p):.4f}")
for n, cols in sets.items(): run(FA[cols], FB[cols], n)
run(pd.concat([FA, RA], axis=1), pd.concat([FB, RB], axis=1), "engineered + raw stats + type 1-hot")
run(RA, RB, "raw stats + type one-hots only")
