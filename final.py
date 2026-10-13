import json, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from prep import *
from train import features, Anti, MODELS, GROUPS, metrics
from roster import build_roster, eff_matrix, features_idx
t0 = time.time()
R, bp = build_roster(); EFF = eff_matrix(R); N = len(R)
c = pd.read_csv(UP + "combats.csv"); c, A, B = build_battles(bp, c); y = c.y.values
XA, XB = features(A, B), features(B, A); COLS = list(XA.columns)

# ---- (a) calibration + non-tie accuracy on a Pokemon-holdout run (seed 0)
r = np.random.default_rng(0); held = set(r.choice(bp.kid.values, 160, replace=False))
ha = c.First_pokemon.isin(held).values; hb = c.Second_pokemon.isin(held).values
tr = np.where(~ha & ~hb)[0]; te = np.where(ha | hb)[0]
m0 = Anti(MODELS["XGBoost"]).fit(XA.iloc[tr].values, XB.iloc[tr].values, y[tr]); p0 = m0.predict(XA.iloc[te].values, XB.iloc[te].values)
nt = (A.speed.values != B.speed.values)[te]
extra = {"holdout_nontie_acc_xgb": float(accuracy_score(y[te][nt], p0[nt] > .5)),
         "baseline_nontie_acc_same_rows": float(((A.speed.values > B.speed.values)[te][nt] == y[te][nt]).mean())}
bins = np.linspace(0, 1, 11); k = np.digitize(p0, bins[1:-1])
pn_, yn_, kn_ = p0[nt], y[te][nt], k[nt]
extra["calibration"] = [{"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "pred": float(pn_[kn_ == i].mean()), "obs": float(yn_[kn_ == i].mean()), "n": int((kn_ == i).sum())} for i in range(10) if (kn_ == i).sum() > 20]
extra["tie_battles_in_data"] = int((A.speed.values == B.speed.values).sum()); extra["tie_first_win_rate"] = float(y[A.speed.values == B.speed.values].mean())
extra["brier_nontie"] = float(brier_score_loss(yn_, pn_)); extra["logloss_nontie"] = float(log_loss(yn_, pn_))
# accuracy by speed gap (shows where the model adds value beyond 'faster wins')
gap = np.abs(A.speed.values - B.speed.values)[te]
extra["by_speed_gap"] = []
for lo, hi in [(0, 1), (1, 10), (10, 30), (30, 60), (60, 300)]:
    s = (gap >= lo) & (gap < hi)
    row = {"gap": f"{lo}-{hi}", "n": int(s.sum()), "model_acc": float(accuracy_score(y[te][s], p0[s] > .5))}
    ns = s & nt; row["faster_wins_acc"] = float(((A.speed.values > B.speed.values)[te][ns] == y[te][ns]).mean()) if ns.sum() else None
    extra["by_speed_gap"].append(row)

# ---- (b) final model on all battles
final = xgb.XGBClassifier(n_estimators=500, learning_rate=.04, max_depth=5, subsample=.8, colsample_bytree=.8, min_child_weight=3,
                          reg_lambda=2.0, tree_method="hist", n_jobs=4, random_state=0, verbosity=0)
final.fit(np.vstack([XA.values, XB.values]), np.r_[y, 1 - y]); final.get_booster().save_model("artifacts/xgb.json")
bst = final.get_booster(); print("trained", round(time.time() - t0), "s", flush=True)

# ---- (c) full round-robin matrix
I, J = np.divmod(np.arange(N * N), N)
margin = np.zeros(N * N, dtype=np.float32)
for s in range(0, N * N, 300000):
    F = features_idx(R, EFF, I[s:s+300000], J[s:s+300000], COLS)
    margin[s:s+300000] = bst.predict(xgb.DMatrix(F.values), output_margin=True)
M = margin.reshape(N, N); P = 1 / (1 + np.exp(-(M - M.T) / 2)); np.fill_diagonal(P, 0.5)
np.save("artifacts/P.npy", P.astype(np.float16)); print("matrix", round(time.time() - t0), "s", flush=True)

# ---- (d) global grouped SHAP (sample of battles) via xgboost TreeSHAP
sm = np.random.default_rng(2).choice(len(c), 4000, replace=False)
cA = bst.predict(xgb.DMatrix(XA.iloc[sm].values), pred_contribs=True)[:, :-1]; cB = bst.predict(xgb.DMatrix(XB.iloc[sm].values), pred_contribs=True)[:, :-1]
contrib = (cA - cB) / 2
grp_of = {col: g for g, cs in GROUPS.items() for col in cs}
# map raw columns a_*/b_* to group by stat
def group_of(col):
    if col in grp_of: return grp_of[col]
    return grp_of.get(col, "Other")
gi = pd.Series([group_of(cn) for cn in COLS])
imp_feat = pd.Series(np.abs(contrib).mean(0), index=COLS).sort_values(ascending=False)
imp_grp = pd.Series({g: np.abs(contrib[:, (gi == g).values].sum(1)).mean() for g in GROUPS})
extra["shap_feature"] = imp_feat.head(15).round(4).to_dict(); extra["shap_group"] = imp_grp.round(4).sort_values(ascending=False).to_dict()
extra["unassigned_cols"] = [cn for cn, g in zip(COLS, gi) if g == "Other"]

# ---- (e) sanity: predicted vs empirical win rate for roster members seen in battles
emp = pd.concat([pd.DataFrame({"kid": c.First_pokemon, "w": c.y}), pd.DataFrame({"kid": c.Second_pokemon, "w": 1 - c.y})]).groupby("kid").w.mean()
bp2 = bp.set_index("kid"); R["pred_winrate"] = (P.sum(1) - 0.5) / (N - 1)
rid_of_kid = {}
for kid, row in bp2.iterrows():
    if kid not in emp.index: continue
    idx = R.index[(R.pokedex_id == (10000 + kid))] if row.is_mega else (R.index[R.pokedex_id == int(row.pokedex_id)] if pd.notna(row.pokedex_id) else [])
    if len(idx): rid_of_kid[kid] = idx[0]
tmp = pd.DataFrame({"ridx": list(rid_of_kid.values()), "emp": emp.loc[list(rid_of_kid)].values}).groupby("ridx").emp.mean()
pairs = pd.DataFrame({"emp": tmp.values, "pred": R.pred_winrate.loc[tmp.index].values})
extra["corr_pred_vs_empirical_winrate"] = float(pairs.corr().iloc[0, 1]); extra["n_species_compared"] = len(pairs)
extra["no_battles_in_data"] = int((~bp2.index.isin(emp.index)).sum())
R["empirical_winrate"] = np.nan; R.loc[tmp.index, "empirical_winrate"] = tmp.values

# ---- (f) data quality: do shared Pokemon have different stats in the two sources?
base = R[R.form == "base"].set_index("pokedex_id"); bpl = bp[bp.pokedex_id.notna() & ~bp.is_mega].copy(); bpl["pokedex_id"] = bpl.pokedex_id.astype(int)
bpl = bpl.drop_duplicates("pokedex_id").set_index("pokedex_id"); j = base.join(bpl[STATS], rsuffix="_k", how="inner")
diff = (j[[s for s in STATS]].values != j[[s + "_k" for s in STATS]].values).any(1)
extra["species_with_different_stats_between_sources"] = int(diff.sum()); extra["species_compared"] = int(len(j))
extra["diff_examples"] = j[diff].name.head(12).tolist()

R.to_csv("artifacts/roster.csv", index=False)
m = json.load(open("artifacts/metrics.json")); m.update(extra); m["columns"] = COLS; json.dump(m, open("artifacts/metrics.json", "w"), indent=1)
print(json.dumps({k: extra[k] for k in ["holdout_nontie_acc_xgb", "baseline_nontie_acc_same_rows", "shap_group", "corr_pred_vs_empirical_winrate", "n_species_compared", "species_with_different_stats_between_sources", "species_compared", "diff_examples", "unassigned_cols"]}, indent=1))
print(pd.DataFrame(extra["by_speed_gap"]).to_string()); print(pd.DataFrame(extra["calibration"]).to_string())
print("done", round(time.time() - t0), "s")
