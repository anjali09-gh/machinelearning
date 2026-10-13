"""Roster (1025 Gen1-9 + 50 Mega/Primal from battle data) and fully vectorised pair features."""
import numpy as np, pandas as pd
from prep import *

def build_roster():
    bp = load_battle_pokemon(); pool = load_pool(); bp["pokedex_id"] = link(bp, pool)
    keep = ["pokedex_id","name","generation","type1","type2","hp","attack","defense","sp_atk","sp_def","speed","legendary"]
    R = pool[keep + ["sprite_url"]].copy(); R["form"] = "base"
    R["in_battle_data"] = R.pokedex_id.isin(bp.pokedex_id.dropna().astype(int))
    M = bp[bp.is_mega].copy()
    base_id = {}
    for n in M.name:
        b = n.replace("Mega ", "").replace("Primal ", "").split()[0]
        base_id[n] = int(pool.loc[pool.name.map(norm) == norm(b), "pokedex_id"].iloc[0]) if (pool.name.map(norm) == norm(b)).any() else None
    M["pokedex_id"] = 10000 + M.kid; M["form"] = "mega/primal"; M["in_battle_data"] = True
    M["sprite_url"] = [pool.set_index("pokedex_id").sprite_url.get(base_id[n], None) if base_id[n] else None for n in M.name]
    M = M[keep + ["sprite_url", "form", "in_battle_data"]]
    R = pd.concat([R, M], ignore_index=True)
    R["bst"] = R[STATS].sum(axis=1); R["extrapolated"] = R.generation >= 7
    R["type2"] = R["type2"].where(R["type2"].notna(), None)
    return R.reset_index(drop=True), bp

def eff_matrix(R):
    """EFF[i,j] = best damage multiplier of i's own type(s) hitting j."""
    ti = {t: k for k, t in enumerate(TYPES)}
    t1 = R.type1.map(ti).values; has2 = R.type2.map(lambda x: isinstance(x, str)).values
    t2 = np.array([ti[x] if isinstance(x, str) else 0 for x in R.type2])
    D = TC[:, t1] * np.where(has2[None, :], TC[:, t2], 1.0)          # D[a, j]
    e1 = D[t1, :]; e2 = np.where(has2[:, None], D[t2, :], 0.0)
    return np.maximum(e1, e2)

def features_idx(R, EFF, I, J, BASE):
    a = R[STATS].values.astype(float)[I]; b = R[STATS].values.astype(float)[J]
    leg = R.legendary.values; n2 = R.type2.map(lambda x: isinstance(x, str)).values.astype(int) + 1
    eab = EFF[I, J]; eba = EFF[J, I]
    f = {}
    for k, s in enumerate(STATS): f[f"d_{s}"] = a[:, k] - b[:, k]
    f["d_bst"] = a.sum(1) - b.sum(1); f["speed_a"] = a[:, 5]; f["speed_b"] = b[:, 5]; f["a_speed_adv"] = np.sign(a[:, 5] - b[:, 5])
    f["eff_ab"] = eab; f["eff_ba"] = eba; f["eff_diff"] = np.log2(np.maximum(eab, .25)) - np.log2(np.maximum(eba, .25))
    f["a_off_vs_b_def"] = np.maximum(a[:,1]/np.maximum(b[:,2],1), a[:,3]/np.maximum(b[:,4],1))
    f["b_off_vs_a_def"] = np.maximum(b[:,1]/np.maximum(a[:,2],1), b[:,3]/np.maximum(a[:,4],1))
    f["offence_ratio_log"] = np.log(f["a_off_vs_b_def"] / f["b_off_vs_a_def"])
    f["bulk_ratio_log"] = np.log((a[:,0]*(a[:,2]+a[:,4])) / (b[:,0]*(b[:,2]+b[:,4])))
    f["d_legendary"] = leg[I] - leg[J]; f["d_dualtype"] = n2[I] - n2[J]
    for k, s in enumerate(STATS): f[f"a_{s}"] = a[:, k]
    for k, s in enumerate(STATS): f[f"b_{s}"] = b[:, k]
    df = pd.DataFrame(f)
    return df[BASE]

if __name__ == "__main__":
    from train import features
    R, bp = build_roster(); print(R.shape, R.form.value_counts().to_dict(), "extrapolated:", int(R.extrapolated.sum()))
    EFF = eff_matrix(R)
    rng = np.random.default_rng(0); I = rng.integers(0, len(R), 3000); J = rng.integers(0, len(R), 3000)
    A = R.iloc[I].reset_index(drop=True); B = R.iloc[J].reset_index(drop=True)
    F_old = features(A, B); F_new = features_idx(R, EFF, I, J, list(F_old.columns))
    print("max abs diff old vs vectorised features:", float(np.abs(F_old.values - F_new.values).max()))
    print("Charizard vs Blastoise eff:", EFF[R.index[R.name=="Charizard"][0], R.index[R.name=="Blastoise"][0]],
          "| Pikachu vs Gyarados:", EFF[R.index[R.name=="Pikachu"][0], R.index[R.name=="Gyarados"][0]],
          "| Pikachu vs Garchomp:", EFF[R.index[R.name=="Pikachu"][0], R.index[R.name=="Garchomp"][0]])
    print(R[R.form!="base"].head(3)[["name","type1","type2","sprite_url"]].to_string())
