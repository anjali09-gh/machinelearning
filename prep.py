"""Data prep: clean Kaggle battle data, build type chart, link to Gen1-9 CSV, build pairwise features."""
import numpy as np, pandas as pd, re, unicodedata

UP = "/mnt/user-data/uploads/"
TYPES = ["Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground","Flying",
         "Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"]
# attacker (row) -> defender (col) multiplier, modern 18-type chart
_SE = {"Normal":{},"Fire":{"Grass":2,"Ice":2,"Bug":2,"Steel":2,"Fire":.5,"Water":.5,"Rock":.5,"Dragon":.5},
"Water":{"Fire":2,"Ground":2,"Rock":2,"Water":.5,"Grass":.5,"Dragon":.5},
"Electric":{"Water":2,"Flying":2,"Electric":.5,"Grass":.5,"Dragon":.5,"Ground":0},
"Grass":{"Water":2,"Ground":2,"Rock":2,"Fire":.5,"Grass":.5,"Poison":.5,"Flying":.5,"Bug":.5,"Dragon":.5,"Steel":.5},
"Ice":{"Grass":2,"Ground":2,"Flying":2,"Dragon":2,"Fire":.5,"Water":.5,"Ice":.5,"Steel":.5},
"Fighting":{"Normal":2,"Ice":2,"Rock":2,"Dark":2,"Steel":2,"Poison":.5,"Flying":.5,"Psychic":.5,"Bug":.5,"Fairy":.5,"Ghost":0},
"Poison":{"Grass":2,"Fairy":2,"Poison":.5,"Ground":.5,"Rock":.5,"Ghost":.5,"Steel":0},
"Ground":{"Fire":2,"Electric":2,"Poison":2,"Rock":2,"Steel":2,"Grass":.5,"Bug":.5,"Flying":0},
"Flying":{"Grass":2,"Fighting":2,"Bug":2,"Electric":.5,"Rock":.5,"Steel":.5},
"Psychic":{"Fighting":2,"Poison":2,"Psychic":.5,"Steel":.5,"Dark":0},
"Bug":{"Grass":2,"Psychic":2,"Dark":2,"Fire":.5,"Fighting":.5,"Poison":.5,"Flying":.5,"Ghost":.5,"Steel":.5,"Fairy":.5},
"Rock":{"Fire":2,"Ice":2,"Flying":2,"Bug":2,"Fighting":.5,"Ground":.5,"Steel":.5},
"Ghost":{"Ghost":2,"Psychic":2,"Dark":.5,"Normal":0},
"Dragon":{"Dragon":2,"Steel":.5,"Fairy":0},
"Dark":{"Ghost":2,"Psychic":2,"Fighting":.5,"Dark":.5,"Fairy":.5},
"Steel":{"Ice":2,"Rock":2,"Fairy":2,"Fire":.5,"Water":.5,"Electric":.5,"Steel":.5},
"Fairy":{"Fighting":2,"Dragon":2,"Dark":2,"Fire":.5,"Poison":.5,"Steel":.5}}
CHART = pd.DataFrame(1.0, index=TYPES, columns=TYPES)
for a, d in _SE.items():
    for t, m in d.items(): CHART.loc[a, t] = m
TC = CHART.values

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)

STATS = ["hp","attack","defense","sp_atk","sp_def","speed"]

def load_battle_pokemon():
    p = pd.read_csv(UP+"pokemon.csv")
    p.columns = ["kid","name","type1","type2","hp","attack","defense","sp_atk","sp_def","speed","generation","legendary"]
    p.loc[p.kid == 63, "name"] = "Primeape"            # blank name in source file
    p["type2"] = p["type2"].where(p["type2"].notna(), None)
    p["legendary"] = p["legendary"].astype(int)
    p["bst"] = p[STATS].sum(axis=1)
    p["is_mega"] = p.name.str.startswith("Mega ") | p.name.str.startswith("Primal ")
    return p

def load_pool():
    g = pd.read_csv(UP+"pokemon_gen1_to_gen9.csv", keep_default_na=False, na_values=[""])
    g = g.rename(columns={"special_attack":"sp_atk","special_defense":"sp_def","primary_type":"type1","secondary_type":"type2"})
    g["type2"] = g["type2"].replace({"None": None})
    g["legendary"] = (g.is_legendary | g.is_mythical).astype(int)
    return g

MANUAL = {"Nidoran♀":"Nidoran F","Nidoran♂":"Nidoran M","Mime Jr.":"Mime Jr","Mr. Mime":"Mr Mime",
          "Frillish":"Frillish Male","Jellicent":"Jellicent Male","Pyroar":"Pyroar Male","Basculin":"Basculin Red Striped",
          "Flabébé":"Flabebe","DeoxysAttack Forme":"Deoxys Normal"}

def link(bp, pool):
    """Map each Kaggle row to a pokedex_id in the 1025 pool. Alt forms map to their base species; Megas/Primals stay unlinked."""
    pn = {norm(n): i for n, i in zip(pool.name, pool.pokedex_id)}
    species = {}
    for n, i in zip(pool.name, pool.pokedex_id):
        species.setdefault(norm(n.split()[0]), i)       # first word, e.g. 'deoxys', 'frillish'
    out = []
    for n, mega in zip(bp.name, bp.is_mega):
        if mega: out.append(None); continue
        k = norm(MANUAL.get(n, n))
        if k in pn: out.append(pn[k]); continue
        toks = n.replace("♀"," F").replace("♂"," M").split()
        hit = None
        for t in [toks[0], toks[-1]]:
            if norm(t) in species: hit = species[norm(t)]; break
        out.append(hit)
    return out

def type_vec(t1, t2):
    v = np.zeros(18)
    v[TYPES.index(t1)] = 1
    if isinstance(t2, str): v[TYPES.index(t2)] = 1
    return v

_IX = {t: i for i, t in enumerate(TYPES)}
def eff(att_types, def_types):
    """damage multiplier of the best STAB-type attack of att vs def"""
    best = 0
    for a in att_types:
        m = 1.0
        for d in def_types: m *= TC[_IX[a], _IX[d]]
        best = max(best, m)
    return best

def prep_table(df):
    """df must have type1,type2,STATS,legendary. Returns arrays used for pairwise features."""
    T = [[t for t in (a, b) if isinstance(t, str)] for a, b in zip(df.type1, df.type2)]
    return T

def pair_features(A, B):
    """A,B: DataFrames (aligned rows) of attacker/defender attributes. Returns feature DataFrame (antisymmetric-friendly)."""
    TA, TB = prep_table(A), prep_table(B)
    effAB = np.array([eff(a, b) for a, b in zip(TA, TB)])
    effBA = np.array([eff(b, a) for a, b in zip(TA, TB)])
    f = pd.DataFrame(index=range(len(A)))
    a = A[STATS].values.astype(float); b = B[STATS].values.astype(float)
    for i, s in enumerate(STATS):
        f[f"d_{s}"] = a[:, i] - b[:, i]
    f["d_bst"] = a.sum(1) - b.sum(1)
    f["speed_a"] = a[:, 5]; f["speed_b"] = b[:, 5]
    f["a_speed_adv"] = np.sign(a[:, 5] - b[:, 5])
    f["eff_ab"] = effAB; f["eff_ba"] = effBA; f["eff_diff"] = np.log2(np.maximum(effAB, .25)) - np.log2(np.maximum(effBA, .25))
    # offence of A vs defence of B (best of physical/special) and vice-versa
    f["a_off_vs_b_def"] = np.maximum(a[:,1]/np.maximum(b[:,2],1), a[:,3]/np.maximum(b[:,4],1))
    f["b_off_vs_a_def"] = np.maximum(b[:,1]/np.maximum(a[:,2],1), b[:,3]/np.maximum(a[:,4],1))
    f["offence_ratio_log"] = np.log(f.a_off_vs_b_def / f.b_off_vs_a_def)
    f["bulk_ratio_log"] = np.log((a[:,0]*(a[:,2]+a[:,4])) / (b[:,0]*(b[:,2]+b[:,4])))
    f["d_legendary"] = A.legendary.values - B.legendary.values
    f["d_dualtype"] = np.array([len(x) for x in TA]) - np.array([len(x) for x in TB])
    f = pd.concat([f, sim_features(A, B, TA, TB, effAB, effBA)], axis=1)
    return f


def sim_features(A, B, TA, TB, effAB, effBA):
    """Physics-informed features: hits-to-KO with an 80-power STAB move at L50, plus who wins the turn-order race."""
    a = A[STATS].values.astype(float); b = B[STATS].values.astype(float)
    def dmg(att, dfn, e):
        phys = ((22 * 80 * att[:, 1] / dfn[:, 2]) / 50 + 2); spec = ((22 * 80 * att[:, 3] / dfn[:, 4]) / 50 + 2)
        return np.maximum(phys, spec) * 1.5 * e
    dA = dmg(a, b, effAB); dB = dmg(b, a, effBA)
    hitsA = np.where(dA > 0, np.ceil(b[:, 0] * 2 + 110) / np.maximum(dA, 1e-6), 99)   # L50 HP = 2*base+110
    hitsB = np.where(dB > 0, np.ceil(a[:, 0] * 2 + 110) / np.maximum(dB, 1e-6), 99)
    ka, kb = np.ceil(np.minimum(hitsA, 99)), np.ceil(np.minimum(hitsB, 99))
    fa = a[:, 5] > b[:, 5]; tie = a[:, 5] == b[:, 5]
    win = np.where(fa, ka <= kb, np.where(tie, 0.5 * np.ones(len(a)), ka < kb)).astype(float)
    return pd.DataFrame({"hits_a": np.minimum(hitsA, 30), "hits_b": np.minimum(hitsB, 30),
                         "ko_ratio_log": np.log(np.minimum(hitsB, 99) / np.minimum(hitsA, 99)),
                         "sim_win": win, "ko_turn_diff": kb - ka})

def build_battles(bp, c):
    c = c.copy()
    c["y"] = (c.Winner == c.First_pokemon).astype(int)
    idx = bp.set_index("kid")
    A = idx.loc[c.First_pokemon].reset_index(); B = idx.loc[c.Second_pokemon].reset_index()
    return c, A, B

if __name__ == "__main__":
    bp = load_battle_pokemon(); pool = load_pool()
    bp["pokedex_id"] = link(bp, pool)
    print("battle roster:", len(bp), " megas/primals:", bp.is_mega.sum(),
          " linked to 1025 pool:", bp.pokedex_id.notna().sum())
    unl = bp[bp.pokedex_id.isna() & ~bp.is_mega]
    print("non-mega unlinked (alt forms):", len(unl)); print(unl.name.tolist())
    c = pd.read_csv(UP+"combats.csv")
    print("duplicates:", c.duplicated().sum(), " self battles:", (c.First_pokemon==c.Second_pokemon).sum(),
          " first-wins rate:", round((c.Winner==c.First_pokemon).mean(),4))
    print("battles per Pokemon min/median/max:", pd.concat([c.First_pokemon,c.Second_pokemon]).value_counts().agg(["min","median","max"]).tolist())
    print(CHART.loc[["Fire","Electric","Dragon"],["Water","Ground","Fairy","Steel"]])
