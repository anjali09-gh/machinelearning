"""Pokémon Battle Win-Probability Dashboard  (run:  streamlit run app.py)"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from prep import STATS, TYPES
from roster import eff_matrix, features_idx
from train import GROUPS

ART = Path(__file__).parent / "artifacts"
STAT_LABEL = {"hp": "HP", "attack": "Atk", "defense": "Def", "sp_atk": "SpA", "sp_def": "SpD", "speed": "Spe"}

st.set_page_config(page_title="Pokémon Win Predictor", page_icon="⚔️", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_all():
    R = pd.read_csv(ART / "roster.csv")
    R["type2"] = R["type2"].where(R["type2"].notna(), None)
    P = np.load(ART / "P.npy").astype(np.float32)
    bst = xgb.Booster(); bst.load_model(str(ART / "xgb.json"))
    M = json.load(open(ART / "metrics.json"))
    return R, P, bst, M


R, P, BST, MET = load_all()
COLS = MET["columns"]
N = len(R)
NAMES = R.name.tolist()
NAME2IDX = {n: i for i, n in enumerate(NAMES)}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}


@st.cache_data
def base_eff():
    return eff_matrix(R)


def explain(R_, EFF_, i, j):
    """Live model call. Returns (P(i beats j), grouped log-odds contributions favouring i)."""
    Fab = features_idx(R_, EFF_, np.array([i]), np.array([j]), COLS)
    Fba = features_idx(R_, EFF_, np.array([j]), np.array([i]), COLS)
    cab = BST.predict(xgb.DMatrix(Fab.values), pred_contribs=True)[0]
    cba = BST.predict(xgb.DMatrix(Fba.values), pred_contribs=True)[0]
    m_ab, m_ba = cab.sum(), cba.sum()
    p = 1 / (1 + np.exp(-(m_ab - m_ba) / 2))
    contrib = (cab[:-1] - cba[:-1]) / 2
    groups = {g: float(sum(c for col, c in zip(COLS, contrib) if GROUP_OF.get(col) == g)) for g in GROUPS}
    return float(p), groups, Fab.iloc[0]


def type_badge(row):
    return row.type1 if not row.type2 else f"{row.type1} / {row.type2}"


def show_card(row, col, extra=""):
    with col:
        if isinstance(row.get("sprite_url"), str):
            st.image(row["sprite_url"], width=170)
        st.markdown(f"### {row['name']}")
        st.caption(f"{type_badge(row)} · Gen {int(row['generation'])}{' · ' + extra if extra else ''}")
        st.write("  ".join(f"**{STAT_LABEL[s]}** {int(row[s])}" for s in STATS))


def radar(rows, names):
    fig = go.Figure()
    for r, n in zip(rows, names):
        v = [r[s] for s in STATS]
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=[STAT_LABEL[s] for s in STATS] + [STAT_LABEL[STATS[0]]], name=n, fill="toself", opacity=.55))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 200])), height=330, margin=dict(l=30, r=30, t=20, b=20), legend=dict(orientation="h"))
    return fig


def contribution_chart(groups, nA, nB):
    s = pd.Series(groups).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=["#2a9d8f" if v > 0 else "#e76f51" for v in s.values]))
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=f"← favours {nB}      log-odds      favours {nA} →",
                      title="What drives this prediction (Shapley contributions, grouped)")
    return fig


def flags(i_rows):
    out = []
    for r in i_rows:
        if r.get("extrapolated"):
            out.append(f"⚠️ **{r['name']}** is Gen {int(r['generation'])}: the battle data only covers Gen 1–6, so this prediction is an extrapolation.")
        if r.get("form") == "mega/primal":
            out.append(f"ℹ️ **{r['name']}** is a Mega/Primal form (sprite shown is the base species).")
    return out


def team_win(Pm, A_idx, B_idx):
    """Exact P(team A wins) in a 'winner stays on field, loser replaced in order' format (fresh HP each duel: a simplification)."""
    nA, nB = len(A_idx), len(B_idx)

    @lru_cache(None)
    def f(a, b, ia, ib):
        p = Pm[A_idx[a], B_idx[b]]
        wa = 1.0 if ib == nB else f(a, ib, ia, ib + 1)
        wb = 0.0 if ia == nA else f(ia, b, ia + 1, ib)
        return p * wa + (1 - p) * wb

    return float(f(0, 0, 1, 1))


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚔️ Win Predictor")
page = st.sidebar.radio("Page", ["Matchup simulator", "Matchup heatmap", "Tier list", "Team builder 6v6", "Model lab"])
st.sidebar.markdown("---")
st.sidebar.caption("Trained on 50,000 Kaggle battles (Gen 1–6 roster). Pokémon from Gen 7–9 are scored by extrapolation and cannot be verified.")
include_ext = st.sidebar.checkbox("Include Gen 7–9 (extrapolated) in rankings", value=False)

# ============================================================================= 1. SIMULATOR
if page == "Matchup simulator":
    st.title("Matchup simulator")
    c1, c2 = st.columns(2)
    default_a, default_b = NAME2IDX["Garchomp"], NAME2IDX["Blastoise"]
    custom = st.toggle("Build a custom Pokémon for side B (set your own stats and types)")
    a_name = c1.selectbox("Pokémon A", NAMES, index=default_a)
    R_, EFF_ = R, base_eff()
    if not custom:
        b_name = c2.selectbox("Pokémon B", NAMES, index=default_b)
        i, j = NAME2IDX[a_name], NAME2IDX[b_name]
    else:
        with c2:
            st.markdown("**Custom Pokémon (side B)**")
            t1 = st.selectbox("Type 1", TYPES, index=TYPES.index("Water"))
            t2 = st.selectbox("Type 2", ["None"] + TYPES)
            vals = {}
            cc = st.columns(3)
            for k, s in enumerate(STATS):
                vals[s] = cc[k % 3].slider(STAT_LABEL[s], 5, 255, 90, key=f"cs_{s}")
        row = dict(pokedex_id=-1, name="Custom", generation=1, type1=t1, type2=None if t2 == "None" else t2, legendary=0,
                   sprite_url=None, form="custom", in_battle_data=False, extrapolated=False, **vals)
        R_ = pd.concat([R, pd.DataFrame([row])], ignore_index=True)
        EFF_ = eff_matrix(R_)
        i, j = NAME2IDX[a_name], len(R_) - 1

    p, groups, _ = explain(R_, EFF_, i, j)
    A_, B_ = R_.iloc[i], R_.iloc[j]
    k1, k2, k3 = st.columns([2, 3, 2])
    show_card(A_, k1)
    show_card(B_, k3)
    with k2:
        st.metric(f"P({A_['name']} wins)", f"{p:.1%}")
        st.progress(float(p))
        ea, eb = EFF_[i, j], EFF_[j, i]
        st.write(f"**Type pressure:** {A_['name']} hits {B_['name']} for up to **×{ea:g}**, and is hit back for up to **×{eb:g}**.")
        if A_.speed == B_.speed:
            st.warning("Speed tie. In the training data, every one of the 1,328 tied battles was won by the *second* listed Pokémon, "
                       "so the model can only say ~50/50 here; this is a quirk of the data, not a real battle rule.")
        elif p > .5:
            top = max(groups, key=lambda g: groups[g])
            st.write(f"Biggest factor for **{A_['name']}**: *{top}*.")
        else:
            top = min(groups, key=lambda g: groups[g])
            st.write(f"Biggest factor for **{B_['name']}**: *{top}*.")
    for m in flags([A_, B_]):
        st.info(m)
    g1, g2 = st.columns(2)
    g1.plotly_chart(radar([A_, B_], [A_["name"], B_["name"]]), width="stretch")
    g2.plotly_chart(contribution_chart(groups, A_["name"], B_["name"]), width="stretch")
    st.caption("Explanations come from exact TreeSHAP on the XGBoost model, antisymmetrised so that swapping A and B flips the sign. "
               "They describe the *model*, not necessarily the true mechanics of the battle generator.")

# ============================================================================= 2. HEATMAPS
elif page == "Matchup heatmap":
    st.title("Matchup heatmaps")
    mask = np.ones(N, bool) if include_ext else (~R.extrapolated.values)
    t1, t2 = st.tabs(["Type vs type", "Pick your own roster"])
    with t1:
        st.write("Average win probability of a Pokémon with the row's primary type against one with the column's primary type.")
        Rm = R[mask]; idx = {t: np.where(mask & (R.type1.values == t))[0] for t in TYPES}
        H = np.array([[P[np.ix_(idx[a], idx[b])].mean() for b in TYPES] for a in TYPES])
        fig = px.imshow(H, x=TYPES, y=TYPES, color_continuous_scale="RdBu", zmin=.2, zmax=.8, text_auto=".2f", aspect="auto")
        fig.update_layout(height=640, xaxis_title="Defender primary type", yaxis_title="Attacker primary type")
        st.plotly_chart(fig, width="stretch")
        st.caption("A type's average win probability correlates 0.97 with its average Speed (Flying and Electric are fastest, Fairy, Steel and Rock slowest), "
                   "so this map mostly reflects Speed distributions rather than type effectiveness.")
    with t2:
        default = R[mask].assign(w=(P.sum(1) / N)[mask]).sort_values("w", ascending=False).name.head(14).tolist()
        pick = st.multiselect("Pokémon (up to 25)", NAMES, default=default, max_selections=25)
        if len(pick) >= 2:
            ids = [NAME2IDX[n] for n in pick]
            fig = px.imshow(P[np.ix_(ids, ids)], x=pick, y=pick, color_continuous_scale="RdBu", zmin=0, zmax=1, aspect="auto")
            fig.update_layout(height=720, xaxis_title="Opponent", yaxis_title="Pokémon (row's win probability)")
            st.plotly_chart(fig, width="stretch")

# ============================================================================= 3. TIER LIST
elif page == "Tier list":
    st.title("Tier list")
    N_ = N
    Rt = R.copy()
    Rt["pred_winrate"] = (P.sum(1) - .5) / (N_ - 1)
    mask_ext = np.ones(N_, bool) if include_ext else ~Rt.extrapolated.values
    # BST-adjusted residual
    fit = np.polyfit(Rt.bst, Rt.pred_winrate, 2); Rt["vs_bst_expected"] = Rt.pred_winrate - np.polyval(fit, Rt.bst)
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect("Type (either slot)", TYPES)
    gens = f2.multiselect("Generation", sorted(Rt.generation.unique()))
    kind = f3.selectbox("Show", ["All", "Legendary/Mythical only", "Non-legendary only", "Mega/Primal only"])
    d = Rt[mask_ext]
    if types: d = d[d.type1.isin(types) | d.type2.isin(types)]
    if gens: d = d[d.generation.isin(gens)]
    if kind == "Legendary/Mythical only": d = d[d.legendary == 1]
    elif kind == "Non-legendary only": d = d[d.legendary == 0]
    elif kind == "Mega/Primal only": d = d[d.form == "mega/primal"]
    d = d.sort_values("pred_winrate", ascending=False).reset_index(drop=True)
    d.insert(0, "rank", d.index + 1)
    view = d[["rank", "sprite_url", "name", "type1", "type2", "generation", "bst", "speed", "pred_winrate", "empirical_winrate", "vs_bst_expected"]]
    st.dataframe(view, width="stretch", height=520, hide_index=True,
                 column_config={"sprite_url": st.column_config.ImageColumn("", width="small"),
                                "pred_winrate": st.column_config.ProgressColumn("Predicted win rate", min_value=0, max_value=1, format="%.3f"),
                                "empirical_winrate": st.column_config.NumberColumn("Observed win rate (Kaggle)", format="%.3f"),
                                "vs_bst_expected": st.column_config.NumberColumn("vs. BST-expected", format="%+.3f",
                                                                                 help="Predicted win rate minus what a Pokémon with the same base stat total typically gets. Positive = punches above its BST.")})
    st.caption("Predicted win rate = average win probability against every other roster Pokémon (equal weighting). "
               "‘Observed’ is the win rate in the Kaggle battles (opponents there were random, so it is comparable but only for Gen 1–6 and Megas). "
               "The final model was trained on those battles, so agreement for those Pokémon is in-sample; see Model lab for honest held-out numbers.")
    s1, s2 = st.columns(2)
    dd = d.dropna(subset=["empirical_winrate"])
    if len(dd):
        fig = px.scatter(dd, x="empirical_winrate", y="pred_winrate", hover_name="name", color="speed", color_continuous_scale="Viridis", opacity=.75)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        fig.update_layout(title="Predicted vs observed win rate", height=380)
        s1.plotly_chart(fig, width="stretch")
    fig = px.scatter(d, x="bst", y="pred_winrate", hover_name="name", color="speed", color_continuous_scale="Viridis", opacity=.75)
    fig.update_layout(title="Base stat total vs predicted win rate (colour = Speed)", height=380)
    s2.plotly_chart(fig, width="stretch")
    o1, o2 = st.columns(2)
    o1.markdown("**Punch above their BST**"); o1.dataframe(d.nlargest(8, "vs_bst_expected")[["name", "bst", "speed", "pred_winrate", "vs_bst_expected"]], hide_index=True, width="stretch")
    o2.markdown("**Underperform their BST**"); o2.dataframe(d.nsmallest(8, "vs_bst_expected")[["name", "bst", "speed", "pred_winrate", "vs_bst_expected"]], hide_index=True, width="stretch")

# ============================================================================= 4. TEAM BUILDER
elif page == "Team builder 6v6":
    st.title("Team builder 6v6")
    st.write("Pick up to six per side (order = send-out order). Format: the winner of each duel stays on the field and the loser is replaced in order. "
             "HP is not carried over between duels, which is a simplification, because the data only contains full-health 1v1 battles.")
    defaults_a = ["Garchomp", "Charizard", "Alakazam", "Gengar", "Scizor", "Dragonite"]
    defaults_b = ["Blastoise", "Venusaur", "Tyranitar", "Jolteon", "Lucario", "Gyarados"]
    c1, c2 = st.columns(2)
    ta = c1.multiselect("Team A", NAMES, default=defaults_a, max_selections=6)
    tb = c2.multiselect("Team B", NAMES, default=defaults_b, max_selections=6)
    if ta and tb:
        ia, ib = [NAME2IDX[n] for n in ta], [NAME2IDX[n] for n in tb]
        pw = team_win(P, tuple(ia), tuple(ib))
        m1, m2, m3 = st.columns(3)
        m1.metric("Team A wins", f"{pw:.1%}")
        m2.metric("Team B wins", f"{1 - pw:.1%}")
        mean_pair = P[np.ix_(ia, ib)].mean()
        m3.metric("Average 1v1 win probability (A)", f"{mean_pair:.1%}")
        for r in flags([R.iloc[k] for k in ia + ib]): st.info(r)
        fig = px.imshow(P[np.ix_(ia, ib)], x=tb, y=ta, color_continuous_scale="RdBu", zmin=0, zmax=1, text_auto=".2f", aspect="auto")
        fig.update_layout(height=420, xaxis_title="Team B", yaxis_title="Team A", title="1v1 win probability of the row Pokémon")
        st.plotly_chart(fig, width="stretch")
        w = pd.DataFrame({"Pokémon": ta, "avg win prob vs Team B": P[np.ix_(ia, ib)].mean(1)}).sort_values("avg win prob vs Team B")
        st.write(f"Weakest link on Team A: **{w.iloc[0, 0]}** ({w.iloc[0, 1]:.1%} average against Team B).")
        st.markdown("### Best counters to Team B")
        o1, o2, o3 = st.columns(3)
        no_leg = o1.checkbox("Exclude legendary/mythical", True)
        no_mega = o2.checkbox("Exclude Mega/Primal", True)
        only_seen = o3.checkbox("Only Pokémon with battle data (Gen 1–6)", True)
        cand = np.ones(N, bool)
        if no_leg: cand &= R.legendary.values == 0
        if no_mega: cand &= R.form.values == "base"
        if only_seen or not include_ext: cand &= ~R.extrapolated.values
        score = P[:, ib].mean(1); worst = P[:, ib].min(1)
        out = R.assign(mean_win_vs_B=score, worst_case=worst)[cand].sort_values("mean_win_vs_B", ascending=False).head(12)
        st.dataframe(out[["sprite_url", "name", "type1", "type2", "speed", "mean_win_vs_B", "worst_case"]], hide_index=True, width="stretch",
                     column_config={"sprite_url": st.column_config.ImageColumn("", width="small"),
                                    "mean_win_vs_B": st.column_config.NumberColumn("Avg win prob vs Team B", format="%.3f"),
                                    "worst_case": st.column_config.NumberColumn("Worst matchup", format="%.3f")})

# ============================================================================= 5. MODEL LAB
else:
    st.title("Model lab")
    st.markdown("""
**Honest summary.** In this dataset the *faster* Pokémon wins about 94% of non-tied battles, so raw accuracy is a weak yardstick:
the models add roughly +1.7 points of accuracy over that one-line rule on unseen Pokémon, and a much bigger gain in the *quality of the probabilities*
(log-loss and Brier score). Repeated matchups in the data agree 98.5% of the time, so the labels look close to deterministic, and ~98% is a realistic ceiling.
""")
    hp = MET["holdout_pokemon"]; rs = MET["random_split"]; ng = MET["new_generation"]; bt = MET["holdout_pokemon_both_unseen"]
    rows = []
    for n in hp:
        rows.append({"Model": n, "Random battle split: acc": rs[n]["acc"], "Random: log-loss": rs[n]["logloss"],
                     "Held-out Pokémon (≥1 unseen): acc": hp[n]["acc"][0], "log-loss": hp[n]["logloss"][0], "Brier": hp[n]["brier"][0],
                     "Both unseen: acc": bt[n]["acc"][0], "Train Gen1–4 → test Gen5–6: acc": ng[n]["acc"], "Gen test log-loss": ng[n]["logloss"]})
    tdf = pd.DataFrame(rows)
    st.dataframe(tdf, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in tdf.columns if c != "Model"})
    st.caption("Held-out Pokémon: 20% of species removed from training (5 random repeats, mean shown); test battles involve at least one never-seen Pokémon. "
               "Accuracies include speed-tie battles, which are unpredictable (the data always awards ties to the second Pokémon).")
    a1, a2 = st.columns(2)
    with a1:
        g = pd.DataFrame(MET["by_speed_gap"]); g = g[g.gap != "0-1"]
        fig = go.Figure()
        fig.add_bar(x=g.gap, y=g.faster_wins_acc, name="‘Faster wins’ rule")
        fig.add_bar(x=g.gap, y=g.model_acc, name="XGBoost")
        fig.update_layout(barmode="group", title="Accuracy by speed gap (held-out Pokémon, non-tie)", yaxis=dict(range=[.85, 1]), xaxis_title="|Speed A − Speed B|", height=380)
        a1.plotly_chart(fig, width="stretch")
        a1.caption(f"Non-tie accuracy on held-out run: model {MET['holdout_nontie_acc_xgb']:.3f} vs faster-wins {MET['baseline_nontie_acc_same_rows']:.3f}. "
                   "The model's edge grows when the speed gap is large, i.e. when bulk/offence/type can overturn the speed advantage.")
    with a2:
        cal = pd.DataFrame(MET["calibration"])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="perfect")
        fig.add_scatter(x=cal.pred, y=cal.obs, mode="lines+markers", name="XGBoost", text=cal.n, hovertemplate="pred %{x:.2f}, observed %{y:.2f}, n=%{text}")
        fig.update_layout(title="Calibration (held-out Pokémon, non-tie)", xaxis_title="Predicted P(first wins)", yaxis_title="Observed frequency", height=380)
        a2.plotly_chart(fig, width="stretch")
        a2.caption(f"Brier {MET['brier_nontie']:.3f}, log-loss {MET['logloss_nontie']:.3f}. Slightly over-confident in the 0.5–0.8 range.")
    b1, b2 = st.columns(2)
    sg = pd.Series(MET["shap_group"]).sort_values()
    fig = px.bar(x=sg.values, y=sg.index, orientation="h", title="Global importance by feature group (mean |Shapley|, log-odds)")
    fig.update_layout(height=320, xaxis_title="mean |contribution|", yaxis_title="")
    b1.plotly_chart(fig, width="stretch")
    sf = pd.Series(MET["shap_feature"]).sort_values()
    fig = px.bar(x=sf.values, y=sf.index, orientation="h", title="Top individual features")
    fig.update_layout(height=320, xaxis_title="mean |contribution|", yaxis_title="")
    b2.plotly_chart(fig, width="stretch")
    st.markdown("### Limitations to keep in mind")
    st.markdown(f"""
- **Speed dominates.** It carries about 5× the weight of the next feature group, and a model with only speed features reaches ~93.8% accuracy.
- **Near-deterministic labels.** The battle generator is not documented. A hand-written damage-formula simulator I tried agreed with the data only ~69% of the time, so the model learns the dataset's behaviour, not official game mechanics.
- **Speed ties** ({MET['tie_battles_in_data']:,} battles) were *always* won by the second Pokémon; the dashboard reports ~50/50 and warns instead of copying this artefact.
- **Gen 7–9 are extrapolations** (304 roster entries). No battle data exists for them, so their ranks cannot be validated.
- **Stat versions differ slightly:** {MET['species_with_different_stats_between_sources']} of {MET['species_compared']} shared species have different base stats between the two files (e.g. {', '.join(MET['diff_examples'][:5])}); the roster uses your 1,025-row file.
- **Team mode** assumes fresh HP for each duel; there is no real 6v6 data to validate it.
""")
