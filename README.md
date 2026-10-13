# Pokémon Battle Win-Probability Dashboard

## Run
    pip install -r requirements.txt
    streamlit run app.py

The trained model and precomputed matchup matrix are in `artifacts/`, so the app runs without the raw CSVs.

## Rebuild from scratch
Put `pokemon.csv`, `combats.csv`, `pokemon_gen1_to_gen9.csv` where `prep.py` expects (variable `UP`), then:

    python train.py     # compares 4 models under 3 validation protocols -> artifacts/metrics.json
    python final.py     # final XGBoost, 1075x1075 matchup matrix, calibration, SHAP -> artifacts/
    python roster.py    # (optional) checks that vectorised features equal the reference features

## Files
- `prep.py`    type chart (18x18), name linking, pair features
- `roster.py`  roster of 1,025 Pokémon + 50 Mega/Primal forms, vectorised features
- `train.py`   models + validation (random split, held-out Pokémon, train Gen1-4 -> test Gen5-6)
- `final.py`   final model + artifacts
- `app.py`     Streamlit dashboard (5 pages)
- `baseline.py`, `ablate.py`  exploratory analyses (label-noise ceiling, feature ablation)
