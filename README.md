# Explainable Shipment ETA and Delay Propagation

Aiqon AI Internship take-home assignment. The system predicts a maritime shipment's final delivery
delay and ETA at the moment the vessel arrives at the destination port, and explains which factors
drive the predicted remaining delay. All data are **synthetic**; see [Limitations](#synthetic-data-limitations).

## Problem framing

| Quantity | Definition |
|---|---|
| Checkpoint | Actual arrival at the destination port, `T1` |
| Observed arrival delay `A` | `T1 - scheduled port arrival` (a measured fact at the checkpoint) |
| Target `Y` | Downstream delay `D3 + D4 + D5` (port processing, customs, inland delivery vs plan) |
| Prediction | `Y_hat` from the model |
| Predicted final delay | `A + Y_hat` |
| Predicted ETA | `scheduled_delivery + A + Y_hat` (reported in UTC and destination-local time) |

Stage delays are derived from milestone timestamps (`actual - planned` stage durations), so
`final delay = D1 + ... + D5` holds exactly and no waiting time is counted in two stages.

## Architecture

```
src/
  delay_propagation.py   UTC/timezone arithmetic, customs calendar, stage durations and delays,
                         truck-window waiting (deterministic; no randomness)
  data_generator.py      synthetic generator (one seeded config, 5 independent RNG streams),
                         column schema and the canonical feature allowlist
  preprocessing.py       loading, schema/identity validation, forward folds with label embargo,
                         repeated route-stratified CV
  feature_engineering.py checkpoint feature selection (forbidden-column guard), GBR and Ridge
                         preprocessing pipelines, transformed-feature traceability
  train.py               baselines, Ridge, GBR; forward + CV evaluation; carrier ablation;
                         bootstrap CIs; final full-data artifacts
  explainability.py      SHAP for both models, driver groups, forward out-of-sample importance
  predict.py             single-shipment prediction + explanation (CLI and Python API)
  stage_analysis.py      descriptive historical stage-delay tables and figures
data/                    frozen dataset + generation config (byte-exact in git, see .gitattributes)
models/                  final full-data model artifacts (joblib)
reports/                 evaluation outputs, SHAP outputs, held-out example, figures
tests/                   pytest suite
```

## Environment setup

Tested with Python 3.12.10 on Windows: numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, shap 0.54.1,
matplotlib 3.11.2, joblib 1.6.0, tzdata 2026.6, pytest 9.2.0. `requirements.txt` is unpinned;
the saved `models/*.joblib` files are scikit-learn pickles and should be loaded with scikit-learn 1.9.1.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

All commands below run from the repository root.

## Commands

| Step | Command | Output |
|---|---|---|
| Tests | `pytest -q` | full test suite |
| Regenerate dataset | `python -m src.data_generator --seed 42 --out-dir data` | `data/shipments.csv`, `data/generation_config.json` |
| Train and evaluate | `python -m src.train` | `models/*.joblib`, `reports/{forward_predictions,model_comparison,cv_metrics,carrier_ablation}.csv`, `reports/training_metrics.json` |
| SHAP importance | `python -m src.explainability` | `reports/forward_shap_*.csv` (incl. per-shipment `forward_shap_values.csv`), `reports/insample_shap_*.csv`, `reports/shap_insample_vs_forward.csv`, `reports/figures/{forward,insample}_shap_*.png` |
| Stage analysis | `python -m src.stage_analysis` | `reports/stage_analysis.md`, `reports/figures/*.png` |
| Held-out example | `python -m src.predict --shipment-id SHP-0209 --model gbr --held-out --out reports/heldout_example_SHP-0209_gbr.json` | explanation + verification JSON |
| Dashboard | `python -m streamlit run dashboard/app.py` | interactive app at http://localhost:8501 (read-only) |

The frozen dataset hashes are
`shipments.csv` `b7dd07481d2968274e9c3c88dd0d860c352a78e70c77f4aaf6f79a7bba3f1677` and
`generation_config.json` `7a6d944bed653d300cdf605f6562af442ac9e362131e93bf78a0bdf48341742e`.
Regenerating with seed 42 reproduces both byte-for-byte; `python -m src.train` reproduces the
reports and the model files.

## Dashboard

```bat
venv\Scripts\activate
python -m streamlit run dashboard/app.py
```

Four sections, all computed from the frozen artifacts and project modules (the app never fits a model,
regenerates data or writes files):

- **Overview:** dataset size, checkpoint and target definitions, forward folds, the pooled and per-fold
  comparison of schedule-only, carry-forward, Ridge and Gradient Boosting, and limitations derived
  from the evaluation outputs.
- **Predict ETA:** a form for a new scenario (destination-local times, `A = D1 + D2`, optional
  missing congestion); returns A, `Y_hat`, `A + Y_hat`, the local ETA, the >24 h status and SHAP
  drivers. Labelled as a scenario forecast from the full-data model, not an out-of-sample result.
- **Explain prediction:** any of the 126 held-out shipments (default SHP-0209), explained by its
  forward-fold evaluation model from `reports/forward_shap_values.csv`.
- **Stage analysis:** descriptive stage-delay patterns, labelled as synthetic simulation output.

## Prediction usage

New shipment (checkpoint fields only; post-checkpoint, target and latent fields are rejected):

```bash
python -m src.predict --input shipment.json --model gbr --out result.json
```

```json
{
  "shipment_id": "NEW-0001",
  "destination_port": "Rotterdam",
  "cargo_type": "general",
  "documentation_complete": true,
  "dest_congestion_index": 0.52,
  "dest_weather_severity": 0.20,
  "planned_port_dwell_h": 60.0,
  "inland_distance_km": 180.0,
  "planned_inland_h": 12.5,
  "scheduled_port_arrival": "2026-03-02T06:00:00+00:00",
  "actual_port_arrival": "2026-03-02T20:30:00+00:00",
  "scheduled_delivery": "2026-03-06T10:00:00+00:00"
}
```

`dest_congestion_index` may be `null` (handled by imputation plus a missing indicator).
`arrival_delay_h` and `projected_nonworking_wait_h` are derived from the timestamps.

Python API:

```python
from src.predict import ShipmentPredictor, explain_held_out
result = ShipmentPredictor("gbr").predict(record)      # final full-data model
print(result.explanation, result.group_contributions_h)
result, check = explain_held_out("SHP-0209", "gbr")    # forward-fold model that never saw SHP-0209
```

Each result carries `model_scope`. The saved models were fitted on all 250 shipments, so for any
shipment in the dataset their output is labelled **in-sample**; `--held-out` uses the forward-fold
evaluation model instead.

## Evaluation methodology

- **Features (10, fixed before training):** `arrival_delay_h`, `dest_congestion_index`,
  `dest_weather_severity`, `destination_port`, `planned_port_dwell_h`, `projected_nonworking_wait_h`,
  `cargo_type`, `documentation_complete`, `inland_distance_km`, `planned_inland_h`.
- **Models:** schedule-only baseline (final delay 0), carry-forward baseline (final delay = `A`),
  Ridge (`alpha=1`), GradientBoostingRegressor (Huber, 300 trees, depth 3, lr 0.05). No tuning.
- **Primary evaluation:** three chronological forward folds by `T1` (2025 Jan-Apr, May-Aug,
  Sep-Dec; 42 test shipments each, 7 per route). Training rows must have arrived **and been
  delivered** before the test window (label-availability embargo). Each fold fits fresh pipelines.
- **Secondary:** repeated route-stratified 5x5 CV. The 25 splits overlap; their spread is not a set
  of independent replications.
- **Uncertainty:** 95% percentile bootstrap within forward folds (seed 20250), conditional on the
  three evaluated windows.

Forward results (126 held-out shipments, final-delay MAE in hours; `reports/model_comparison.csv`):

| Model | MAE [95% CI] | Skill vs carry-forward | Fold 1 / 2 / 3 MAE |
|---|---|---|---|
| Schedule-only | 35.65 [28.75, 42.96] | -0.271 | 25.69 / 37.11 / 44.16 |
| Carry-forward | 28.06 [22.72, 33.51] | 0 | 20.56 / 27.11 / 36.50 |
| Ridge | 23.54 [19.96, 27.42] | +0.161 | 22.55 / 21.73 / 26.35 |
| GBR | 23.51 [19.86, 27.56] | +0.162 | 21.89 / 18.98 / 29.64 |

Both learned models are worse than carry-forward in fold 1, beat it on large downstream delays and
lose to it on shipments whose downstream delay is near zero (`reports/training_metrics.json`).

## Explanation methodology

- SHAP explains **only `Y_hat`**. The observed `A` is reported separately and added once:
  `final = A + Y_hat`. For every explanation, `SHAP baseline + sum(contributions) = Y_hat`
  (verified to < 1e-6 h; observed errors are ~1e-13 h).
- GBR: `shap.TreeExplainer` (path-dependent). Ridge: `shap.LinearExplainer` with the full training
  rows as background (equivalent to `coef * (z - mean(z))`). Because the two methods differ,
  magnitudes are not strictly comparable across models.
- Transformed features are mapped to raw inputs and to five driver groups (upstream propagation,
  port conditions, customs risk, inland delivery, destination context) using each fitted pipeline's
  feature names; positions are never hard-coded.
- **Feature importance** = `mean(|SHAP_feature|)`.
- **Group importance** = mean absolute **net** group contribution
  `mean(|sum of the group's feature SHAP values|)`, computed per shipment. Opposite-signed
  contributions inside a group cancel, so it is not the sum of its features' mean `|SHAP|`.
- **Out-of-sample importance** (`reports/forward_shap_*`) uses the fold evaluation models on their
  126 held-out shipments. In-sample importance (`reports/insample_shap_*`) uses the final full-data
  models and is descriptive only.
- Group-level importance is stable between in-sample and held-out views; feature rankings among the
  correlated upstream terms of Ridge are not. GBR and Ridge rank the groups differently (held-out:
  GBR upstream 10.48 h vs port 10.33 h; Ridge customs 10.05 h vs port 9.25 h), so no single group is
  claimed as the top driver across models.
- SHAP values describe model behaviour, **not physical or causal effects**; correlated inputs can
  share attribution.

## Synthetic-data limitations

- Every route, port, duration, congestion, customs and carrier parameter is a synthetic assumption
  (`data/generation_config.json`), not a verified schedule or operational statistic. Port names are
  labels only.
- Mechanisms are simplified: customs starts only Mon-Fri from 08:00 local (no holidays); truck
  windows are 6 h with congestion-dependent availability sampled independently per shipment (no
  shared capacity); early cargo may be collected early; upstream factors are not seasonal.
- Models largely recover relationships the generator encodes (for example the 12 h berth-window
  effect); results show the method works on this simulation, not real-world accuracy.
- 250 shipments, 42 per forward test fold: confidence intervals are wide and only cover the three
  evaluated windows. No per-prediction uncertainty intervals are produced.
