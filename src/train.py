"""Leakage-safe training and evaluation of the downstream-delay models.

Models predict the downstream delay ``Y = D3 + D4 + D5`` at the port-arrival
checkpoint. The final-delay prediction adds the observed arrival delay::

    predicted_final_delay_h = arrival_delay_h + predicted_downstream_delay_h
    predicted_eta           = scheduled_delivery + predicted_final_delay_h

Because ``A`` is observed, the final-delay error equals the downstream error.

Evaluation
----------
Primary: three chronological forward folds (``preprocessing.forward_folds``)
with a label-availability embargo; a fresh pipeline is fitted per fold on
that fold's training rows only, producing one out-of-sample prediction per
2025 shipment.

Secondary: repeated route-stratified 5x5 CV. The 25 splits overlap, so their
spread describes split-to-split variability, not independent replications.

Final artifacts are refitted on all 250 rows *after* evaluation; they are
never used to produce evaluation predictions.

All results are on synthetic data generated under documented assumptions and
do not establish performance on real logistics data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from src import data_generator as dg
from src import delay_propagation as dp
from src import feature_engineering as fe
from src import preprocessing as pp

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

GBR_PARAMS: dict = {
    "loss": "huber", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
    "subsample": 0.8, "min_samples_leaf": 10, "random_state": 42,
}
RIDGE_PARAMS: dict = {"alpha": 1.0}
MODEL_NAMES = ["schedule_only", "carry_forward", "ridge", "gbr"]
LEARNED_MODELS = ["ridge", "gbr"]
REFERENCE_MODEL = "carry_forward"
DELAY_THRESHOLD_H = float(dg.DEFAULT_CONFIG["reporting"]["delayed_threshold_h"])
BOOTSTRAP_SEED = 20250
N_BOOTSTRAP = 2000
ABLATION_MATERIALITY = 0.02  # +/-2% of primary MAE, declared before running


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_gbr_pipeline(extra_categorical: Sequence[str] = ()) -> Pipeline:
    """Fresh GBR pipeline; ``extra_categorical`` is only used by the carrier ablation."""
    pre = fe.build_gbr_preprocessor(fe.NUMERIC_FEATURES, fe.CATEGORICAL_FEATURES + list(extra_categorical))
    return Pipeline([("preprocess", pre), ("model", GradientBoostingRegressor(**GBR_PARAMS))])


def make_ridge_pipeline() -> Pipeline:
    """Fresh Ridge pipeline."""
    return Pipeline([("preprocess", fe.build_ridge_preprocessor()), ("model", Ridge(**RIDGE_PARAMS))])


PIPELINE_FACTORIES: dict[str, Callable[[], Pipeline]] = {"ridge": make_ridge_pipeline, "gbr": make_gbr_pipeline}


def baseline_downstream(df: pd.DataFrame, name: str) -> np.ndarray:
    """Baseline downstream predictions: schedule-only (``-A``) or carry-forward (0)."""
    a = df["arrival_delay_h"].to_numpy(dtype=float)
    if name == "schedule_only":
        return -a  # predicted final delay = 0
    if name == "carry_forward":
        return np.zeros_like(a)  # predicted final delay = A
    raise ValueError(f"unknown baseline {name!r}")


def predicted_final_delay(arrival_delay_h: np.ndarray, predicted_downstream_h: np.ndarray) -> np.ndarray:
    """``A + Y_hat``."""
    return np.asarray(arrival_delay_h, dtype=float) + np.asarray(predicted_downstream_h, dtype=float)


def predicted_eta(scheduled_delivery: pd.Series, predicted_final_h: np.ndarray) -> pd.Series:
    """``scheduled_delivery + predicted final delay`` (UTC), via delay_propagation arithmetic."""
    return pd.Series(
        [dp.add_hours(s, float(h)) for s, h in zip(scheduled_delivery, predicted_final_h)],
        index=scheduled_delivery.index, dtype=scheduled_delivery.dtype,
    )


def fit_predict(name: str, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame,
                train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, Pipeline | None]:
    """Fit ``name`` on ``train_idx`` only and predict downstream delay for ``test_idx``."""
    if name in ("schedule_only", "carry_forward"):
        return baseline_downstream(df.iloc[test_idx], name), None
    pipe = PIPELINE_FACTORIES[name]()
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    return pipe.predict(X.iloc[test_idx]), pipe


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(actual_final: np.ndarray, pred_final: np.ndarray) -> dict[str, float]:
    """Error metrics on final delivery delay (hours)."""
    err = np.abs(np.asarray(pred_final, float) - np.asarray(actual_final, float))
    return {
        "n": int(err.size),
        "mae_h": float(err.mean()),
        "rmse_h": float(np.sqrt(np.mean(err ** 2))),
        "median_ae_h": float(np.median(err)),
        "p90_ae_h": float(np.percentile(err, 90)),
        "within_12h_pct": float(100 * np.mean(err <= 12)),
        "within_24h_pct": float(100 * np.mean(err <= 24)),
    }


def skill_score(model_mae: float, reference_mae: float) -> float:
    """``1 - model_MAE / reference_MAE`` (positive = better than the reference)."""
    if reference_mae <= 0:
        raise ValueError("reference MAE must be positive")
    return 1.0 - model_mae / reference_mae


def flag_metrics(actual_final: np.ndarray, pred_final: np.ndarray,
                 threshold: float = DELAY_THRESHOLD_H) -> dict[str, float | int | None]:
    """Predicted-late flag (``pred > threshold``) and ranking quality of the continuous score.

    ROC-AUC / PR-AUC treat the predicted final delay as a *ranking score*;
    it is not a calibrated probability.
    """
    actual = np.asarray(actual_final, float) > threshold
    score = np.asarray(pred_final, float)
    predicted = score > threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    both_classes = 0 < actual.sum() < actual.size
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "roc_auc_ranking": float(roc_auc_score(actual, score)) if both_classes else None,
        "pr_auc_ranking": float(average_precision_score(actual, score)) if both_classes else None,
        "actual_positive_rate": float(actual.mean()),
    }


def bootstrap_ci(abs_err: np.ndarray, groups: np.ndarray, ref_abs_err: np.ndarray | None = None,
                 n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED, level: float = 0.95) -> dict[str, float]:
    """Percentile bootstrap CI for pooled MAE, or for paired improvement ``MAE_ref - MAE``.

    Rows are resampled with replacement *within* each group (forward fold), so
    every replicate keeps each fold's size. The same indices are used for the
    model and the reference, giving a paired interval.
    """
    abs_err = np.asarray(abs_err, float)
    groups = np.asarray(groups)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(m, size=m.size, replace=True) for m in members])
        stats[b] = abs_err[idx].mean() if ref_abs_err is None else ref_abs_err[idx].mean() - abs_err[idx].mean()
    point = abs_err.mean() if ref_abs_err is None else np.asarray(ref_abs_err, float).mean() - abs_err.mean()
    alpha = (1 - level) / 2
    return {"point": float(point), "ci_low": float(np.quantile(stats, alpha)),
            "ci_high": float(np.quantile(stats, 1 - alpha)), "n_boot": n_boot, "seed": seed}


# ---------------------------------------------------------------------------
# Forward evaluation
# ---------------------------------------------------------------------------

@dataclass
class ForwardResult:
    predictions: pd.DataFrame
    fold_pipelines: dict[tuple[int, str], Pipeline] = field(default_factory=dict)


def forward_evaluation(df: pd.DataFrame, models: Sequence[str] = MODEL_NAMES) -> ForwardResult:
    """Out-of-sample predictions for every 2025 shipment from fold-specific models."""
    X, y = fe.select_features(df), fe.get_target(df)
    frames, pipelines = [], {}
    for fold in pp.forward_folds(df):
        test = df.iloc[fold.test_idx]
        out = pd.DataFrame({
            "shipment_id": test["shipment_id"].to_numpy(),
            "fold": fold.fold,
            "route_id": test["route_id"].to_numpy(),
            "destination_port": test["destination_port"].to_numpy(),
            "actual_port_arrival": test["actual_port_arrival"].reset_index(drop=True),
            "scheduled_delivery": test["scheduled_delivery"].reset_index(drop=True),
            "actual_delivery": test["actual_delivery"].reset_index(drop=True),
            "arrival_delay_h": test["arrival_delay_h"].to_numpy(),
            "actual_downstream_delay_h": test["downstream_delay_h"].to_numpy(),
            "actual_final_delay_h": test["final_delivery_delay_h"].to_numpy(),
            "n_train": len(fold.train_idx),
        })
        for name in models:
            pred_y, pipe = fit_predict(name, X, y, df, fold.train_idx, fold.test_idx)
            if pipe is not None:
                pipelines[(fold.fold, name)] = pipe
            pred_f = predicted_final_delay(out["arrival_delay_h"].to_numpy(), pred_y)
            out[f"pred_downstream_{name}"] = pred_y
            out[f"pred_final_{name}"] = pred_f
            out[f"pred_eta_{name}"] = predicted_eta(out["scheduled_delivery"], pred_f)
        frames.append(out)
    return ForwardResult(pd.concat(frames, ignore_index=True), pipelines)


def summarise_forward(pred: pd.DataFrame, models: Sequence[str] = MODEL_NAMES,
                      n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Fold-level and pooled metrics, skill vs carry-forward and bootstrap CIs."""
    actual = pred["actual_final_delay_h"].to_numpy()
    ref_err = np.abs(pred[f"pred_final_{REFERENCE_MODEL}"].to_numpy() - actual)
    result: dict = {"per_fold": {}, "pooled": {}}
    for name in models:
        p = pred[f"pred_final_{name}"].to_numpy()
        err = np.abs(p - actual)
        pooled = regression_metrics(actual, p) | flag_metrics(actual, p)
        pooled["skill_vs_carry_forward"] = skill_score(pooled["mae_h"], ref_err.mean())
        pooled["mae_ci95"] = bootstrap_ci(err, pred["fold"].to_numpy(), n_boot=n_boot, seed=seed)
        if name != REFERENCE_MODEL:
            pooled["improvement_vs_carry_forward_ci95"] = bootstrap_ci(
                err, pred["fold"].to_numpy(), ref_abs_err=ref_err, n_boot=n_boot, seed=seed)
        result["pooled"][name] = pooled
        for fold, g in pred.groupby("fold"):
            a, pf = g["actual_final_delay_h"].to_numpy(), g[f"pred_final_{name}"].to_numpy()
            ref_mae = np.abs(g[f"pred_final_{REFERENCE_MODEL}"].to_numpy() - a).mean()
            m = regression_metrics(a, pf) | flag_metrics(a, pf)
            m["skill_vs_carry_forward"] = skill_score(m["mae_h"], ref_mae)
            m["n_train"] = int(g["n_train"].iloc[0])
            result["per_fold"].setdefault(int(fold), {})[name] = m
    return result


# ---------------------------------------------------------------------------
# Secondary CV and carrier ablation
# ---------------------------------------------------------------------------

def cv_evaluation(df: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
                  models: Sequence[str] = MODEL_NAMES) -> pd.DataFrame:
    """Per-split final-delay MAE for each model on repeated route-stratified CV (secondary)."""
    splits = pp.repeated_route_cv(df) if splits is None else splits
    X, y = fe.select_features(df), fe.get_target(df)
    actual = df["final_delivery_delay_h"].to_numpy()
    a = df["arrival_delay_h"].to_numpy()
    rows = []
    for s, (train_idx, test_idx) in enumerate(splits):
        for name in models:
            pred_y, _ = fit_predict(name, X, y, df, train_idx, test_idx)
            pred_f = predicted_final_delay(a[test_idx], pred_y)
            rows.append({"split": s, "repeat": s // 5, "fold": s % 5, "model": name,
                         "n_train": len(train_idx), "n_test": len(test_idx),
                         "mae_h": float(np.abs(pred_f - actual[test_idx]).mean())})
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> dict:
    """Mean/SD across splits and paired differences against carry-forward."""
    wide = cv.pivot(index="split", columns="model", values="mae_h")
    out = {}
    for name in wide.columns:
        diff = wide[name] - wide[REFERENCE_MODEL]  # negative = better than carry-forward
        out[name] = {
            "mean_mae_h": float(wide[name].mean()), "sd_mae_h": float(wide[name].std(ddof=1)),
            "repeat_mean_mae_h": [float(v) for v in cv[cv.model == name].groupby("repeat")["mae_h"].mean()],
            "paired_diff_vs_carry_forward_mean_h": float(diff.mean()),
            "paired_diff_vs_carry_forward_sd_h": float(diff.std(ddof=1)),
            "splits_better_than_carry_forward": int((diff < 0).sum()),
            "n_splits": int(len(diff)),
        }
    return out


def carrier_ablation(df: pd.DataFrame, splits: Sequence[tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """Paired per-split MAE of GBR without vs with ``carrier`` (diagnostic only)."""
    splits = pp.repeated_route_cv(df) if splits is None else splits
    X_primary = fe.select_features(df)
    X_carrier = fe.select_features(df, fe.FEATURES + list(dg.ABLATION_FEATURES), allow_ablation=True)
    y = fe.get_target(df)
    actual_y = y.to_numpy()
    rows = []
    for s, (train_idx, test_idx) in enumerate(splits):
        maes = {}
        for label, X, extra in (("primary", X_primary, ()), ("with_carrier", X_carrier, dg.ABLATION_FEATURES)):
            pipe = make_gbr_pipeline(extra_categorical=extra)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            maes[label] = float(np.abs(pipe.predict(X.iloc[test_idx]) - actual_y[test_idx]).mean())
        rows.append({"split": s, "mae_primary_h": maes["primary"], "mae_with_carrier_h": maes["with_carrier"],
                     "delta_mae_h": maes["with_carrier"] - maes["primary"]})
    return pd.DataFrame(rows)


def summarise_ablation(abl: pd.DataFrame, materiality: float = ABLATION_MATERIALITY) -> dict:
    mean_primary = float(abl["mae_primary_h"].mean())
    delta = abl["delta_mae_h"]
    rel = float(delta.mean() / mean_primary)
    return {
        "mean_mae_primary_h": mean_primary,
        "mean_mae_with_carrier_h": float(abl["mae_with_carrier_h"].mean()),
        "mean_delta_mae_h": float(delta.mean()), "sd_delta_mae_h": float(delta.std(ddof=1)),
        "min_delta_mae_h": float(delta.min()), "max_delta_mae_h": float(delta.max()),
        "relative_mean_delta": rel,
        "splits_where_carrier_improves": int((delta < 0).sum()), "n_splits": int(len(delta)),
        "materiality_threshold": materiality,
        "material_improvement_flag": bool(rel < -materiality),
        "interpretation": ("Diagnostic only. Splits overlap, so no significance test is applied. "
                           "Under a conditionally irrelevant feature a small positive delta (extra variance) is expected."),
    }


# ---------------------------------------------------------------------------
# Final artifacts and orchestration
# ---------------------------------------------------------------------------

def fit_final_models(df: pd.DataFrame) -> dict[str, Pipeline]:
    """Fit Ridge and GBR on all rows. These are deployment artifacts, NOT evaluation models."""
    X, y = fe.select_features(df), fe.get_target(df)
    return {name: PIPELINE_FACTORIES[name]().fit(X, y) for name in LEARNED_MODELS}


def save_model(pipe: Pipeline, name: str, path: Path, dataset_sha256: str) -> None:
    payload = {
        "pipeline": pipe, "model_name": name, "features": list(fe.FEATURES), "target": fe.TARGET,
        "artifact_type": "final_full_data_fit",
        "note": "Fitted on all 250 rows after evaluation; not used for any reported evaluation metric.",
        "dataset_sha256": dataset_sha256, "sklearn_version": sklearn.__version__,
        "params": GBR_PARAMS if name == "gbr" else RIDGE_PARAMS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(data_path: Path = pp.DATA_PATH, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR,
        n_boot: int = N_BOOTSTRAP) -> dict:
    """Full evaluation, ablation, final fit and artifact export."""
    df = pp.load_dataset(data_path)
    dataset_sha = _sha256(Path(data_path))

    fwd = forward_evaluation(df)
    forward_summary = summarise_forward(fwd.predictions, n_boot=n_boot)
    cv = cv_evaluation(df)
    cv_summary = summarise_cv(cv)
    ablation = carrier_ablation(df)
    ablation_summary = summarise_ablation(ablation)

    reports_dir.mkdir(parents=True, exist_ok=True)
    fwd.predictions.to_csv(reports_dir / "forward_predictions.csv", index=False)
    cv_out = cv.copy()
    cv_out["analysis"] = "secondary_repeated_route_cv"
    abl_out = ablation.assign(analysis="carrier_ablation_gbr")
    cv_out.to_csv(reports_dir / "cv_metrics.csv", index=False)
    abl_out.to_csv(reports_dir / "carrier_ablation.csv", index=False)

    rows = []
    for name in MODEL_NAMES:
        p = forward_summary["pooled"][name]
        row = {"model": name, "evaluation": "forward_pooled_2025"}
        row.update({k: v for k, v in p.items() if not isinstance(v, dict)})
        row["mae_ci95_low"], row["mae_ci95_high"] = p["mae_ci95"]["ci_low"], p["mae_ci95"]["ci_high"]
        imp = p.get("improvement_vs_carry_forward_ci95")
        row["improvement_vs_cf_h"] = imp["point"] if imp else 0.0
        row["improvement_vs_cf_ci95_low"] = imp["ci_low"] if imp else None
        row["improvement_vs_cf_ci95_high"] = imp["ci_high"] if imp else None
        for fold, metrics in forward_summary["per_fold"].items():
            row[f"fold{fold}_mae_h"] = metrics[name]["mae_h"]
        row["cv_mean_mae_h"] = cv_summary[name]["mean_mae_h"]
        row["cv_sd_mae_h"] = cv_summary[name]["sd_mae_h"]
        row["cv_paired_diff_vs_cf_mean_h"] = cv_summary[name]["paired_diff_vs_carry_forward_mean_h"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(reports_dir / "model_comparison.csv", index=False)

    final = fit_final_models(df)
    for name, pipe in final.items():
        save_model(pipe, name, models_dir / f"{name}_model.joblib", dataset_sha)

    metrics = {
        "dataset": {"path": str(Path(data_path).relative_to(ROOT)) if Path(data_path).is_relative_to(ROOT) else str(data_path),
                    "sha256": dataset_sha, "rows": len(df)},
        "target": fe.TARGET, "features": list(fe.FEATURES),
        "params": {"gbr": GBR_PARAMS, "ridge": RIDGE_PARAMS},
        "forward_folds": [{"fold": f.fold, "test_start": f.test_start.isoformat(), "test_end": f.test_end.isoformat(),
                           "n_train": len(f.train_idx), "n_test": len(f.test_idx), "n_embargoed": f.n_embargoed}
                          for f in pp.forward_folds(df)],
        "forward": forward_summary,
        "secondary_cv": {"description": "Repeated route-stratified 5x5 CV; overlapping splits, not independent replications.",
                         "summary": cv_summary},
        "carrier_ablation": ablation_summary,
        "artifacts": {"final_models": {n: str((models_dir / f"{n}_model.joblib").relative_to(ROOT))
                                       if (models_dir / f"{n}_model.joblib").is_relative_to(ROOT)
                                       else str(models_dir / f"{n}_model.joblib") for n in LEARNED_MODELS},
                      "note": "Final models are refitted on all rows after evaluation and are not used for any metric."},
        "notes": {
            "confidence_intervals": ("95% percentile bootstrap, resampling shipments within each forward fold "
                                     f"(seed {BOOTSTRAP_SEED}). Quantifies test-sample uncertainty conditional on the "
                                     "three evaluated 2025 windows and the fitted fold models; not model-refit uncertainty."),
            "ranking_metrics": "ROC-AUC / PR-AUC use the predicted final delay as a ranking score, not a probability.",
            "synthetic": "Synthetic data under documented assumptions; results do not establish real-world performance.",
        },
    }
    (reports_dir / "training_metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the shipment ETA models.")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args(argv)
    m = run(n_boot=args.n_boot)
    for name, p in m["forward"]["pooled"].items():
        print(f"{name:14s} forward MAE {p['mae_h']:7.2f} h  [{p['mae_ci95']['ci_low']:.2f}, {p['mae_ci95']['ci_high']:.2f}]"
              f"  skill vs CF {p['skill_vs_carry_forward']:+.3f}")


if __name__ == "__main__":
    main()
