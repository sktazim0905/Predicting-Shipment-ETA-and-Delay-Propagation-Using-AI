"""Historical stage-delay analysis of completed shipments.

These functions read realised outcomes (D3, D4, D5, waits) of *completed*
shipments for descriptive analysis only. They never modify their input and
are never used to build model inputs. Columns prefixed ``_latent_`` are
simulator audit variables and are labelled as such where used.

Relationships shown here are associations in synthetic data generated under
documented assumptions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STAGES = {"port_processing_delay_h": "D3 port", "customs_delay_h": "D4 customs",
          "inland_delivery_delay_h": "D5 inland"}
CONGESTION_BINS = [0.0, 0.3, 0.45, 0.6, 1.0001]
CONGESTION_LABELS = ["<0.30", "0.30-0.45", "0.45-0.60", ">=0.60"]
ARRIVAL_BINS = [-np.inf, 0.0, 12.0, 24.0, np.inf]
ARRIVAL_LABELS = ["early (<0h)", "0-12h late", "12-24h late", ">24h late"]
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def _summary(grouped) -> pd.DataFrame:
    return grouped.agg(["count", "mean", "median"]).round(2)


def stage_delays_by_destination(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and median D3, D4, D5 and downstream delay per destination port."""
    cols = list(STAGES) + ["downstream_delay_h"]
    out = df.groupby("destination_port")[cols].agg(["mean", "median"]).round(2)
    out.insert(0, ("n", ""), df.groupby("destination_port").size())
    return out


def port_delay_by_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """D3 by observed congestion-at-arrival band (rows with missing congestion reported separately)."""
    band = pd.cut(df["dest_congestion_index"], CONGESTION_BINS, labels=CONGESTION_LABELS, right=False)
    band = band.astype(object).where(band.notna(), "missing")
    out = _summary(df.groupby(band, sort=False)["port_processing_delay_h"])
    return out.reindex([b for b in CONGESTION_LABELS + ["missing"] if b in out.index])


def port_delay_by_arrival_status(df: pd.DataFrame) -> pd.DataFrame:
    """D3 by arrival-delay band (how late the vessel reached port)."""
    band = pd.cut(df["arrival_delay_h"], ARRIVAL_BINS, labels=ARRIVAL_LABELS)
    return _summary(df.groupby(band, observed=False)["port_processing_delay_h"])


def customs_delay_by_documentation(df: pd.DataFrame) -> pd.DataFrame:
    """D4 by documentation completeness, with the realised inspection rate."""
    g = df.groupby("documentation_complete")
    out = _summary(g["customs_delay_h"])
    out["inspection_rate"] = g["customs_inspection"].mean().round(3)
    return out


def truck_wait_by_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Truck-window waiting (simulator audit column) by congestion during dwell.

    Uses ``_latent_truck_wait_h`` and ``_latent_congestion_dwell``, which exist
    only because the data are simulated. Rows are shipments that missed their
    booked window.
    """
    missed = df[df["booked_window_missed"]]
    band = pd.cut(missed["_latent_congestion_dwell"], CONGESTION_BINS, labels=CONGESTION_LABELS, right=False)
    g = missed.groupby(band, observed=False)
    out = _summary(g["_latent_truck_wait_h"])
    out["zero_wait_share"] = g["_latent_truck_wait_h"].apply(lambda s: (s == 0).mean() if len(s) else np.nan).round(3)
    return out


def upstream_vs_downstream(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mean stage delays by arrival-delay band, and Spearman correlations of A with each stage."""
    band = pd.cut(df["arrival_delay_h"], ARRIVAL_BINS, labels=ARRIVAL_LABELS)
    cols = list(STAGES) + ["downstream_delay_h"]
    means = df.groupby(band, observed=False)[cols].mean().round(2)
    means.insert(0, "n", df.groupby(band, observed=False).size())
    corr = df[cols].corrwith(df["arrival_delay_h"], method="spearman").round(3).rename("spearman_vs_arrival_delay")
    return means, corr


def run_all(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """All tables in one dictionary."""
    means, corr = upstream_vs_downstream(df)
    return {
        "stage_delays_by_destination": stage_delays_by_destination(df),
        "port_delay_by_congestion": port_delay_by_congestion(df),
        "port_delay_by_arrival_status": port_delay_by_arrival_status(df),
        "customs_delay_by_documentation": customs_delay_by_documentation(df),
        "truck_wait_by_congestion": truck_wait_by_congestion(df),
        "upstream_vs_downstream_means": means,
        "upstream_vs_downstream_spearman": corr,
    }


def save_figures(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Write the descriptive stage-analysis figures; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def _save(fig, name):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    by_dest = df.groupby("destination_port")[list(STAGES)].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    by_dest.rename(columns=STAGES).plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean stage delay (h)")
    ax.set_title("Mean downstream stage delay by destination (synthetic)")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "stage_delays_by_destination.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    obs = df.dropna(subset=["dest_congestion_index"])
    late = obs["arrival_delay_h"] > 12
    ax.scatter(obs.loc[~late, "dest_congestion_index"], obs.loc[~late, "port_processing_delay_h"], s=14, label="arrival delay <= 12 h")
    ax.scatter(obs.loc[late, "dest_congestion_index"], obs.loc[late, "port_processing_delay_h"], s=14, label="arrival delay > 12 h")
    ax.set_xlabel("observed congestion at arrival")
    ax.set_ylabel("D3 port delay (h)")
    ax.set_title("Port delay vs congestion (synthetic)")
    ax.legend()
    _save(fig, "port_delay_vs_congestion.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    band = pd.cut(df["arrival_delay_h"], ARRIVAL_BINS, labels=ARRIVAL_LABELS)
    df.groupby(band, observed=False)[list(STAGES)].mean().rename(columns=STAGES).plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("arrival delay band")
    ax.set_ylabel("mean stage delay (h)")
    ax.set_title("Downstream stage delays by upstream lateness (synthetic)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "stage_delays_by_arrival_band.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    groups = [df.loc[df["documentation_complete"] == v, "customs_delay_h"] for v in (True, False)]
    ax.boxplot(groups, tick_labels=["complete", "incomplete"])
    ax.set_ylabel("D4 customs delay (h)")
    ax.set_title("Customs delay by documentation (synthetic)")
    _save(fig, "customs_delay_by_documentation.png")
    return paths


def save_tables(df: pd.DataFrame, out_path: Path) -> Path:
    """Write all stage-analysis tables to one Markdown file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Historical stage analysis (completed shipments, synthetic data)\n",
             "Descriptive associations only; never used as model inputs. "
             "Columns prefixed `_latent_` are simulator audit variables.\n"]
    for name, table in run_all(df).items():
        parts.append(f"\n## {name}\n\n```\n{table.to_string()}\n```\n")
    text = "\n".join(line.rstrip() for line in "".join(parts).splitlines()) + "\n"
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path


if __name__ == "__main__":
    from src import preprocessing as pp

    data = pp.load_dataset()
    figures = save_figures(data)
    tables = save_tables(data, FIGURES_DIR.parent / "stage_analysis.md")
    print("wrote", [str(p.name) for p in figures], "and", tables.name)
