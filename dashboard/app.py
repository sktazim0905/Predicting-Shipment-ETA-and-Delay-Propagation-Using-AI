"""Streamlit dashboard for the synthetic shipment ETA project.

Run from the repository root:  python -m streamlit run dashboard/app.py

UI only: every number comes from frozen artifacts or the project's modules
(``src.dashboard_data``, ``src.predict``, ``src.stage_analysis``). The app never
fits a model, regenerates data or writes files.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import dashboard_data as dd  # noqa: E402
from src import data_generator as dg  # noqa: E402
from src import explainability as ex  # noqa: E402
from src import predict as pr  # noqa: E402
from src import stage_analysis as sa  # noqa: E402

st.set_page_config(page_title="Shipment ETA - Aiqon", page_icon=":ship:", layout="wide")

# Categorical slots 1-4 (fixed order) and the diverging pair, from the validated reference palette.
MODEL_COLORS = {"Schedule-only": "#2a78d6", "Carry-forward": "#eb6834", "Ridge": "#1baf7a", "Gradient Boosting": "#eda100"}
STAGE_COLORS = {"Port (D3)": "#2a78d6", "Customs (D4)": "#eb6834", "Inland (D5)": "#1baf7a"}
SINGLE = "#2a78d6"
RAISES, LOWERS = "#e34948", "#2a78d6"
SCENARIO_LABEL = "Scenario forecast using the full-data model. This is not an out-of-sample evaluation result."
SYNTHETIC = ("All data are synthetic, generated under documented assumptions. Results describe the simulation, "
             "not real shipping operations.")
SHAP_NOTE = ("SHAP values show how the model's prediction depends on its inputs (model attribution). They are not "
             "causal effects, and correlated inputs can share attribution.")
PAGES = ["Overview", "Predict ETA", "Explain prediction", "Stage analysis"]


# ---------------------------------------------------------------------------
# Cached, read-only data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_all():
    df = dd.load_dataset()
    metrics = dd.load_training_metrics()
    return {"df": df, "metrics": metrics, "comparison": dd.load_model_comparison(),
            "folds": dd.fold_metrics(metrics), "forward": dd.load_forward_predictions(),
            "shap": dd.load_forward_shap_values(), "stage": sa.run_all(df)}


@st.cache_resource(show_spinner=False)
def predictor(model: str) -> pr.ShipmentPredictor:
    return pr.ShipmentPredictor(model)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def bar_base(data: pd.DataFrame) -> alt.Chart:
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=2)


#: Category labels carry "\n(n=..)": render them on two lines.
TWO_LINE_LABELS = alt.Axis(labelAngle=0, labelExpr="split(datum.value, '\\n')", labelLimit=200)


def grouped_bars(data, x, x_title, y, y_title, series, colors, x_sort=None, height=280, extra_tooltips=()):
    order = list(colors)
    return (bar_base(data).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=x_sort, axis=TWO_LINE_LABELS),
        xOffset=alt.XOffset(f"{series}:N", sort=order),
        y=alt.Y(f"{y}:Q", title=y_title),
        color=alt.Color(f"{series}:N", sort=order, scale=alt.Scale(domain=order, range=list(colors.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip(f"{x}:N", title=x_title), alt.Tooltip(f"{series}:N", title=series),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=".1f"), *extra_tooltips],
    ).properties(height=height))


def single_bars(data, x, x_title, y, y_title, extra_tooltips=(), x_sort=None, height=240):
    return (bar_base(data).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=x_sort, axis=TWO_LINE_LABELS),
        y=alt.Y(f"{y}:Q", title=y_title),
        color=alt.value(SINGLE),
        tooltip=[alt.Tooltip(f"{x}:N", title=x_title), alt.Tooltip(f"{y}:Q", title=y_title, format=".1f"),
                 *extra_tooltips],
    ).properties(height=height))


NET_AXIS_TITLE = "Net SHAP contribution per driver group (h)"
NET_RAISES, NET_LOWERS = "Raises delay", "Lowers delay"


def contribution_chart(groups: pd.DataFrame, height=300) -> alt.Chart:
    """One bar per driver group: its net SHAP contribution to Y_hat (hours).

    ``groups`` comes from :func:`dashboard_data.group_breakdown`; the positive and
    negative feature parts of each net value are shown in the tooltip.
    """
    data = groups.assign(direction=lambda d: d["contribution_h"].map(
        lambda v: NET_RAISES if v >= 0 else NET_LOWERS))
    order = data.sort_values("contribution_h", key=abs, ascending=False)["label"].tolist()
    lim = max(1.0, float(data["contribution_h"].abs().max())) * 1.25  # symmetric axis with room for end labels
    bars = bar_base(data).encode(
        y=alt.Y("label:N", title=None, sort=order, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
        x=alt.X("contribution_h:Q", title=NET_AXIS_TITLE, scale=alt.Scale(domain=[-lim, lim])),
        color=alt.Color("direction:N", scale=alt.Scale(domain=[NET_RAISES, NET_LOWERS], range=[RAISES, LOWERS]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("label:N", title="Driver group"),
                 alt.Tooltip("contribution_h:Q", title="Net contribution (h)", format="+.2f"),
                 alt.Tooltip("positive_h:Q", title="Sum of positive feature contributions (h)", format="+.2f"),
                 alt.Tooltip("negative_h:Q", title="Sum of negative feature contributions (h)", format="+.2f"),
                 alt.Tooltip("n_features:Q", title="Features in group")],
    )
    text = dict(y=alt.Y("label:N", sort=order), x="contribution_h:Q",
                text=alt.Text("contribution_h:Q", format="+.1f"))
    pos = alt.Chart(data).transform_filter("datum.contribution_h >= 0").mark_text(
        dx=4, align="left", color="#52514e").encode(**text)
    neg = alt.Chart(data).transform_filter("datum.contribution_h < 0").mark_text(
        dx=-4, align="right", color="#52514e").encode(**text)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8984").encode(x="x:Q")
    return (bars + zero + pos + neg).properties(height=height)


def hours_column(label: str, fmt: str = "%.1f") -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format=fmt)


CHART_NOTE = ("Each bar is a driver group's net contribution: the sum of its features' SHAP values. Features "
              "inside a group can push in opposite directions; the tooltip shows the positive and negative parts.")


def shap_section(groups: pd.DataFrame, features: pd.DataFrame, reconciliation: str) -> None:
    """Group chart, then the feature table below it at full content width."""
    st.markdown("**Main factors: net SHAP contribution of each driver group**")
    st.altair_chart(contribution_chart(groups), width="stretch")
    st.caption(f"{reconciliation} {CHART_NOTE} {SHAP_NOTE}")
    st.markdown("**Largest individual feature contributions**")
    table = features.head(8).assign(Group=lambda d: d["group"].map(dd.GROUP_LABELS))
    st.dataframe(table[["feature", "Group", "contribution_h"]].rename(columns={"feature": "Feature"}),
                 hide_index=True, width="stretch", column_config={
                     "Feature": st.column_config.TextColumn("Feature", width="large"),
                     "Group": st.column_config.TextColumn("Driver group", width="medium"),
                     "contribution_h": st.column_config.NumberColumn("SHAP contribution (h)", format="%+.2f",
                                                                     width="medium")})


def status_line(final_h: float, prefix: str = "") -> None:
    delayed, text = dd.delay_status(final_h)
    icon = ":material/warning:" if delayed else ":material/check_circle:"
    st.markdown(f"{icon} **{prefix}{text}** (reporting threshold: final delay > {dd.DELAY_THRESHOLD_H:.0f} h)")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(data) -> None:
    summary = dd.dataset_summary(data["df"], data["metrics"])
    st.title("Shipment ETA and delay propagation")
    st.caption(SYNTHETIC)
    c = st.columns(4)
    c[0].metric("Synthetic shipments", summary["n_shipments"])
    c[1].metric("Routes", summary["n_routes"])
    c[2].metric("Forward validation folds", summary["n_folds"])
    c[3].metric("Held-out shipments evaluated", summary["n_held_out"])

    st.subheader("What the system predicts")
    st.markdown(
        "- **Checkpoint:** the vessel's actual arrival at the destination port.\n"
        "- **Observed arrival delay A** (departure delay D1 + ocean transit delay D2) is a measured fact at that point.\n"
        "- **Target:** the downstream delay Y still to come (port, customs and inland stages versus plan). "
        "The model predicts this as the **predicted downstream delay**.\n"
        "- **Predicted final delay = A + predicted downstream delay**; **predicted ETA = scheduled delivery + predicted final delay**."
    )
    folds = pd.DataFrame(summary["folds"]).rename(columns={
        "fold": "Fold", "test_start": "Test window start (UTC)", "test_end": "Test window end (UTC)",
        "n_train": "Training shipments", "n_test": "Held-out shipments",
        "n_embargoed": "Excluded by delivery embargo"})
    folds["Test window start (UTC)"] = folds["Test window start (UTC)"].str[:10]
    folds["Test window end (UTC)"] = folds["Test window end (UTC)"].str[:10]
    with st.expander("Forward validation folds (train on earlier, delivered shipments; test on the next window)"):
        st.dataframe(folds, hide_index=True, width="stretch")

    st.subheader("Model comparison on held-out shipments")
    st.caption("Final-delay error in hours, pooled over the three forward folds (out-of-sample).")
    st.dataframe(dd.comparison_table(data["comparison"]), hide_index=True, width="stretch", column_config={
        "MAE (h)": hours_column("MAE (h)", "%.2f"), "RMSE (h)": hours_column("RMSE (h)"),
        "P90 abs. error (h)": hours_column("P90 abs. error (h)"),
        "Within ±24 h (%)": hours_column("Within ±24 h (%)"),
        "Skill vs carry-forward": hours_column("Skill vs carry-forward", "%+.3f")})

    st.subheader("Performance by validation fold")
    f = data["folds"].assign(Fold=lambda d: "Fold " + d["fold"].astype(str) + " (" + d["window"] + ")")
    st.altair_chart(grouped_bars(f, "Fold", "Validation fold", "mae_h", "MAE (h)", "Model", MODEL_COLORS),
                    width="stretch")
    with st.expander("Per-fold metrics table"):
        st.dataframe(f[["Fold", "Model", "n_train", "mae_h", "rmse_h", "p90_ae_h", "within_24h_pct",
                        "skill_vs_carry_forward"]].rename(columns={"n_train": "Training shipments"}),
                     hide_index=True, width="stretch", column_config={
                         "mae_h": hours_column("MAE (h)"), "rmse_h": hours_column("RMSE (h)"),
                         "p90_ae_h": hours_column("P90 abs. error (h)"),
                         "within_24h_pct": hours_column("Within ±24 h (%)"),
                         "skill_vs_carry_forward": hours_column("Skill vs carry-forward", "%+.3f")})

    st.subheader("Key findings and limitations")
    st.markdown("\n".join(f"- {line}" for line in dd.key_findings(data["comparison"], data["folds"])))
    st.markdown(f"- {SYNTHETIC}")


def _local_input(label: str, default: datetime, key: str) -> datetime:
    d, t = st.columns(2)
    day = d.date_input(f"{label} date", value=default.date(), key=f"{key}_date")
    clock = t.time_input(f"{label} time", value=default.time(), step=900, key=f"{key}_time")
    return datetime.combine(day, clock)


def page_predict() -> None:
    st.title("Predict ETA for a new scenario")
    st.info(SCENARIO_LABEL, icon=":material/info:")
    destinations = list(dg.DEFAULT_CONFIG["destinations"])
    cargo_types = list(dg.DEFAULT_CONFIG["cargo"]["mix"])
    default_arrival, default_delivery = dd.default_scenario_times(date.today())

    with st.form("scenario"):
        left, mid, right = st.columns(3)
        with left:
            st.markdown("**Shipment**")
            model_label = st.radio("Model", ["Gradient Boosting", "Ridge"], horizontal=True)
            port = st.selectbox("Destination port", destinations, index=destinations.index("Rotterdam"))
            cargo = st.selectbox("Cargo type", cargo_types, index=0)
            docs = st.checkbox("Customs documentation complete", value=True)
        with mid:
            st.markdown("**Arrival at the destination port** (destination-local time)")
            arrival_local = _local_input("Actual port arrival", default_arrival, "arr")
            d1 = st.number_input("Departure delay D1 (h)", value=4.0, step=0.5, format="%.1f")
            d2 = st.number_input("Ocean transit delay D2 (h)", value=10.5, step=0.5, format="%.1f")
            st.caption("Observed arrival delay A = D1 + D2. Negative values mean early. "
                       "If only the total is known, enter it as D1 and set D2 to 0.")
        with right:
            st.markdown("**Port conditions and plan**")
            unknown_cong = st.checkbox("Congestion reading unavailable", value=False)
            congestion = st.slider("Destination congestion index", 0.0, 1.0, 0.45, 0.01)
            weather = st.slider("Destination weather severity", 0.0, 1.0, 0.20, 0.01)
            dwell = st.number_input("Planned port dwell (h)", min_value=1.0, value=60.0, step=1.0)
            km = st.number_input("Inland distance (km)", min_value=0.0, value=180.0, step=10.0)
            inland_h = st.number_input("Planned inland duration (h)", min_value=0.5, value=12.5, step=0.5)
            delivery_local = _local_input("Scheduled delivery", default_delivery, "del")
        st.caption(dd.SCENARIO_DATE_NOTE)
        submitted = st.form_submit_button("Predict ETA", type="primary")

    if submitted:
        model = "gbr" if model_label == "Gradient Boosting" else "ridge"
        try:
            result = pr.predict_scenario(
                model, predictor=predictor(model), destination_port=port, cargo_type=cargo,
                documentation_complete=docs, dest_congestion_index=None if unknown_cong else congestion,
                dest_weather_severity=weather, planned_port_dwell_h=dwell, inland_distance_km=km,
                planned_inland_h=inland_h, arrival_delay_h=d1 + d2,
                actual_port_arrival_local=arrival_local, scheduled_delivery_local=delivery_local)
            st.session_state["scenario_result"] = {"result": result, "d1": d1, "d2": d2, "model_label": model_label}
        except pr.InvalidShipmentError as exc:
            st.session_state.pop("scenario_result", None)
            st.error(f"Input not accepted: {exc}")

    scenario = st.session_state.get("scenario_result")
    if not scenario:
        return
    r: pr.PredictionResult = scenario["result"]
    st.divider()
    st.subheader(f"Scenario forecast ({scenario['model_label']}, {r.destination_port})")
    st.caption(SCENARIO_LABEL)
    c = st.columns(4)
    c[0].metric("Observed arrival delay A", dd.fmt_hours(r.arrival_delay_h),
                help=f"D1 {scenario['d1']:+.1f} h + D2 {scenario['d2']:+.1f} h (measured, not predicted)")
    c[1].metric("Predicted downstream delay", dd.fmt_hours(r.predicted_downstream_delay_h))
    c[2].metric("Predicted final delay (A + predicted downstream)", dd.fmt_hours(r.predicted_final_delay_h))
    c[3].metric("Predicted ETA (local)", dd.fmt_local(r.predicted_eta_utc, r.destination_timezone).split(" (")[0],
                help=f"{r.destination_timezone}; scheduled delivery "
                     f"{dd.fmt_local(r.scheduled_delivery_utc, r.destination_timezone)}")
    st.caption(f"Predicted ETA {dd.fmt_local(r.predicted_eta_utc, r.destination_timezone)} in "
               f"{r.destination_timezone}; scheduled delivery "
               f"{dd.fmt_local(r.scheduled_delivery_utc, r.destination_timezone)}.")
    status_line(r.predicted_final_delay_h, "Predicted: ")

    feats = dd.feature_frame(r.feature_contributions_h, predictor(r.model_name).explainer.feature_groups)
    total = sum(r.feature_contributions_h.values())
    shap_section(dd.group_breakdown(feats), feats,
                 f"SHAP baseline {r.shap_base_value_h:+.1f} h + contributions {total:+.1f} h = "
                 f"predicted downstream delay (Y_hat) {r.predicted_downstream_delay_h:+.1f} h.")
    with st.expander("Model inputs used (including derived values)"):
        inputs = pd.DataFrame({"Input": list(r.model_inputs), "Value": [
            "missing (imputed by the pipeline, with a missing indicator)" if v is None else v
            for v in r.model_inputs.values()]})
        inputs["Value"] = inputs["Value"].astype(str)
        st.dataframe(inputs, hide_index=True, width="stretch")
        st.caption("arrival_delay_h and projected_nonworking_wait_h are derived from the timestamps: the projected "
                   "wait applies the customs calendar to arrival time + planned dwell. Actual future stage "
                   "durations are unknown at prediction time and are not shown.")
    with st.expander("Plain-language explanation"):
        st.write(r.explanation)


def page_explain(data) -> None:
    st.title("Explain a held-out prediction")
    fwd = data["forward"]
    ids = sorted(fwd["shipment_id"])
    c1, c2 = st.columns([2, 1])
    sid = c1.selectbox("Held-out shipment (2025 forward test windows)", ids, index=ids.index("SHP-0209"))
    model_label = c2.radio("Model", ["Gradient Boosting", "Ridge"], horizontal=True, key="explain_model")
    model = "gbr" if model_label == "Gradient Boosting" else "ridge"
    v = dd.held_out_view(sid, model, fwd, data["shap"])
    fold_info = next(f for f in data["metrics"]["forward_folds"] if f["fold"] == v.fold)
    st.success(f"Out-of-sample: explained by the fold-{v.fold} evaluation model, trained on "
               f"{fold_info['n_train']} shipments delivered before {fold_info['test_start'][:10]}. "
               f"{v.shipment_id} was not in its training data.", icon=":material/verified:")

    st.markdown(f"**{v.shipment_id}** · route {v.route_id} · destination {v.destination_port} ({v.timezone}) · "
                f"fold {v.fold}")
    c = st.columns(4)
    c[0].metric("Actual arrival delay A", dd.fmt_hours(v.arrival_delay_h))
    c[1].metric("Actual downstream delay Y", dd.fmt_hours(v.actual_downstream_h))
    c[2].metric("Predicted downstream delay", dd.fmt_hours(v.predicted_downstream_h))
    c[3].metric("Prediction error (predicted - actual)", dd.fmt_hours(v.prediction_error_h))
    c = st.columns(4)
    c[0].metric("Actual final delay", dd.fmt_hours(v.actual_final_h))
    c[1].metric("Predicted final delay (A + predicted downstream)", dd.fmt_hours(v.predicted_final_h))
    c[2].metric("Actual delivery (local)", v.actual_eta_local.split(" (")[0])
    c[3].metric("Predicted ETA (local)", v.predicted_eta_local.split(" (")[0])
    st.caption(f"Local times in {v.timezone}: scheduled {v.scheduled_delivery_local}, actual {v.actual_eta_local}, "
               f"predicted {v.predicted_eta_local}.")
    status_line(v.actual_final_h, "Actual: ")
    status_line(v.predicted_final_h, "Predicted: ")

    shap_section(dd.group_breakdown(v.features), v.features,
                 f"SHAP baseline {v.shap_base_value_h:+.1f} h + contributions "
                 f"{v.groups['contribution_h'].sum():+.1f} h = predicted downstream delay (Y_hat) {v.predicted_downstream_h:+.1f} h "
                 f"(additivity error {v.additivity_error_h:.1e} h). The observed arrival delay A is added once, "
                 f"outside SHAP.")


def page_stage(data) -> None:
    st.title("Historical stage analysis")
    st.warning("These are patterns produced by the synthetic simulation (completed shipments, 250 records). "
               "They are not empirical findings about real shipping operations, and they are never model inputs.",
               icon=":material/science:")
    t = data["stage"]
    frames = dd.stage_chart_frames(t)
    shipments = alt.Tooltip("n:Q", title="Shipments")
    st.caption("Bar labels show the number of shipments (n) in each group.")

    left, right = st.columns(2)
    with left:
        dest = frames["destination"]
        st.markdown("**Mean stage delay by destination**")
        st.altair_chart(grouped_bars(dest, "label", "Destination port", "mean_h", "Mean delay vs plan (h)",
                                     "Stage", STAGE_COLORS, height=270, extra_tooltips=[shipments]),
                        width="stretch")
        arr = frames["arrival"]
        st.markdown("**Port delay by how late the vessel arrived**")
        st.altair_chart(single_bars(arr, "label", "Arrival delay A", "mean", "Mean port delay D3 (h)",
                                    [shipments], x_sort=list(arr["label"])), width="stretch")
    with right:
        cong = frames["congestion"]
        st.markdown("**Port delay by congestion at arrival**")
        st.altair_chart(single_bars(cong, "label", "Observed congestion index", "mean", "Mean port delay D3 (h)",
                                    [shipments], x_sort=list(cong["label"]), height=270), width="stretch")
        docs = frames["documentation"]
        st.markdown("**Customs delay by documentation**")
        st.altair_chart(single_bars(docs, "label", "Customs documentation", "mean", "Mean customs delay D4 (h)",
                                    [shipments, alt.Tooltip("inspection_rate:Q", title="Inspection rate", format=".1%")],
                                    x_sort=list(docs["label"])), width="stretch")

    note = dd.truck_wait_note(data["df"])
    st.markdown(f"**Truck waiting after missing the booked {note['window_h']:.0f}-hour pickup window**, "
                f"by congestion during port dwell ({note['n_missed']} shipments)")
    st.caption(
        f"Covers the {note['n_missed']} shipments whose cargo was not ready within its booked "
        f"{note['window_h']:.0f}-hour truck window. Missing the booked window does not always mean waiting: if "
        f"the {note['window_h']:.0f}-hour window in which the cargo became ready still had a truck available, it "
        f"was collected immediately. 'Share with zero wait' counts those cases ({note['n_missed_zero_wait']} of "
        f"{note['n_missed']}). The other {note['n_made']} shipments made their booked window and are not in this "
        f"table; {note['n_made_with_wait']} of them waited for a truck.")
    truck = t["truck_wait_by_congestion"].reset_index().rename(columns={
        "_latent_congestion_dwell": "Congestion during dwell", "count": "Shipments", "mean": "Mean wait (h)",
        "median": "Median wait (h)", "zero_wait_share": "Share with zero wait"})
    truck["Congestion during dwell"] = truck["Congestion during dwell"].astype(str)
    st.dataframe(truck, hide_index=True, width="stretch", column_config={
        "Mean wait (h)": hours_column("Mean wait (h)"), "Median wait (h)": hours_column("Median wait (h)"),
        "Share with zero wait": st.column_config.NumberColumn("Share with zero wait", format="percent")})
    st.caption("Truck-window waiting and congestion during dwell are simulator audit variables; they would not be "
               "observable in this form in real data.")
    with st.expander("Upstream lateness vs downstream stage delays"):
        means = t["upstream_vs_downstream_means"].reset_index().rename(columns={"arrival_delay_h": "Arrival delay band"})
        means["Arrival delay band"] = means["Arrival delay band"].astype(str)
        st.dataframe(means, hide_index=True, width="stretch")
        st.dataframe(t["upstream_vs_downstream_spearman"].rename("Spearman correlation with A").to_frame(),
                     width="stretch")


def main() -> None:
    with st.sidebar:
        st.markdown("### Shipment ETA")
        page = st.radio("Section", PAGES, label_visibility="collapsed")
    with st.spinner("Loading frozen artifacts..."):
        data = load_all()
    if page == "Overview":
        page_overview(data)
    elif page == "Predict ETA":
        page_predict()
    elif page == "Explain prediction":
        page_explain(data)
    else:
        page_stage(data)


main()
