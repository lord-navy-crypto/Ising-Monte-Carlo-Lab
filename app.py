from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ising_lab import (
    IsingParams,
    Method,
    equilibration_diagnostics,
    estimate_equilibration_sweep,
    exact_observables,
    finite_size_scan,
    magnetization_estimators,
    method_compatibility,
    method_comparison,
    multi_chain_convergence,
    notebook_temperature_grid,
    onsager_tc,
    simulate,
    thermodynamic_scan,
)


APP_VERSION = "1.1.3"
METHOD_LABELS = {
    "Metropolis (random, notebook)": Method.METROPOLIS,
    "Metropolis (checkerboard)": Method.CHECKERBOARD,
    "Heat-bath": Method.HEAT_BATH,
    "Wolff cluster": Method.WOLFF,
}

st.set_page_config(
    page_title="Ising Monte Carlo Lab",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp{background:linear-gradient(145deg,#f7faff 0%,#fff 45%,#f7f4ff 100%)}
    .hero{padding:1.45rem 1.7rem;border-radius:20px;color:white;
      background:linear-gradient(120deg,#172a46 0%,#365fa0 50%,#6b3fa0 100%);
      box-shadow:0 14px 34px rgba(23,42,70,.2);margin-bottom:1rem}
    .hero h1{margin:0 0 .35rem;font-size:2.1rem}.hero p{margin:0;opacity:.93}
    .note{padding:.85rem 1rem;border-left:4px solid #365fa0;border-radius:8px;
      background:#edf4ff;margin:.5rem 0 1rem}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce5ef;
      padding:.7rem;border-radius:14px;box-shadow:0 4px 14px rgba(23,42,70,.05)}
    </style>
    """,
    unsafe_allow_html=True,
)


def defaults() -> dict[str, object]:
    return {
        "size": 16,
        "coupling": 1.0,
        "field": 0.0,
        "temperature": 2.5,
        "dimension": 2,
        "seed": 2026,
        "eq_sweeps": 400,
        "mc_sweeps": 800,
        "measure_every": 1,
        "record_every": 10,
        "comparison_eq": 250,
        "comparison_mc": 400,
        "thermo_method": "Wolff cluster",
        "T_min": 0.8,
        "T_max": 3.5,
        "T_points": 17,
        "use_notebook_grid": False,
        "eq_diagnostic_sweeps": 1200,
        "finite_sizes": "8,12,16,24",
        "snapshot_sweeps": 200,
        "snapshot_initial": "random",
    }


RESULT_KEYS = (
    "comparison_result",
    "equilibration_result",
    "multi_chain_result",
    "thermo_1d_result",
    "thermo_2d_result",
    "finite_size_result",
    "snapshot_result",
    "external_result",
    "validation_result",
)


def initialize() -> None:
    for key, value in defaults().items():
        st.session_state.setdefault(key, value)
    for key in RESULT_KEYS:
        st.session_state.setdefault(key, None)


def params(**overrides: float | int) -> IsingParams:
    values = {
        "size": int(st.session_state.size),
        "coupling": float(st.session_state.coupling),
        "field": float(st.session_state.field),
        "temperature": float(st.session_state.temperature),
        "dimension": int(st.session_state.dimension),
    }
    values.update(overrides)
    return IsingParams(**values)


def config() -> dict[str, object]:
    return {
        "schema": "ising-monte-carlo-lab-v2",
        "app_version": APP_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **{key: st.session_state[key] for key in defaults()},
    }


def load_config(uploaded) -> None:
    if uploaded is None:
        return
    signature = (uploaded.name, uploaded.size)
    if st.session_state.get("loaded_signature") == signature:
        return
    try:
        loaded = json.loads(uploaded.getvalue().decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("configuration root must be a JSON object")
        candidate = defaults()
        for key, fallback in candidate.items():
            if key in loaded:
                candidate[key] = type(fallback)(loaded[key])
        if candidate["size"] < 4 or candidate["coupling"] <= 0 or candidate["temperature"] <= 0:
            raise ValueError("size, coupling, and temperature must be positive with size >= 4")
        if candidate["thermo_method"] not in METHOD_LABELS:
            raise ValueError("unknown Monte Carlo method")
        if int(candidate["dimension"]) not in (1, 2):
            raise ValueError("dimension must be 1 or 2")
        for key, value in candidate.items():
            st.session_state[key] = value
        for key in RESULT_KEYS:
            st.session_state[key] = None
        st.session_state.loaded_signature = signature
        st.success("Configuration loaded. Run an experiment to refresh results.")
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        st.error(f"Could not load configuration: {exc}")


def progress(label: str):
    bar = st.progress(0, text=label)

    def update(done: int, total: int) -> None:
        bar.progress(done / total, text=f"{label}: {done}/{total}")

    return bar, update


def download_frame(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


DISPLAY_LABELS = {
    "method": "Monte Carlo method",
    "temperature": "Temperature",
    "n_sites": "Number of lattice sites",
    "energy_per_site": "Mean energy per site",
    "energy_per_site_error": "Correlated standard error of energy/site",
    "energy_per_site_block_error": "Fixed-block error of energy/site",
    "mean_abs_magnetization": "Mean absolute magnetization <|M|>/N",
    "abs_mean_magnetization": "Notebook estimator |<M>|/N",
    "signed_magnetization": "Signed magnetization <M>/N",
    "magnetization_error": "Correlated standard error of <|M|>/N",
    "magnetization_block_error": "Fixed-block error of <|M|>/N",
    "specific_heat": "Specific heat per site",
    "susceptibility": "Susceptibility (standard alias)",
    "susceptibility_standard": "Thermodynamic susceptibility Var(M)/(N T)",
    "susceptibility_abs_connected": "|M|-connected susceptibility diagnostic",
    "susceptibility_signed": "Signed susceptibility alias",
    "tau_int_energy_samples": "Integrated autocorrelation time of energy (samples)",
    "tau_int_energy_cycles": "Integrated autocorrelation time of energy (update cycles)",
    "tau_int_energy_sweeps": "Integrated autocorrelation time of energy (legacy sweep alias)",
    "effective_samples_energy": "Effective energy samples",
    "tau_int_abs_magnetization_samples": "Integrated autocorrelation time of |M| (samples)",
    "tau_int_abs_magnetization_cycles": "Integrated autocorrelation time of |M| (update cycles)",
    "tau_int_abs_magnetization_sweeps": "Integrated autocorrelation time of |M| (legacy sweep alias)",
    "effective_samples_abs_magnetization": "Effective |M| samples",
    "total_spin_touches": "Total spin touches",
    "work_units": "Accumulated work units = spin touches / N_sites",
    "record_stride": "Trajectory record stride",
    "samples": "Number of production samples",
    "measure_every": "Measurement spacing in update cycles",
    "size": "Linear lattice size N",
    "analytic_energy": "Independent reference energy/site",
    "analytic_magnetization": "Independent reference magnetization/site",
    "analytic_specific_heat": "Independent reference specific heat/site",
    "analytic_susceptibility": "Independent reference susceptibility/site",
    "method_used": "Actual method used",
    "rhat_energy": "R-hat for energy",
    "rhat_abs_magnetization": "R-hat for |M|/N",
    "samples_per_chain": "Production samples per chain",
    "estimated_equilibration_cycles": "Heuristic equilibration cycle",
    "estimated_equilibration_sweeps": "Heuristic equilibration sweep alias",
    "mean_drift_sigma": "First-half vs second-half mean drift / σ",
    "tau_int_samples": "Integrated autocorrelation time (samples)",
    "tau_int_cycles": "Integrated autocorrelation time (update cycles)",
    "effective_samples": "Effective recorded samples",
    "passed": "Compliance suite passed",
}

DISPLAY_UNITS = {
    "temperature": "J/kB",
    "energy_per_site": "J",
    "energy_per_site_error": "J",
    "energy_per_site_block_error": "J",
    "specific_heat": "kB",
    "analytic_energy": "J",
    "analytic_specific_heat": "kB",
    "tau_int_energy_samples": "samples",
    "tau_int_energy_cycles": "update cycles",
    "tau_int_energy_sweeps": "legacy sweep units",
    "tau_int_abs_magnetization_samples": "samples",
    "tau_int_abs_magnetization_cycles": "update cycles",
    "tau_int_abs_magnetization_sweeps": "legacy sweep units",
    "estimated_equilibration_cycles": "update cycles",
    "estimated_equilibration_sweeps": "legacy sweep units",
    "tau_int_samples": "samples",
    "tau_int_cycles": "update cycles",
    "effective_samples": "samples",
    "effective_samples_energy": "samples",
    "effective_samples_abs_magnetization": "samples",
    "samples": "samples",
    "samples_per_chain": "samples",
    "record_stride": "update cycles",
    "measure_every": "update cycles",
    "work_units": "spin touches / N_sites",
}


def format_display_value(value: object) -> str:
    """Render numbers without UI truncation while keeping round-trip float detail."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return "NaN"
        if np.isposinf(number):
            return "+∞"
        if np.isneginf(number):
            return "−∞"
        return repr(number)
    return str(value)


def readable_label(key: object) -> str:
    key_text = str(key)
    return DISPLAY_LABELS.get(key_text, key_text.replace("_", " ").strip().title())


def table_height(row_count: int) -> int:
    # Prefer generous vertical space over compressed horizontal cards.
    return max(220, min(50000, 38 * int(row_count) + 44))


def render_vertical_values(
    data: dict[str, object] | pd.Series,
    *,
    title: str | None = None,
    context: str | None = None,
) -> None:
    """Show one scalar value per row so no numeric result is squeezed into metric cards."""
    mapping = data.to_dict() if isinstance(data, pd.Series) else dict(data)
    rows: list[dict[str, str]] = []
    for key, value in mapping.items():
        if isinstance(value, (np.ndarray, list, tuple, dict, pd.DataFrame, pd.Series)):
            continue
        row = {
            "Quantity": readable_label(key),
            "Value (full)": format_display_value(value),
            "Unit / meaning": DISPLAY_UNITS.get(str(key), ""),
        }
        if context is not None:
            row = {"Context": context, **row}
        rows.append(row)
    if title:
        st.markdown(f"#### {title}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.caption("No scalar values to display.")
        return
    column_config = {
        "Quantity": st.column_config.TextColumn("Quantity", width="large"),
        "Value (full)": st.column_config.TextColumn("Value (full)", width="large"),
        "Unit / meaning": st.column_config.TextColumn("Unit / meaning", width="large"),
    }
    if "Context" in frame.columns:
        column_config["Context"] = st.column_config.TextColumn("Context", width="medium")
    st.dataframe(
        frame,
        width="stretch",
        height=table_height(len(frame)),
        hide_index=True,
        row_height=36,
        column_config=column_config,
    )


def complete_data_button(key: str, label: str) -> bool:
    """Actual button gate: complete data are built only after the user asks for them."""
    state_key = f"show_complete_{key}"
    current = bool(st.session_state.get(state_key, False))
    button_label = f"Hide {label.lower()}" if current else label
    if st.button(button_label, key=f"{state_key}_button"):
        current = not current
        st.session_state[state_key] = current
    return current


def render_key_frame(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    title: str = "Key data",
) -> None:
    """Render only a small set of high-value columns with full numeric strings."""
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return
    display = frame.loc[:, available].copy()
    for column in available:
        display[column] = display[column].map(format_display_value)
    display = display.rename(columns={column: readable_label(column) for column in available})
    st.markdown(f"#### {title}")
    st.dataframe(
        display,
        width="stretch",
        height=table_height(len(display)),
        hide_index=True,
        row_height=36,
    )


def key_scan_rows(frame: pd.DataFrame, critical: float | None = None) -> pd.DataFrame:
    """Keep first, representative/critical, and last scan rows for the default view."""
    if frame.empty:
        return frame
    indices = [0]
    if len(frame) > 2:
        if critical is not None and "temperature" in frame.columns:
            middle = int(np.argmin(np.abs(frame["temperature"].to_numpy(dtype=float) - float(critical))))
        else:
            middle = len(frame) // 2
        indices.append(middle)
    if len(frame) > 1:
        indices.append(len(frame) - 1)
    unique = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    return frame.iloc[unique].reset_index(drop=True)


def frame_to_long(frame: pd.DataFrame, *, record_name: str = "Record") -> pd.DataFrame:
    """Convert a wide result table into a vertical cell-by-cell table."""
    if frame.empty:
        return pd.DataFrame(columns=[record_name, "Quantity", "Value (full)", "Unit / meaning"])
    rows: list[dict[str, str]] = []
    for record_index, (_, record) in enumerate(frame.iterrows(), start=1):
        context_bits = [f"{record_name} {record_index}"]
        if "temperature" in frame.columns:
            context_bits.append(f"T={format_display_value(record['temperature'])}")
        elif "size" in frame.columns:
            context_bits.append(f"N={format_display_value(record['size'])}")
        elif "method" in frame.columns:
            context_bits.append(str(record["method"]))
        elif "initial" in frame.columns:
            context_bits.append(f"initial={record['initial']}")
        context = " · ".join(context_bits)
        for key, value in record.items():
            rows.append(
                {
                    record_name: context,
                    "Quantity": readable_label(key),
                    "Value (full)": format_display_value(value),
                    "Unit / meaning": DISPLAY_UNITS.get(str(key), ""),
                }
            )
    return pd.DataFrame(rows)


def render_complete_frame(
    frame: pd.DataFrame,
    *,
    title: str = "Complete numerical data",
    record_name: str = "Record",
) -> None:
    st.markdown(f"#### {title}")
    long_frame = frame_to_long(frame, record_name=record_name)
    st.dataframe(
        long_frame,
        width="stretch",
        height=table_height(len(long_frame)),
        hide_index=True,
        row_height=36,
        column_config={
            record_name: st.column_config.TextColumn(record_name, width="large"),
            "Quantity": st.column_config.TextColumn("Quantity", width="large"),
            "Value (full)": st.column_config.TextColumn("Value (full)", width="large"),
            "Unit / meaning": st.column_config.TextColumn("Unit / meaning", width="large"),
        },
    )


def render_array_series(
    series: dict[str, np.ndarray],
    *,
    title: str,
    index_values: np.ndarray | None = None,
    index_label: str = "Index",
) -> None:
    """Show every 1D array element vertically; one result value per row."""
    rows: list[dict[str, str]] = []
    for key, values in series.items():
        array = np.asarray(values)
        if array.ndim != 1:
            continue
        for index, value in enumerate(array):
            coordinate = index_values[index] if index_values is not None and index < len(index_values) else index
            rows.append(
                {
                    index_label: format_display_value(coordinate),
                    "Quantity": readable_label(key),
                    "Value (full)": format_display_value(value),
                    "Unit / meaning": DISPLAY_UNITS.get(key, ""),
                }
            )
    st.markdown(f"#### {title}")
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        width="stretch",
        height=table_height(len(frame)),
        hide_index=True,
        row_height=36,
        column_config={
            index_label: st.column_config.TextColumn(index_label, width="medium"),
            "Quantity": st.column_config.TextColumn("Quantity", width="large"),
            "Value (full)": st.column_config.TextColumn("Value (full)", width="large"),
            "Unit / meaning": st.column_config.TextColumn("Unit / meaning", width="large"),
        },
    )


def render_spin_configuration(
    config_array: np.ndarray,
    *,
    title: str = "Complete final spin data",
) -> None:
    """List every spin explicitly with coordinates; avoids a 128-column squeezed matrix."""
    array = np.asarray(config_array)
    rows: list[dict[str, str]] = []
    if array.ndim == 1:
        for x, spin in enumerate(array):
            rows.append({"x": str(x), "spin": format_display_value(spin)})
    elif array.ndim == 2:
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                rows.append({"x": str(x), "y": str(y), "spin": format_display_value(array[y, x])})
    else:
        rows.append({"spin": f"Unsupported configuration rank: {array.ndim}"})
    st.markdown(f"#### {title}")
    spin_frame = pd.DataFrame(rows)
    spin_columns = {
        "x": st.column_config.TextColumn("x", width="small"),
        "spin": st.column_config.TextColumn("Spin σ", width="large"),
    }
    if "y" in spin_frame.columns:
        spin_columns["y"] = st.column_config.TextColumn("y", width="small")
    st.dataframe(
        spin_frame,
        width="stretch",
        height=table_height(len(spin_frame)),
        hide_index=True,
        row_height=34,
        column_config=spin_columns,
    )


def plot_lines(
    x: np.ndarray,
    series: dict[str, np.ndarray],
    *,
    title: str,
    x_title: str,
    y_title: str,
    log_y: bool = False,
    vline: float | None = None,
) -> go.Figure:
    colors = ["#1f5f99", "#d1495b", "#2a9d8f", "#7b2cbf", "#f4a261", "#111111"]
    figure = go.Figure()
    for index, (name, values) in enumerate(series.items()):
        figure.add_scatter(
            x=x,
            y=values,
            mode="lines",
            name=name,
            line={"width": 2, "color": colors[index % len(colors)]},
        )
    if vline is not None:
        figure.add_vline(x=vline, line_dash="dash", line_color="#111111", annotation_text="Tc")
    figure.update_layout(
        template="plotly_white",
        height=450,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
        hovermode="x unified",
    )
    return figure


def selected_method() -> Method:
    requested = METHOD_LABELS[st.session_state.thermo_method]
    current = params()
    compatible, reason = method_compatibility(current, requested)
    if compatible:
        return requested
    st.warning(f"{reason}; this run will use random-site Metropolis instead.")
    return Method.METROPOLIS


def temperature_grid() -> np.ndarray:
    if st.session_state.use_notebook_grid:
        return notebook_temperature_grid()
    return np.linspace(st.session_state.T_min, st.session_state.T_max, int(st.session_state.T_points))


initialize()

st.markdown(
    """
    <section class="hero">
      <h1>Ising Monte Carlo Lab</h1>
      <p>1D transfer-matrix and 2D Onsager references, four Monte Carlo
      update methods, equilibration diagnostics, and finite-size checks.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Physical configuration")
    load_config(st.file_uploader("Import configuration", type=["json"]))
    st.selectbox("Dimension", [1, 2], key="dimension")
    st.number_input("Linear size N", min_value=4, max_value=128, step=1, key="size")
    st.number_input("Coupling J", min_value=0.001, max_value=100.0, key="coupling")
    st.number_input("Magnetic field h", min_value=-10.0, max_value=10.0, key="field")
    st.number_input("Temperature T (J/kB)", min_value=0.05, max_value=20.0, key="temperature")
    st.number_input("Random seed", min_value=0, max_value=10_000_000, step=1, key="seed")
    st.subheader("Monte Carlo update budget")
    c1, c2 = st.columns(2)
    c1.number_input("Equilibration update cycles", min_value=0, max_value=200000, step=50, key="eq_sweeps")
    c2.number_input("Measurement update cycles", min_value=10, max_value=200000, step=50, key="mc_sweeps")
    st.number_input(
        "Measure every n update cycles",
        min_value=1,
        max_value=1000,
        step=1,
        key="measure_every",
        help="Controls the spacing between production samples used for thermodynamic statistics and autocorrelation diagnostics.",
    )
    st.selectbox("Default scan method", list(METHOD_LABELS), key="thermo_method")
    st.download_button(
        "Download configuration",
        json.dumps(config(), indent=2).encode("utf-8"),
        file_name="ising_lab_config.json",
        mime="application/json",
        width="stretch",
    )
    if st.button("Reset configuration", width="stretch"):
        for key, value in defaults().items():
            st.session_state[key] = value
        for key in RESULT_KEYS:
            st.session_state[key] = None
        st.rerun()
    critical = onsager_tc(float(st.session_state.coupling))
    st.caption(f"2D Onsager Tc = {critical:.6g} J/kB (infinite volume, h = 0)")
    st.caption("1D has no finite-temperature transition at h = 0.")
    st.caption(f"Platform version: {APP_VERSION}")
    if int(st.session_state.size) % 2:
        st.caption("Odd periodic N: checkerboard Metropolis is not bipartite and will fall back to random-site Metropolis; heat-bath uses a sequential exact fallback.")

overview_tab, methods_tab, equilibration_tab, d1_tab, d2_tab, finite_tab, snapshot_tab, external_tab, validation_tab = st.tabs(
    [
        "Overview",
        "Method comparison",
        "Equilibration",
        "1D vs exact",
        "2D vs Onsager",
        "Finite size",
        "Snapshots",
        "External data",
        "Validation",
    ]
)

with overview_tab:
    st.subheader("What this platform investigates")
    st.latex(r"\mathcal{H} = -J\sum_{\langle ij\rangle}\sigma_i\sigma_j - h\sum_i\sigma_i,\quad \sigma_i=\pm 1")
    st.markdown(
        """
        <div class="note"><b>Interpretation rule:</b> a smooth magnetization curve is not proof of
        equilibrium. The original notebook used |&lt;M&gt;|/N, which collapses under sign flips;
        this platform reports &lt;|M|&gt;/N, measures mixing, and compares Monte Carlo to an
        independent exact reference. Finite-N data are not like-for-like with the infinite-volume
        Onsager solution.</div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "The notebook also reused the 2D critical temperature on 1D plots, thresholded extensive "
        "energy variance, and drew one specific-heat axvline on the magnetization panel. Those "
        "defects are repaired here."
    )
    grid = np.linspace(0.6, 3.6, 181)
    one_d = [exact_observables(IsingParams(size=64, coupling=float(st.session_state.coupling), temperature=float(t), dimension=1)) for t in grid]
    two_d = [exact_observables(IsingParams(size=64, coupling=float(st.session_state.coupling), temperature=float(t), dimension=2)) for t in grid]
    overview_frame = pd.DataFrame(
        {
            "temperature": grid,
            "exact_1d_energy_per_site": [row["energy_per_site"] for row in one_d],
            "exact_1d_magnetization": [row["magnetization"] for row in one_d],
            "exact_1d_specific_heat": [row["specific_heat"] for row in one_d],
            "exact_1d_susceptibility": [row["susceptibility"] for row in one_d],
            "onsager_2d_energy_per_site": [row["energy_per_site"] for row in two_d],
            "onsager_2d_magnetization": [row["magnetization"] for row in two_d],
            "onsager_2d_specific_heat": [row["specific_heat"] for row in two_d],
            "onsager_2d_susceptibility": [row["susceptibility"] for row in two_d],
        }
    )

    # Trend first: the default view leads with the physics, not a wall of numbers.
    st.plotly_chart(
        plot_lines(
            grid,
            {
                "1D exact energy": np.array([row["energy_per_site"] for row in one_d]),
                "2D Onsager energy": np.array([row["energy_per_site"] for row in two_d]),
            },
            title="Independent exact energy references",
            x_title="Temperature T (J/kB) — scanned independent variable",
            y_title="Energy per site (J)",
            vline=onsager_tc(float(st.session_state.coupling)),
        ),
        width="stretch",
    )
    st.plotly_chart(
        plot_lines(
            grid,
            {
                "1D exact M = 0": np.array([row["magnetization"] for row in one_d]),
                "2D Onsager spontaneous M": np.array([row["magnetization"] for row in two_d]),
            },
            title="1D has no finite-T transition; 2D does",
            x_title="Temperature T (J/kB)",
            y_title="Magnetization per site",
            vline=onsager_tc(float(st.session_state.coupling)),
        ),
        width="stretch",
    )

    render_vertical_values(
        {
            "exact_1d_reference": "Transfer matrix",
            "exact_2d_reference": "Onsager / Yang",
            "critical_temperature_2d": onsager_tc(float(st.session_state.coupling)),
        },
        title="Platform reference data",
    )
    render_key_frame(
        key_scan_rows(overview_frame, onsager_tc(float(st.session_state.coupling))),
        ["temperature", "exact_1d_energy_per_site", "onsager_2d_energy_per_site", "onsager_2d_magnetization"],
        title="Key reference points",
    )
    if complete_data_button("overview_reference", "More data / complete reference grid"):
        render_complete_frame(
            overview_frame,
            title="Complete independent-reference grid",
            record_name="Reference point",
        )

with methods_tab:
    st.subheader("Four Monte Carlo methods versus the independent exact reference")
    c1, c2 = st.columns(2)
    c1.number_input("Comparison equilibration cycles", min_value=0, max_value=50000, step=50, key="comparison_eq")
    c2.number_input("Comparison measurement cycles", min_value=10, max_value=50000, step=50, key="comparison_mc")
    wolff_blocked = abs(float(st.session_state.field)) > 0
    if wolff_blocked:
        st.warning("Wolff is omitted while h ≠ 0 because the cluster embedding used here is zero-field.")
    if int(st.session_state.size) % 2:
        st.warning("Checkerboard Metropolis is omitted for odd periodic N. Heat-bath remains valid through a sequential single-site fallback.")
    if st.button("Run method comparison", type="primary"):
        bar, update = progress("Method comparison")
        st.session_state.comparison_result = method_comparison(
            params(),
            equilibration_sweeps=int(st.session_state.comparison_eq),
            measurement_sweeps=int(st.session_state.comparison_mc),
            measure_every=int(st.session_state.measure_every),
            seed=int(st.session_state.seed),
            progress_callback=update,
        )
        bar.progress(1.0, text="Method comparison complete")
    if st.session_state.comparison_result is not None:
        result = st.session_state.comparison_result
        metrics = []
        for name, data in result.items():
            if name == "analytic":
                metrics.append(
                    {
                        "method": name,
                        "energy_per_site": data["energy_per_site"],
                        "mean_abs_magnetization": data["magnetization"],
                        "abs_mean_magnetization": data.get("magnetization", np.nan),
                        "specific_heat": data["specific_heat"],
                        "susceptibility_standard": data.get("susceptibility", np.nan),
                        "tau_int_energy_cycles": np.nan,
                        "effective_samples_energy": np.nan,
                        "ensemble": data.get("ensemble", ""),
                    }
                )
                continue
            metrics.append(
                {
                    "method": name,
                    "energy_per_site": data["energy_per_site"],
                    "mean_abs_magnetization": data["mean_abs_magnetization"],
                    "abs_mean_magnetization": data["abs_mean_magnetization"],
                    "specific_heat": data["specific_heat"],
                    "susceptibility_standard": data["susceptibility_standard"],
                    "tau_int_energy_cycles": data["tau_int_energy_cycles"],
                    "effective_samples_energy": data["effective_samples_energy"],
                    "ensemble": "finite-N Monte Carlo",
                }
            )

        mixing = go.Figure()
        for name, data in result.items():
            if name == "analytic":
                continue
            mixing.add_scatter(
                x=np.asarray(data["trajectory_work_units"], dtype=float),
                y=np.asarray(data["trajectory_energy"], dtype=float),
                mode="lines",
                name=name,
            )
        mixing.update_layout(
            template="plotly_white",
            height=450,
            title="Energy mixing versus accumulated spin-touch work",
            xaxis_title="Cumulative work units = spin touches / N_sites",
            yaxis_title="Energy per site (J)",
            hovermode="x unified",
        )

        # Trend first.
        st.plotly_chart(mixing, width="stretch")
        st.caption("Local methods contribute one work unit per full lattice sweep. Wolff contributes cluster_size/N per cluster update; this avoids the biased state-dependent stopping rule used in the earlier build.")

        metric_frame = pd.DataFrame(metrics)
        render_key_frame(
            metric_frame,
            ["method", "energy_per_site", "mean_abs_magnetization", "specific_heat", "susceptibility_standard"],
            title="Key method-comparison data",
        )

        if complete_data_button("method_outputs", "More data / complete method outputs"):
            render_complete_frame(
                metric_frame,
                title="Complete method-comparison numbers",
                record_name="Method record",
            )
            st.markdown("### Full per-method outputs")
            for method_name, method_data in result.items():
                render_vertical_values(
                    method_data,
                    title=f"All scalar outputs — {method_name}",
                )
                if method_name != "analytic":
                    array_outputs = {
                        key: np.asarray(value)
                        for key, value in method_data.items()
                        if isinstance(value, np.ndarray) and np.asarray(value).ndim == 1 and key != "final_config"
                    }
                    if array_outputs:
                        render_array_series(
                            array_outputs,
                            title=f"All sampled series — {method_name}",
                            index_label="Sample / record index",
                        )
                    if "final_config" in method_data:
                        render_spin_configuration(
                            np.asarray(method_data["final_config"]),
                            title=f"Complete final spin configuration — {method_name}",
                        )

            trajectory_rows = []
            for name, data in result.items():
                if name == "analytic":
                    continue
                trajectory_length = len(np.asarray(data["trajectory_energy"]))
                for index in range(trajectory_length):
                    trajectory_rows.append(
                        {
                            "method": name,
                            "trajectory_index": index + 1,
                            "trajectory_work_units": np.asarray(data["trajectory_work_units"])[index],
                            "trajectory_energy": np.asarray(data["trajectory_energy"])[index],
                            "trajectory_magnetization": np.asarray(data["trajectory_magnetization"])[index],
                        }
                    )
            if trajectory_rows:
                render_complete_frame(
                    pd.DataFrame(trajectory_rows),
                    title="Complete method trajectories",
                    record_name="Trajectory record",
                )
            download_frame("Download method metrics", metric_frame, "method_comparison_metrics.csv")

with equilibration_tab:
    st.subheader("Intensive equilibration diagnostic")
    st.number_input("Diagnostic update cycles", min_value=50, max_value=200000, step=50, key="eq_diagnostic_sweeps")
    st.number_input("Record every n update cycles", min_value=1, max_value=200, step=1, key="record_every")
    if st.button("Run equilibration trajectory", type="primary"):
        bar, update = progress("Equilibration")
        st.session_state.equilibration_result = simulate(
            params(),
            method=selected_method(),
            equilibration_sweeps=0,
            measurement_sweeps=int(st.session_state.eq_diagnostic_sweeps),
            measure_every=1,
            seed=int(st.session_state.seed),
            record_every=int(st.session_state.record_every),
            progress_callback=update,
        )
        bar.progress(1.0, text="Equilibration complete")
    if st.session_state.equilibration_result is not None:
        result = st.session_state.equilibration_result
        energy_series = np.asarray(result["trajectory_energy"])
        times = np.arange(1, energy_series.size + 1) * int(result["record_stride"])
        diagnostic = equilibration_diagnostics(energy_series, int(result["record_stride"]))

        # Trend first.
        st.plotly_chart(
            plot_lines(
                times,
                {"Energy per site": energy_series, "Magnetization per site": np.asarray(result["trajectory_magnetization"])},
                title="Equilibration uses energy per site, not extensive energy",
                x_title="Update cycle (local sweep or Wolff cluster update)",
                y_title="Intensive observable",
            ),
            width="stretch",
        )
        st.caption("The plateau time is a heuristic, not proof of equilibrium. τ_int and effective sample size expose serial correlation that a flat-looking trace can hide.")

        key_diagnostic = {
            key: diagnostic[key]
            for key in ("estimated_equilibration_cycles", "mean_drift_sigma", "tau_int_samples", "tau_int_cycles", "effective_samples")
            if key in diagnostic
        }
        render_vertical_values(key_diagnostic, title="Key equilibration diagnostics")

        frame = pd.DataFrame(
            {
                "sweep": times,
                "energy_per_site": energy_series,
                "magnetization_per_site": np.asarray(result["trajectory_magnetization"]),
            }
        )
        if complete_data_button("equilibration", "More data / complete equilibration data"):
            render_vertical_values(diagnostic, title="Complete equilibration diagnostic values")
            render_complete_frame(
                frame,
                title="Complete equilibration trajectory data",
                record_name="Recorded cycle",
            )
            equilibration_arrays = {
                key: np.asarray(value)
                for key, value in result.items()
                if isinstance(value, np.ndarray) and np.asarray(value).ndim == 1 and key != "final_config"
            }
            render_array_series(
                equilibration_arrays,
                title="All raw 1D arrays from the equilibration run",
                index_label="Array index",
            )
            render_spin_configuration(
                np.asarray(result["final_config"]),
                title="Complete final spin configuration from the equilibration run",
            )
            download_frame("Download equilibration series", frame, "equilibration_series.csv")

    st.divider()
    st.subheader("Multi-chain convergence cross-check")
    st.caption("Runs random, all-plus, and all-minus initial states against the same target distribution. R-hat near 1 supports agreement across chains but is not a proof of global mixing.")
    if st.button("Run multi-chain convergence"):
        bar, update = progress("Multi-chain convergence")
        st.session_state.multi_chain_result = multi_chain_convergence(
            params(),
            method=selected_method(),
            equilibration_sweeps=int(st.session_state.eq_sweeps),
            measurement_sweeps=int(st.session_state.mc_sweeps),
            measure_every=int(st.session_state.measure_every),
            seed=int(st.session_state.seed),
            progress_callback=update,
        )
        bar.progress(1.0, text="Multi-chain convergence complete")
    if st.session_state.multi_chain_result is not None:
        chains = st.session_state.multi_chain_result
        render_vertical_values(
            {
                "method": chains["method"],
                "rhat_energy": chains["rhat_energy"],
                "rhat_abs_magnetization": chains["rhat_abs_magnetization"],
                "samples_per_chain": chains["samples_per_chain"],
                "interpretation": chains["interpretation"],
            },
            title="Key multi-chain convergence data",
        )
        chain_frame = pd.DataFrame(chains["chains"])
        if complete_data_button("multi_chain", "More data / complete multi-chain data"):
            render_complete_frame(
                chain_frame,
                title="Complete per-chain numerical data",
                record_name="Chain",
            )
            download_frame("Download multi-chain summary", chain_frame, "ising_multi_chain_summary.csv")

with d1_tab:
    st.subheader("1D thermodynamics versus the periodic transfer matrix")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Minimum T", min_value=0.05, max_value=10.0, key="T_min")
    c2.number_input("Maximum T", min_value=0.1, max_value=20.0, key="T_max")
    c3.slider("Scan points", 5, 60, key="T_points")
    st.checkbox("Use original notebook temperature mesh", key="use_notebook_grid")
    invalid = st.session_state.T_min >= st.session_state.T_max
    if st.button("Run 1D temperature scan", type="primary", disabled=invalid):
        bar, update = progress("1D scan")
        st.session_state.thermo_1d_result = thermodynamic_scan(
            temperature_grid(),
            params(dimension=1),
            method=selected_method(),
            equilibration_sweeps=int(st.session_state.eq_sweeps),
            measurement_sweeps=int(st.session_state.mc_sweeps),
            measure_every=int(st.session_state.measure_every),
            seed=int(st.session_state.seed),
            progress_callback=update,
        )
        bar.progress(1.0, text="1D scan complete")
    if st.session_state.thermo_1d_result is not None:
        result = st.session_state.thermo_1d_result
        temps = result["temperature"]
        frame = pd.DataFrame(result)

        # Plots/trends first.
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo energy": result["energy_per_site"],
                    "Exact finite-N energy": result["analytic_energy"],
                },
                title="1D energy per site",
                x_title="Temperature T (J/kB) — scanned independent variable",
                y_title="Energy per site (J)",
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo <|M|>/N": result["mean_abs_magnetization"],
                    "Notebook |<M>|/N": result["abs_mean_magnetization"],
                    "Exact M = 0": result["analytic_magnetization"],
                },
                title="1D magnetization estimators",
                x_title="Temperature T (J/kB)",
                y_title="Magnetization per site",
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo χ = Var(M)/(N T)": result["susceptibility_standard"],
                    "Absolute-connected diagnostic": result["susceptibility_abs_connected"],
                    "Exact finite-N χ": result["analytic_susceptibility"],
                },
                title="1D susceptibility: thermodynamic χ is kept separate from the |M|-connected diagnostic",
                x_title="Temperature T (J/kB)",
                y_title="Susceptibility per site",
            ),
            width="stretch",
        )

        render_key_frame(
            key_scan_rows(frame),
            ["temperature", "energy_per_site", "analytic_energy", "mean_abs_magnetization", "susceptibility_standard"],
            title="Key 1D scan data (first, middle, last)",
        )
        if complete_data_button("scan_1d", "More data / complete 1D scan"):
            render_complete_frame(
                frame,
                title="Complete 1D temperature-scan data",
                record_name="Temperature point",
            )
            download_frame("Download 1D scan", frame, "ising_1d_temperature_scan.csv")

with d2_tab:
    st.subheader("2D thermodynamics versus Onsager, with Tc on the scan axis")
    if abs(float(st.session_state.field)) > 0:
        st.warning("The Onsager overlay requires h = 0. Monte Carlo can still run at finite field.")
    if st.button("Run 2D temperature scan", type="primary", disabled=st.session_state.T_min >= st.session_state.T_max):
        bar, update = progress("2D scan")
        st.session_state.thermo_2d_result = thermodynamic_scan(
            temperature_grid(),
            params(dimension=2),
            method=selected_method(),
            equilibration_sweeps=int(st.session_state.eq_sweeps),
            measurement_sweeps=int(st.session_state.mc_sweeps),
            measure_every=int(st.session_state.measure_every),
            seed=int(st.session_state.seed),
            progress_callback=update,
        )
        bar.progress(1.0, text="2D scan complete")
    if st.session_state.thermo_2d_result is not None:
        result = st.session_state.thermo_2d_result
        temps = result["temperature"]
        critical = onsager_tc(float(st.session_state.coupling))
        frame = pd.DataFrame(result)

        # Plots/trends first.
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo <|M|>/N": result["mean_abs_magnetization"],
                    "Notebook |<M>|/N": result["abs_mean_magnetization"],
                    "Onsager spontaneous M": result["analytic_magnetization"],
                },
                title="2D magnetization; finite N rounds the infinite-volume jump",
                x_title="Temperature T (J/kB) — scanned independent variable",
                y_title="Magnetization per site",
                vline=critical,
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo energy": result["energy_per_site"],
                    "Onsager energy": result["analytic_energy"],
                },
                title="2D energy per site",
                x_title="Temperature T (J/kB)",
                y_title="Energy per site (J)",
                vline=critical,
            ),
            width="stretch",
        )
        heat = go.Figure()
        heat.add_scatter(x=temps, y=result["specific_heat"], name="Monte Carlo Cv")
        analytic_heat = np.asarray(result["analytic_specific_heat"], dtype=float)
        finite = np.isfinite(analytic_heat)
        heat.add_scatter(x=temps[finite], y=analytic_heat[finite], name="Onsager Cv", line={"dash": "dash"})
        heat.add_vline(x=critical, line_dash="dash", line_color="#111111")
        heat.update_layout(
            template="plotly_white",
            height=450,
            title="Specific heat; Onsager diverges only in the infinite-volume limit",
            xaxis_title="Temperature T (J/kB)",
            yaxis_title="Specific heat per site (kB)",
        )
        st.plotly_chart(heat, width="stretch")
        st.plotly_chart(
            plot_lines(
                temps,
                {
                    "Monte Carlo χ = Var(M)/(N T)": result["susceptibility_standard"],
                    "Absolute-connected diagnostic": result["susceptibility_abs_connected"],
                },
                title="2D susceptibility diagnostics (no analytic Onsager χ overlay is implemented)",
                x_title="Temperature T (J/kB)",
                y_title="Susceptibility per site",
                vline=critical,
            ),
            width="stretch",
        )

        render_key_frame(
            key_scan_rows(frame, critical),
            ["temperature", "energy_per_site", "analytic_energy", "mean_abs_magnetization", "analytic_magnetization"],
            title="Key 2D scan data (first, nearest Tc, last)",
        )
        if complete_data_button("scan_2d", "More data / complete 2D scan"):
            render_complete_frame(
                frame,
                title="Complete 2D temperature-scan data",
                record_name="Temperature point",
            )
            download_frame("Download 2D scan", frame, "ising_2d_temperature_scan.csv")

with finite_tab:
    st.subheader("Finite-size approach to the thermodynamic-limit reference")
    st.text_input("Lattice sizes", key="finite_sizes")
    if st.button("Run finite-size scan", type="primary"):
        try:
            sizes = np.asarray([int(part.strip()) for part in str(st.session_state.finite_sizes).split(",") if part.strip()], dtype=int)
            bar, update = progress("Finite-size scan")
            st.session_state.finite_size_result = finite_size_scan(
                sizes,
                params(),
                method=selected_method(),
                equilibration_sweeps=int(st.session_state.eq_sweeps),
                measurement_sweeps=int(st.session_state.mc_sweeps),
                seed=int(st.session_state.seed),
                progress_callback=update,
            )
            bar.progress(1.0, text="Finite-size scan complete")
        except ValueError as exc:
            st.error(f"Could not parse lattice sizes: {exc}")
    if st.session_state.finite_size_result is not None:
        result = st.session_state.finite_size_result
        if "method_used" in result and len(set(result["method_used"].tolist())) > 1:
            st.warning("Some lattice sizes required a method fallback. See `method_used` in the complete data section.")
        if int(st.session_state.dimension) == 1:
            st.caption("1D exact reference is recomputed separately for every finite N. In 2D the Onsager overlay is the infinite-volume reference.")
        frame = pd.DataFrame(result)

        # Trend first.
        st.plotly_chart(
            plot_lines(
                result["size"],
                {
                    "Monte Carlo <|M|>/N": result["mean_abs_magnetization"],
                    "Independent reference M": result["analytic_magnetization"],
                },
                title="Finite N versus the independent reference at the current T",
                x_title="Linear size N — scanned independent variable",
                y_title="Magnetization per site",
            ),
            width="stretch",
        )

        render_key_frame(
            frame,
            ["size", "mean_abs_magnetization", "analytic_magnetization", "energy_per_site", "method_used"],
            title="Key finite-size data",
        )
        if complete_data_button("finite_size", "More data / complete finite-size scan"):
            render_complete_frame(
                frame,
                title="Complete finite-size scan data",
                record_name="Lattice size point",
            )
            download_frame("Download finite-size scan", frame, "ising_finite_size_scan.csv")

with snapshot_tab:
    st.subheader("Single-run lattice snapshot")
    st.selectbox("Initial condition", ["random", "plus", "minus"], key="snapshot_initial")
    st.number_input("Snapshot sweeps", min_value=1, max_value=20000, step=10, key="snapshot_sweeps")
    if st.button("Generate snapshot", type="primary"):
        bar, update = progress("Snapshot")
        st.session_state.snapshot_result = simulate(
            params(),
            method=selected_method(),
            equilibration_sweeps=0,
            measurement_sweeps=int(st.session_state.snapshot_sweeps),
            measure_every=int(st.session_state.measure_every),
            seed=int(st.session_state.seed),
            initial=str(st.session_state.snapshot_initial),
            progress_callback=update,
        )
        bar.progress(1.0, text="Snapshot complete")
    if st.session_state.snapshot_result is not None:
        result = st.session_state.snapshot_result
        config_array = np.asarray(result["final_config"])
        if config_array.ndim == 1:
            figure = go.Figure(
                go.Heatmap(z=config_array[np.newaxis, :], colorscale="RdBu", zmin=-1, zmax=1)
            )
        else:
            figure = go.Figure(go.Heatmap(z=config_array, colorscale="RdBu", zmin=-1, zmax=1))
        figure.update_layout(
            template="plotly_white",
            height=480,
            title="Final spin configuration",
            xaxis_title="Lattice x",
            yaxis_title="Lattice y",
        )

        # Image first.
        st.plotly_chart(figure, width="stretch")

        key_snapshot = {
            key: result[key]
            for key in (
                "method",
                "temperature",
                "energy_per_site",
                "mean_abs_magnetization",
                "specific_heat",
                "susceptibility_standard",
                "tau_int_energy_cycles",
                "effective_samples_energy",
            )
            if key in result
        }
        render_vertical_values(key_snapshot, title="Key snapshot data")

        if complete_data_button("snapshot", "More data / complete snapshot data"):
            render_vertical_values(result, title="Complete snapshot scalar results")
            snapshot_arrays = {
                key: np.asarray(value)
                for key, value in result.items()
                if isinstance(value, np.ndarray) and np.asarray(value).ndim == 1 and key != "final_config"
            }
            render_array_series(
                snapshot_arrays,
                title="All raw 1D arrays from this snapshot run",
                index_label="Array index",
            )
            render_spin_configuration(config_array)

with external_tab:
    st.subheader("Compare an external measurement or simulator CSV")
    st.write(
        "Upload columns named `temperature` and at least one of `magnetization` or `energy`. "
        "Units must be J/kB for temperature, dimensionless magnetization per site, and J for energy per site."
    )
    uploaded = st.file_uploader("Upload external CSV", type=["csv"], key="external_csv")
    if uploaded is not None:
        try:
            measured = pd.read_csv(uploaded)
            if "temperature" not in measured.columns:
                raise ValueError("CSV must contain a temperature column")
            measured = measured.sort_values("temperature").dropna(subset=["temperature"])
            if len(measured) < 3 or not np.all(np.diff(measured.temperature) > 0):
                raise ValueError("temperature must contain at least three strictly increasing values")
            reference_energy = []
            reference_magnetization = []
            for temperature in measured.temperature:
                reference = exact_observables(params(temperature=float(temperature)))
                reference_energy.append(reference["energy_per_site"])
                reference_magnetization.append(reference["magnetization"])
            comparison = measured.copy()
            comparison["model_energy"] = reference_energy
            comparison["model_magnetization"] = reference_magnetization
            metrics = []
            if "energy" in comparison.columns:
                residual = comparison.energy.to_numpy() - comparison.model_energy.to_numpy()
                comparison["energy_residual"] = residual
                metrics.append(("Energy RMSE", float(np.sqrt(np.nanmean(residual**2)))))
            if "magnetization" in comparison.columns:
                residual = comparison.magnetization.to_numpy() - comparison.model_magnetization.to_numpy()
                comparison["magnetization_residual"] = residual
                metrics.append(("Magnetization RMSE", float(np.sqrt(np.nanmean(residual**2)))))

            series = {"Independent reference energy": comparison.model_energy.to_numpy()}
            if "energy" in comparison.columns:
                series["External energy"] = comparison.energy.to_numpy()

            # Trend first.
            st.plotly_chart(
                plot_lines(
                    comparison.temperature.to_numpy(),
                    series,
                    title="External data versus independent reference",
                    x_title="Temperature T (J/kB)",
                    y_title="Energy per site (J)",
                    vline=onsager_tc(float(st.session_state.coupling)) if int(st.session_state.dimension) == 2 else None,
                ),
                width="stretch",
            )
            if metrics:
                render_vertical_values(
                    {label: value for label, value in metrics},
                    title="Key external-data comparison metrics",
                )
            render_key_frame(
                key_scan_rows(comparison, onsager_tc(float(st.session_state.coupling)) if int(st.session_state.dimension) == 2 else None),
                ["temperature", "energy", "model_energy", "magnetization", "model_magnetization"],
                title="Key aligned external-data rows",
            )
            if complete_data_button("external", "More data / complete external-data comparison"):
                render_complete_frame(
                    comparison,
                    title="Complete aligned external-data comparison",
                    record_name="External data row",
                )
                download_frame("Download aligned comparison", comparison, "ising_external_comparison.csv")
            st.session_state.external_result = comparison
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f"Could not interpret CSV: {exc}")

with validation_tab:
    st.subheader("Built-in physical and numerical compliance checks")
    st.markdown(
        """
        - 1D energy uses the periodic transfer matrix, not a high-T guess;
        - 2D energy at Tc equals `-√2 J` from Onsager;
        - spontaneous magnetization uses `<|M|>/N`, not the notebook's `|<M>|/N`;
        - 1D plots no longer inherit the 2D critical temperature as if a transition existed;
        - equilibration is diagnosed from energy per site;
        - local methods and Wolff are compared using explicit spin-touch work; one Wolff cluster is one unbiased Markov transition;
        - odd periodic lattices never use invalid two-color checkerboard updates;
        - thermodynamic susceptibility is `Var(M)/(N T)` and the absolute-magnetization connected quantity is labelled separately;
        - autocorrelation time and effective sample size are reported for production observables;
        - all parameters carry explicit units (`J`, `h`, `T` in `J/kB`).
        """
    )
    if st.button("Run compliance suite", type="primary"):
        critical = onsager_tc()
        onsager = exact_observables(IsingParams(temperature=critical, dimension=2))
        one_d = exact_observables(IsingParams(size=48, temperature=1.0, dimension=1))
        low_t = simulate(
            IsingParams(size=10, temperature=1.4, dimension=2),
            method=Method.WOLFF,
            equilibration_sweeps=60,
            measurement_sweeps=120,
            seed=4,
        )
        estimators = magnetization_estimators(np.array([50.0, -40.0, 30.0, -20.0]), 100)
        report = {
            "Onsager energy at Tc": onsager["energy_per_site"],
            "1D energy at T=1": one_d["energy_per_site"],
            "2D Wolff <|M|> at T=1.4": low_t["mean_abs_magnetization"],
            "notebook estimator gap": estimators["mean_abs_magnetization"] - estimators["abs_mean_magnetization"],
        }
        report["passed"] = bool(
            abs(float(onsager["energy_per_site"]) + np.sqrt(2.0)) < 1e-10
            and abs(float(one_d["energy_per_site"]) + np.tanh(1.0)) < 5e-3
            and float(low_t["mean_abs_magnetization"]) > 0.9
            and float(report["notebook estimator gap"]) > 0.3
        )
        st.session_state.validation_result = report
    if st.session_state.validation_result is not None:
        report = st.session_state.validation_result
        if report["passed"]:
            st.success("Compliance suite passed.")
        else:
            st.error("One or more compliance checks failed.")
        key_report = {key: value for key, value in report.items() if key != "passed"}
        render_vertical_values(key_report, title="Key compliance values")
        if complete_data_button("validation", "More data / complete compliance results"):
            render_vertical_values(
                report,
                title="Complete compliance-suite numerical results",
            )
