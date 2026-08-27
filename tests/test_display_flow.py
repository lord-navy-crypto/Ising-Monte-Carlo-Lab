from pathlib import Path


APP = (Path(__file__).resolve().parents[1] / "app.py").read_text()


def section(start: str, end: str) -> str:
    return APP.split(start, 1)[1].split(end, 1)[0]


def test_complete_data_is_button_gated() -> None:
    assert 'APP_VERSION = "1.1.3"' in APP
    for key in (
        "overview_reference",
        "method_outputs",
        "equilibration",
        "multi_chain",
        "scan_1d",
        "scan_2d",
        "finite_size",
        "snapshot",
        "external",
        "validation",
    ):
        assert f'complete_data_button("{key}"' in APP
    assert 'with st.expander("More data /' not in APP


def test_visuals_precede_default_data_tables() -> None:
    checks = [
        ("with overview_tab:", "with methods_tab:", "st.plotly_chart", 'title="Platform reference data"'),
        ("with methods_tab:", "with equilibration_tab:", "st.plotly_chart(mixing", 'title="Key method-comparison data"'),
        ("with equilibration_tab:", "with d1_tab:", "st.plotly_chart", 'title="Key equilibration diagnostics"'),
        ("with d1_tab:", "with d2_tab:", "st.plotly_chart", 'title="Key 1D scan data'),
        ("with d2_tab:", "with finite_tab:", "st.plotly_chart", 'title="Key 2D scan data'),
        ("with finite_tab:", "with snapshot_tab:", "st.plotly_chart", 'title="Key finite-size data"'),
        ("with snapshot_tab:", "with external_tab:", "st.plotly_chart", 'title="Key snapshot data"'),
        ("with external_tab:", "with validation_tab:", "st.plotly_chart", 'title="Key external-data comparison metrics"'),
    ]
    for start, end, visual, data in checks:
        body = section(start, end)
        assert visual in body and data in body
        assert body.index(visual) < body.index(data)
