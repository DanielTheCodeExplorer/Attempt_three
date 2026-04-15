import pandas as pd

from src.evaluation import (
    build_overall_metrics,
    render_evaluation_report,
    summarise_employee_scores,
    summarise_score_bands,
    summarise_scores,
)


def build_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"talentlinkId": "1001", "skill": "Python", "similarity_score": 0.50, "competency_score": 50.0, "score_band": "high"},
            {"talentlinkId": "1001", "skill": "SQL", "similarity_score": 0.20, "competency_score": 20.0, "score_band": "medium"},
            {"talentlinkId": "1002", "skill": "Python", "similarity_score": 0.10, "competency_score": 10.0, "score_band": "low"},
        ]
    )


def test_summarise_scores_groups_by_skill():
    summary = summarise_scores(build_results_df())

    python_row = summary.loc[summary["skill"] == "Python"].iloc[0]
    assert round(float(python_row["average_similarity"]), 2) == 0.30
    assert round(float(python_row["average_competency_score"]), 2) == 30.00
    assert int(python_row["high_band_count"]) == 1


def test_summarise_score_bands_counts_rows():
    summary = summarise_score_bands(build_results_df())

    assert set(summary["score_band"]) == {"high", "medium", "low"}
    assert int(summary["count"].sum()) == 3


def test_summarise_employee_scores_groups_by_employee():
    summary = summarise_employee_scores(build_results_df())

    top_row = summary.iloc[0]
    assert top_row["talentlinkId"] == "1001"
    assert int(top_row["skill_count"]) == 2
    assert int(top_row["high_band_count"]) == 1


def test_build_overall_metrics_and_report():
    results_df = build_results_df()
    metrics = build_overall_metrics(results_df)
    skill_summary = summarise_scores(results_df)
    band_summary = summarise_score_bands(results_df)
    employee_summary = summarise_employee_scores(results_df)
    report = render_evaluation_report(metrics, skill_summary, band_summary, employee_summary)

    assert metrics["score_rows"] == 3
    assert metrics["employee_count"] == 2
    assert "Competency Score Evaluation" in report
    assert "Score Band Distribution" in report
    assert "Python" in report
