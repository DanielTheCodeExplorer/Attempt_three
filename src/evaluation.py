import pandas as pd


def summarise_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact evaluation view over the competency scoring output."""

    if results_df.empty:
        return pd.DataFrame(
            columns=["skill", "average_similarity", "average_competency_score", "high_band_count"]
        )

    summary = (
        results_df.groupby("skill", as_index=False)
        .agg(
            average_similarity=("similarity_score", "mean"),
            average_competency_score=("competency_score", "mean"),
            high_band_count=("score_band", lambda values: int((values == "high").sum())),
        )
        .sort_values("average_competency_score", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def summarise_score_bands(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise how many scores fall into each score band."""

    if results_df.empty:
        return pd.DataFrame(columns=["score_band", "count", "ratio"])

    summary = (
        results_df.groupby("score_band", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    total = int(summary["count"].sum())
    summary["ratio"] = (summary["count"] / total).round(4)
    return summary


def summarise_employee_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise average competency results at employee level."""

    if results_df.empty:
        return pd.DataFrame(
            columns=["talentlinkId", "skill_count", "average_similarity", "average_competency_score", "high_band_count"]
        )

    summary = (
        results_df.groupby("talentlinkId", as_index=False)
        .agg(
            skill_count=("skill", "count"),
            average_similarity=("similarity_score", "mean"),
            average_competency_score=("competency_score", "mean"),
            high_band_count=("score_band", lambda values: int((values == "high").sum())),
        )
        .sort_values("average_competency_score", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def build_overall_metrics(results_df: pd.DataFrame) -> dict[str, float | int]:
    """Create overall dataset-level evaluation metrics."""

    if results_df.empty:
        return {
            "score_rows": 0,
            "employee_count": 0,
            "skill_count": 0,
            "average_similarity": 0.0,
            "average_competency_score": 0.0,
            "high_band_ratio": 0.0,
        }

    return {
        "score_rows": int(len(results_df)),
        "employee_count": int(results_df["talentlinkId"].nunique()),
        "skill_count": int(results_df["skill"].nunique()),
        "average_similarity": round(float(results_df["similarity_score"].mean()), 4),
        "average_competency_score": round(float(results_df["competency_score"].mean()), 2),
        "high_band_ratio": round(float((results_df["score_band"] == "high").mean()), 4),
    }


def render_evaluation_report(
    metrics: dict[str, float | int],
    skill_summary: pd.DataFrame,
    band_summary: pd.DataFrame,
    employee_summary: pd.DataFrame,
) -> str:
    """Render a compact markdown evaluation report for the current scoring output."""

    top_skills = skill_summary.head(5)
    bottom_skills = skill_summary.tail(5).sort_values("average_competency_score", ascending=True)
    top_employees = employee_summary.head(5)

    lines = [
        "# Competency Score Evaluation",
        "",
        "## Overview",
        f"- Score rows evaluated: {metrics['score_rows']}",
        f"- Employees evaluated: {metrics['employee_count']}",
        f"- Unique skills evaluated: {metrics['skill_count']}",
        f"- Average similarity score: {metrics['average_similarity']}",
        f"- Average competency score: {metrics['average_competency_score']}",
        f"- High-band ratio: {metrics['high_band_ratio']}",
        "",
        "## Score Band Distribution",
    ]

    if band_summary.empty:
        lines.append("- No score rows were available for evaluation.")
    else:
        for _, row in band_summary.iterrows():
            lines.append(f"- {row['score_band']}: {int(row['count'])} rows ({row['ratio']:.2%})")

    lines.extend(["", "## Highest Average Skills"])
    if top_skills.empty:
        lines.append("- No skill summary available.")
    else:
        for _, row in top_skills.iterrows():
            lines.append(
                f"- {row['skill']}: avg score {row['average_competency_score']:.2f}, "
                f"avg similarity {row['average_similarity']:.4f}, "
                f"high-band count {int(row['high_band_count'])}"
            )

    lines.extend(["", "## Lowest Average Skills"])
    if bottom_skills.empty:
        lines.append("- No skill summary available.")
    else:
        for _, row in bottom_skills.iterrows():
            lines.append(
                f"- {row['skill']}: avg score {row['average_competency_score']:.2f}, "
                f"avg similarity {row['average_similarity']:.4f}, "
                f"high-band count {int(row['high_band_count'])}"
            )

    lines.extend(["", "## Highest Average Employees"])
    if top_employees.empty:
        lines.append("- No employee summary available.")
    else:
        for _, row in top_employees.iterrows():
            lines.append(
                f"- {row['talentlinkId']}: avg score {row['average_competency_score']:.2f} "
                f"across {int(row['skill_count'])} skills, high-band count {int(row['high_band_count'])}"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- This evaluation is descriptive rather than benchmarked against a human-labelled ground truth.",
            "- The results show how the current TF-IDF plus cosine-similarity approach distributes support scores across skills and employees.",
            "- Low average scores indicate that many listed skills are weakly evidenced in the available descriptions under the current reference-text design.",
        ]
    )

    return "\n".join(lines) + "\n"
