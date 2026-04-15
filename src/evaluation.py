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
