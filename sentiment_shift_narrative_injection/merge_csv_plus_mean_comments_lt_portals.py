import pandas as pd

# 1. Load narrative classification
df_narrative = pd.read_csv("Lithuanian_News_Narrative_Classification_hyp1_opt5_exp.csv")

# Rename to match other datasets
df_narrative = df_narrative.rename(columns={"source_portal": "portal"})

# 2. Load article sentiment data
df_article_sent = pd.read_csv("C:\\tf\\cursor\\results2_5-5\\article_sentiment_scores_all.csv")

# 3. Load comment sentiment data
df_comm_sent = pd.read_csv("C:\\tf\\cursor\\results2_5-5\\comment_sentiment_scores_all.csv")

# Aggregate mean comment sentiment per article
agg_comm_sent = (
    df_comm_sent
    .groupby(["portal", "article_id"])["sentiment_score"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment_score": "mean_comment_sentiment_score"})
)

# 4. Load semantic similarity data
df_comm_sim = pd.read_csv("C:\\tf\\cursor\\results2_5-5\\semantic_similarity_over_time_all.csv")

# Aggregate mean semantic similarity per article
agg_comm_sim = (
    df_comm_sim
    .groupby(["portal", "article_id"])["semantic_similarity"]
    .mean()
    .reset_index()
    .rename(columns={"semantic_similarity": "mean_comment_semantic_similarity"})
)

# 5. Merge narrative with article sentiment
merged = pd.merge(
    df_narrative,
    df_article_sent[
        [
            "portal",
            "article_id",
            "title",
            "category",
            "article_sentiment_score",
            "article_total_reactions",
            "comments_count",
        ]
    ],
    on=["portal", "article_id"],
    how="inner",
)

# 6. Merge aggregated comment sentiment
merged = pd.merge(
    merged,
    agg_comm_sent,
    on=["portal", "article_id"],
    how="left",
)

# 7. Merge aggregated semantic similarity
merged = pd.merge(
    merged,
    agg_comm_sim,
    on=["portal", "article_id"],
    how="left",
)

# 8. Final clean-up
merged["mean_comment_sentiment_score"] = (
    merged["mean_comment_sentiment_score"].fillna(0).round(3)
)

merged["mean_comment_semantic_similarity"] = (
    merged["mean_comment_semantic_similarity"].fillna(0).round(3)
)

# 9. Column ordering
requested_columns = [
    "portal",
    "article_id",
    "title",
    "category",
    "article_sentiment_score",
    "article_total_reactions",
    "comments_count",
    "mean_comment_sentiment_score",
    "mean_comment_semantic_similarity",
    "Lithuanian domestic politics and government policy",
    "Russia, Ukraine and regional security developments",
    "International politics and geopolitical relations",
    "Economic issues, prices and social welfare policies",
    "Crime, accidents and law enforcement incidents",
    "Society, family life and social issues",
    "Culture, entertainment, lifestyle and sports"
]

final_output = merged[[col for col in requested_columns if col in merged.columns]]

# 10. Save output
final_output.to_csv("Merged_Articles_Comments_Data_hyp1_opt5_exp.csv", index=False)

print("Merged dataset saved as Merged_Articles_Comments_Data_hyp1_opt5_exp.csv")