"""
LT news portals: semantic similarity, emotion shift, and narrative clustering
Dataset folder: LT_datasets
Outputs: results2/
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

import umap  # pip install umap-learn
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # pip install transformers
import torch


# -----------------------------
# CONFIG
# -----------------------------

DATA_DIR = Path("LT_datasets")
RESULTS_DIR = Path("results222")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Multilingual sentence model (works reasonably for Lithuanian)
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Multilingual sentiment model (3-way: neg / neu / pos)
SENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Max comments to embed for very large datasets (None = all)
MAX_COMMENTS_FOR_EMB = None  # e.g. 200000 to cap


# -----------------------------
# UTILS: LOADING
# -----------------------------

def load_all_portals(data_dir: Path):
    """
    Load all *_articles.csv and *_comments.csv from LT_datasets
    Returns:
        articles_df, comments_df (both with 'portal' column)
    """
    article_files = sorted(glob.glob(str(data_dir / "*_articles.csv")))
    comment_files = sorted(glob.glob(str(data_dir / "*_comments.csv")))

    articles_list = []
    comments_list = []

    for af in article_files:
        portal = Path(af).name.replace("_articles.csv", "")
        a = pd.read_csv(af, low_memory=False)
        a["portal"] = portal
        if "published_at" in a.columns:
            a["published_at"] = pd.to_datetime(a["published_at"], errors="coerce")
        articles_list.append(a)

    for cf in comment_files:
        portal = Path(cf).name.replace("_comments.csv", "")
        c = pd.read_csv(cf, low_memory=False)
        c["portal"] = portal
        if "created_at" in c.columns:
            c["created_at"] = pd.to_datetime(c["created_at"], errors="coerce")
        comments_list.append(c)

    articles_df = pd.concat(articles_list, ignore_index=True) if articles_list else pd.DataFrame()
    comments_df = pd.concat(comments_list, ignore_index=True) if comments_list else pd.DataFrame()

    if "article_id" not in comments_df.columns or "article_id" not in articles_df.columns:
        raise ValueError("Both articles and comments must have 'article_id' column")

    return articles_df, comments_df


# -----------------------------
# MODELS
# -----------------------------

def load_embedding_model():
    print("Loading sentence embedding model...")
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model


def load_sentiment_model():
    print("Loading sentiment model...")
    tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME)
    return tokenizer, model


@torch.no_grad()
def compute_sentiment_scores(texts, tokenizer, model, batch_size=32):
    """
    Compute sentiment logits and probabilities for a list/Series of texts.
    Twitter XLM-R sentiment labels: 0=negative, 1=neutral, 2=positive
    Returns:
        DataFrame with columns: neg, neu, pos, sentiment_score (pos - neg)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i + batch_size])
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    df = pd.DataFrame(all_probs, columns=["neg", "neu", "pos"])
    df["sentiment_score"] = df["pos"] - df["neg"]
    return df


# -----------------------------
# 1) SEMANTIC SIMILARITY OVER TIME
# -----------------------------

def compute_comment_article_similarity_over_time(articles, comments, emb_model,
                                                 time_bin_hours=6):
    """
    For each article, compute cosine similarity between article embedding and each comment,
    then aggregate by time since publication in bins.
    Returns:
        similarity_over_time_df: rows = portal x time_bin, with avg similarity
    """

    print("Encoding article texts...")
    articles = articles.copy()
    articles["article_text"] = (
        articles.get("title", "").fillna("") + ". " +
        articles.get("article_content", "").fillna("").astype(str)
    )

    article_ids = articles["article_id"].astype(str).tolist()
    article_texts = articles["article_text"].tolist()
    article_embs = emb_model.encode(article_texts, batch_size=32, show_progress_bar=True)
    article_emb_map = {aid: emb for aid, emb in zip(article_ids, article_embs)}

    comments = comments.copy()
    if MAX_COMMENTS_FOR_EMB is not None and len(comments) > MAX_COMMENTS_FOR_EMB:
        comments = comments.sample(MAX_COMMENTS_FOR_EMB, random_state=42)

    comments = comments[
        comments["article_id"].astype(str).isin(article_ids) &
        comments["created_at"].notna()
    ].copy()
    if comments.empty:
        raise ValueError("No comments matched with articles for similarity computation")

    print("Encoding comment texts...")
    comment_texts = comments["comment"].fillna("").astype(str).tolist()
    comment_embs = emb_model.encode(comment_texts, batch_size=128, show_progress_bar=True)

    print("Computing cosine similarities...")
    sim_list = []
    for (idx, row), c_emb in zip(comments.iterrows(), comment_embs):
        aid = str(row["article_id"])
        a_emb = article_emb_map.get(aid)
        if a_emb is None:
            sim = np.nan
        else:
            sim = float(cosine_similarity(a_emb.reshape(1, -1), c_emb.reshape(1, -1))[0, 0])
        sim_list.append(sim)

    comments["semantic_similarity"] = sim_list

    art_pub = articles[["article_id", "portal", "published_at"]].copy()
    art_pub["article_id"] = art_pub["article_id"].astype(str)
    comments["article_id"] = comments["article_id"].astype(str)
    merged = comments.merge(art_pub, on="article_id", how="left", suffixes=("", "_article"))
    merged = merged[merged["published_at"].notna() & merged["created_at"].notna()].copy()

    merged["delta_hours"] = (
        merged["created_at"] - merged["published_at"]
    ) / pd.Timedelta(hours=1)
    merged = merged[merged["delta_hours"] >= 0]
    merged["time_bin"] = (merged["delta_hours"] // time_bin_hours) * time_bin_hours

    sim_over_time = (
        merged.groupby(["portal", "time_bin"])["semantic_similarity"]
        .mean()
        .reset_index()
        .sort_values(["portal", "time_bin"])
    )

    out_csv = RESULTS_DIR / "semantic_similarity_over_time.csv"
    sim_over_time.to_csv(out_csv, index=False)
    print(f"Saved semantic similarity over time to {out_csv}")

    plt.figure(figsize=(10, 6))
    for portal, dfp in sim_over_time.groupby("portal"):
        plt.plot(dfp["time_bin"], dfp["semantic_similarity"], marker="o", label=portal)
    plt.xlabel(f"Hours since article publication (binned by {time_bin_hours}h)")
    plt.ylabel("Average comment–article semantic similarity")
    plt.title("Comments Semantic Similarity to Article Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = RESULTS_DIR / "semantic_similarity_over_time.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved plot to {out_png}")

    return sim_over_time


# -----------------------------
# 2) EMOTION SHIFT BEFORE / AFTER NARRATIVE INJECTION
# -----------------------------

def compute_emotion_shift_narrative_injection(articles, comments,
                                              tokenizer, sent_model,
                                              min_comments_before=5,
                                              min_comments_after=5):
    """
    Define 'narrative injection' for each article as the single comment
    with the highest engagement (likes + dislikes + other reactions).
    Measure average sentiment_score before vs after that comment.
    """

    comments = comments.copy()

    reaction_cols = [c for c in comments.columns if c.lower().startswith("reactions_")]
    if reaction_cols:
        comments["total_reactions"] = comments[reaction_cols].fillna(0).sum(axis=1)
    else:
        comments["total_reactions"] = 0

    comments = comments[comments["created_at"].notna()].copy()

    print("Computing sentiment for comments (for emotion shift)...")
    sent_df = compute_sentiment_scores(
        comments["comment"].fillna("").astype(str).tolist(),
        tokenizer,
        sent_model,
        batch_size=32,
    )
    comments = pd.concat([comments.reset_index(drop=True), sent_df], axis=1)

    print("Finding narrative injection points...")

    def pick_injection(group):
        idx_max = group["total_reactions"].idxmax()
        return group.loc[idx_max]

    inj = (
        comments.groupby(["portal", "article_id"], as_index=False)
        .apply(pick_injection)
        .reset_index(drop=True)
    )
    inj = inj[["portal", "article_id", "comment_id", "created_at", "total_reactions"]].rename(
        columns={"created_at": "injection_time", "comment_id": "injection_comment_id"}
    )

    comments = comments.merge(
        inj,
        on=["portal", "article_id"],
        how="left"
    )
    comments = comments[comments["injection_time"].notna()].copy()

    comments["relative_position"] = np.where(
        comments["created_at"] < comments["injection_time"], "before", "after"
    )

    grp = comments.groupby(["portal", "article_id"])
    valid_articles = []
    for (portal, aid), g in grp:
        n_before = (g["relative_position"] == "before").sum()
        n_after = (g["relative_position"] == "after").sum()
        if n_before >= min_comments_before and n_after >= min_comments_after:
            valid_articles.append((portal, aid))

    valid_mask = comments.set_index(["portal", "article_id"]).index.isin(valid_articles)
    comments_valid = comments[valid_mask].copy()

    if comments_valid.empty:
        print("No articles with enough comments before/after injection.")
        return pd.DataFrame()

    agg = (
        comments_valid
        .groupby(["portal", "article_id", "relative_position"])["sentiment_score"]
        .mean()
        .reset_index()
        .pivot_table(index=["portal", "article_id"],
                     columns="relative_position",
                     values="sentiment_score")
        .reset_index()
    )

    if "before" not in agg.columns or "after" not in agg.columns:
        print("Not enough data to compute before/after sentiment.")
        return pd.DataFrame()

    agg["delta_after_minus_before"] = agg["after"] - agg["before"]

    out_csv = RESULTS_DIR / "emotion_shift_narrative_injection.csv"
    agg.to_csv(out_csv, index=False)
    print(f"Saved emotion shift per article to {out_csv}")

    plt.figure(figsize=(8, 5))
    sns.histplot(agg["delta_after_minus_before"], bins=40, kde=True)
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Sentiment shift (after - before)")
    plt.ylabel("Article count")
    plt.title("Emotion Shift Before / After Narrative Injection (all portals)")
    plt.tight_layout()
    out_png = RESULTS_DIR / "emotion_shift_distribution.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved emotion shift distribution plot to {out_png}")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=agg, x="portal", y="delta_after_minus_before")
    plt.axhline(0, color="red", linestyle="--", alpha=0.7)
    plt.xlabel("Portal")
    plt.ylabel("Sentiment shift (after - before)")
    plt.title("Emotion Shift Before / After Narrative Injection by Portal")
    plt.tight_layout()
    out_png2 = RESULTS_DIR / "emotion_shift_by_portal.png"
    plt.savefig(out_png2, dpi=300)
    plt.close()
    print(f"Saved portal-level emotion shift plot to {out_png2}")

    return agg


# -----------------------------
# 3) NARRATIVE CLUSTER VISUALIZATION (UMAP / t-SNE)
# -----------------------------

def build_narrative_clusters(articles, emb_model, max_articles=None):
    """
    One point per article: embedding of title + article_content.
    """
    arts = articles.copy()
    arts["article_text"] = (
        arts.get("title", "").fillna("") + ". " +
        arts.get("article_content", "").fillna("").astype(str)
    )

    if max_articles is not None and len(arts) > max_articles:
        arts = arts.sample(max_articles, random_state=42)

    print(f"Encoding {len(arts)} articles for clustering...")
    texts = arts["article_text"].tolist()
    embs = emb_model.encode(texts, batch_size=32, show_progress_bar=True)

    print("Running UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=25, min_dist=0.1)
    umap_coords = umap_reducer.fit_transform(embs)

    df_umap = arts[["portal", "article_id", "category", "published_at"]].copy()
    df_umap["x"] = umap_coords[:, 0]
    df_umap["y"] = umap_coords[:, 1]

    out_csv_u = RESULTS_DIR / "narrative_clusters_umap.csv"
    df_umap.to_csv(out_csv_u, index=False)
    print(f"Saved UMAP coordinates to {out_csv_u}")

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_umap, x="x", y="y", hue="portal", s=15, alpha=0.7)
    plt.title("Narrative Cluster Visualization (UMAP) - by Portal")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    out_png_u = RESULTS_DIR / "narrative_clusters_umap_by_portal.png"
    plt.savefig(out_png_u, dpi=300)
    plt.close()
    print(f"Saved UMAP plot (by portal) to {out_png_u}")

    print("Running t-SNE (this may take a while)...")
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=30,
                      learning_rate=200, n_iter=1000, verbose=1)
    tsne_coords = tsne_model.fit_transform(embs)

    df_tsne = arts[["portal", "article_id", "category", "published_at"]].copy()
    df_tsne["x"] = tsne_coords[:, 0]
    df_tsne["y"] = tsne_coords[:, 1]

    out_csv_t = RESULTS_DIR / "narrative_clusters_tsne.csv"
    df_tsne.to_csv(out_csv_t, index=False)
    print(f"Saved t-SNE coordinates to {out_csv_t}")

    plt.figure(figsize=(9, 7))
    sns.scatterplot(data=df_tsne, x="x", y="y", hue="portal", s=15, alpha=0.7)
    plt.title("Narrative Cluster Visualization (t-SNE) - by Portal")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    out_png_t = RESULTS_DIR / "narrative_clusters_tsne_by_portal.png"
    plt.savefig(out_png_t, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot (by portal) to {out_png_t}")

    return df_umap, df_tsne


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    print("=" * 70)
    print("LT NEWS PORTALS: SEMANTIC & EMOTION ANALYSIS (results2/)")
    print("=" * 70)

    print("\nLoading datasets from LT_datasets/ ...")
    articles, comments = load_all_portals(DATA_DIR)
    print(f"Loaded {len(articles)} articles and {len(comments)} comments.")

    if "created_at" in comments.columns:
        comments = comments[comments["created_at"].notna()].copy()
    if "published_at" in articles.columns:
        articles = articles[articles["published_at"].notna()].copy()

    emb_model = load_embedding_model()
    tokenizer, sent_model = load_sentiment_model()

    print("\n[1] Comments Semantic Similarity to Article Over Time")
    compute_comment_article_similarity_over_time(
        articles, comments, emb_model, time_bin_hours=6
    )

    print("\n[2] Emotion Shift Before / After Narrative Injection")
    compute_emotion_shift_narrative_injection(
        articles, comments, tokenizer, sent_model,
        min_comments_before=5,
        min_comments_after=5
    )

    print("\n[3] Narrative Cluster Visualization (UMAP / t-SNE)")
    build_narrative_clusters(
        articles, emb_model,
        max_articles=None  # or set e.g. 10000 for speed
    )

    print("\nAll results saved in 'results2/'")


if __name__ == "__main__":
    main()

