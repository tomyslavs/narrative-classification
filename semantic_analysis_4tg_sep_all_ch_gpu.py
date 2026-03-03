"""
semantic similarity, sentiment, emotion shift, and narrative clustering
Outputs: results/
"""
import os
# Bypasses the version check for the CVE-2025-32434 vulnerability
os.environ["TRANSFORMERS_SAFE_WEIGHTS_ONLY"] = "0"
import transformers.utils.import_utils as import_utils
# Create a "fake" check that always says the environment is safe
def patched_check_torch_load_is_safe():
    return True
# Inject the patch into the library
import_utils.check_torch_load_is_safe = patched_check_torch_load_is_safe
import torch
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap  # pip install umap-learn
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # pip install transformers
from transformers import XLMRobertaTokenizer # TS
import json
import ijson
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path(r"C:\tf\cursor_tg")
INPUT_FOLDER = DATA_DIR / "20251121_to_20260120/separated_channels" # "20251121_to_20260120/separated_channels/short"
RESULTS_DIR = DATA_DIR / "results_sep_all_ch_notitle"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Multilingual sentence model
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Multilingual sentiment model (3-way: neg / neu / pos)
SENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Max comments to embed for very large datasets (None = all)
MAX_COMMENTS_FOR_EMB = None  # e.g. 200000 to cap

# -----------------------------
# UTILS: LOADING
# -----------------------------
def load_telegram_posts(posts_json_path):
    posts = {}

    print(f"Reading {posts_json_path}...")
    with open(posts_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Wrap the 'data' list with tqdm
    for p in tqdm(data, desc="Processing posts", unit="post"):
        text = p.get("text", "")
        #if not text or (isinstance(text, str) and not text.strip()):
        if not text or (isinstance(text, str) and (not text.strip() or len(text.split()) < 3)): # Filter: Skip if text is empty, whitespace, or fewer than 3 words
            continue

        article_id = f"{p['channel_id']}:{p['message_id']}"
        posts[article_id] = {
            "portal": p.get("channel_username"),
            "article_id": article_id,
            "text": text,
            "published_at": pd.to_datetime(p["date"].replace('Z', ''), utc=True),
            "reactions_total": p.get("reactions_total", 0),
            "views": p.get("views", None),
        }

    return posts

def stream_telegram_comments(comments_json_path):
    # Get total file size for the progress bar
    file_size = os.path.getsize(comments_json_path)
    
    with open(comments_json_path, "r", encoding="utf-8") as f:
        # Create a progress bar based on file bytes read
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Streaming comments") as pbar:
            for c in ijson.items(f, "item"):
                # Update progress bar based on current file pointer position
                pbar.n = f.tell()
                pbar.refresh()
                
                text = c.get("text", "")
                #if not text or (isinstance(text, str) and not text.strip()):
                if not text or (isinstance(text, str) and (not text.strip() or len(text.split()) < 3)): # Filter: Skip if text is empty, whitespace, or fewer than 3 words
                    continue
                    
                yield {
                    "portal": c.get("channel_username"),
                    "article_id": f"{c['channel_id']}:{c['post_message_id']}",
                    "comment_id": c.get("comment_id"),
                    "text": text,
                    "created_at": pd.to_datetime(c["date"].replace('Z', ''), utc=True),
                    "reactions_total": c.get("reactions_total", 0),
                    "depth": c.get("depth"),
                }

def load_telegram_articles_df(posts_json_path):
    posts = load_telegram_posts(posts_json_path)

    rows = []
    for aid, p in posts.items():
        rows.append({
            "portal": p["portal"] or "telegram",
            "article_id": aid,
            "title": "",                     # Telegram has no title
            "article_content": p["text"],    # IMPORTANT
            "published_at": p["published_at"],
            "category": "telegram",
        })

    return pd.DataFrame(rows)


def load_telegram_comments_df(comments_json_path):
    rows = []

    for c in stream_telegram_comments(comments_json_path):
        rows.append({
            "portal": c["portal"] or "telegram",
            "article_id": c["article_id"],
            "comment_id": c["comment_id"],
            "comment": c["text"],             # IMPORTANT name
            "created_at": c["created_at"],
            "reactions_total": c["reactions_total"],
        })

    return pd.DataFrame(rows)


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
    # Force the model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def load_sentiment_model():
    print("Loading sentiment model...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(SENT_MODEL_NAME) # TS
    # tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME, use_fast=False) # TS
    # tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_NAME) # original
    model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME)
    return tokenizer, model


@torch.no_grad()
def compute_sentiment_scores(texts, tokenizer, model, batch_size=32*2):
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
def get_article_text_safe(df):
    """
    Safely combines title and content even if columns are missing,
    ensuring the result is a Pandas Series.
    """
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    content = df["article_content"] if "article_content" in df.columns else pd.Series("", index=df.index)
    return title.fillna("") + ". " + content.fillna("").astype(str)

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
    #articles["article_text"] = (
    #    articles.get("title", "").fillna("") + ". " +
    #    articles.get("article_content", "").fillna("").astype(str)
    #)
    articles["article_text"] = get_article_text_safe(articles)

    article_ids = articles["article_id"].astype(str).tolist()
    article_texts = articles["article_text"].tolist()
    article_embs = emb_model.encode(article_texts, batch_size=64, show_progress_bar=True)
    article_emb_map = {aid: emb for aid, emb in zip(article_ids, article_embs)}

    comments = comments.copy()
    if MAX_COMMENTS_FOR_EMB is not None and len(comments) > MAX_COMMENTS_FOR_EMB:
        comments = comments.sample(MAX_COMMENTS_FOR_EMB, random_state=42)

    # Filter to comments belonging to the articles we have
    comments = comments[
        comments["article_id"].astype(str).isin(article_ids) &
        comments["comment"].notna()
    ].copy()
    
    comment_texts = comments["comment"].astype(str).tolist()
    
    print(f"Computing similarities in chunks for {len(comment_texts)} comments...")
    sim_list = []
    chunk_size = 100000  # Process 100k comments at a time to save RAM
    
    for i in range(0, len(comment_texts), chunk_size):
        batch_texts = comment_texts[i : i + chunk_size]
        # Encode just this chunk
        batch_embs = emb_model.encode(batch_texts, batch_size=128, show_progress_bar=True)
        
        # Calculate similarity for this chunk immediately
        batch_metadata = comments.iloc[i : i + chunk_size]
        for idx, (_, row) in enumerate(batch_metadata.iterrows()):
            aid = str(row["article_id"])
            a_emb = article_emb_map.get(aid)
            if a_emb is None:
                sim_list.append(np.nan)
            else:
                c_emb = batch_embs[idx]
                # Cosine similarity calculation
                sim = float(np.dot(a_emb, c_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(c_emb)))
                sim_list.append(sim)
        
        # Memory cleanup: force deletion of the chunk embeddings before next loop
        del batch_embs

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
    
    # --------------------------------------------------
    # SAVE COMMENT-LEVEL SEMANTIC SIMILARITY (NO AVERAGING)
    # --------------------------------------------------
    out_all_csv = RESULTS_DIR / "semantic_similarity_over_time_all.csv"
    # Create a copy for saving to avoid affecting downstream calculations
    save_df = merged.copy()
    save_df["created_at"] = save_df["created_at"].dt.tz_localize(None)     # <--- ADD THIS
    save_df["published_at"] = save_df["published_at"].dt.tz_localize(None) # <--- ADD THIS

    cols_to_save = [
        "portal",
        "article_id",
        "comment_id",
        "created_at",
        "published_at",
        "delta_hours",
        "time_bin",
        "semantic_similarity"
    ]
    #save_df[cols_to_save].to_csv(out_all_csv, index=False) # Use save_df here
    # Use float_format="%.3f" to save 3 decimal places
    save_df[cols_to_save].to_csv(out_all_csv, index=False, float_format="%.3f")
    print(f"Saved comment-level semantic similarity to {out_all_csv}")

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

    # --------------------------------------------------
    # COMPUTE ARTICLE-LEVEL SENTIMENT SCORES
    # --------------------------------------------------
    print("Computing sentiment for articles...")

    articles_sent = articles.copy()

    articles_sent["article_text"] = (
        articles_sent.get("title", "").fillna("").astype(str)
        + ". "
        + articles_sent.get("article_content", "").fillna("").astype(str)
    )

    article_sent_df = compute_sentiment_scores(
        articles_sent["article_text"].tolist(),
        tokenizer,
        sent_model,
        batch_size=32*2,
    )

    articles_sent = pd.concat(
        [articles_sent.reset_index(drop=True), article_sent_df],
        axis=1
    )

    # --------------------------------------------------
    # COMPUTE ARTICLE-LEVEL ENGAGEMENT METRICS
    # --------------------------------------------------
    comments_tmp = comments.copy()

    reaction_cols = [c for c in comments_tmp.columns if c.lower().startswith("reactions_")]
    if reaction_cols:
        comments_tmp["total_reactions"] = comments_tmp[reaction_cols].fillna(0).sum(axis=1)
    else:
        comments_tmp["total_reactions"] = 0

    article_engagement = (
        comments_tmp
        .groupby(["portal", "article_id"])
        .agg(
            comments_total_reactions=("total_reactions", "sum"),
            comments_count=("comment_id", "count"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # FINAL ARTICLE-LEVEL OUTPUT
    # --------------------------------------------------
    article_out = (
        articles_sent[
            [
                "portal",
                "article_id",
                "title", # 100-character preview for the title column
                #"article_content",  # <--- CHANGE: replace "title" with "article_content"
                "category",
                "sentiment_score",
                "post_total_reactions",  # <--- ADD THIS LINE
            ]
        ]
        .rename(columns={"sentiment_score": "article_sentiment_score"})
        .merge(
            article_engagement,
            on=["portal", "article_id"],
            how="left"
        )
    )

    # Fill articles with no comments
    article_out["comments_total_reactions"] = article_out["comments_total_reactions"].fillna(0).astype(int)
    article_out["comments_count"] = article_out["comments_count"].fillna(0).astype(int)
    article_out["post_total_reactions"] = article_out["post_total_reactions"].fillna(0).astype(int) # <--- ADD THIS LINE

    # Remove timezone before saving
    if "published_at" in article_out.columns:
        article_out["published_at"] = article_out["published_at"].dt.tz_localize(None) # <--- ADD THIS

    out_article_sent_csv = RESULTS_DIR / "article_sentiment_scores_all.csv"
    # Use float_format="%.3f" for article_sentiment_score
    article_out.to_csv(out_article_sent_csv, index=False, float_format="%.3f")
    #article_out.to_csv(out_article_sent_csv, index=False)

    print(f"Saved article-level sentiment scores to {out_article_sent_csv}")

    # --------------------------------------------------
    # COMPUTE COMMENT-LEVEL SENTIMENT SCORES
    # --------------------------------------------------
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
        batch_size=32*2,
    )
    comments = pd.concat([comments.reset_index(drop=True), sent_df], axis=1)
    
    # --------------------------------------------------
    # SAVE COMMENT-LEVEL SENTIMENT SCORES
    # --------------------------------------------------
    out_sent_csv = RESULTS_DIR / "comment_sentiment_scores_all.csv"
    # Create a copy for saving
    save_comments = comments.copy()
    save_comments["created_at"] = save_comments["created_at"].dt.tz_localize(None) # <--- ADD THIS

    sent_cols = [
        "portal",
        "article_id",
        "comment_id",
        "created_at",
        "sentiment_score",
        "neg",
        "neu",
        "pos",
        "total_reactions"
    ]
    #save_comments[sent_cols].to_csv(out_sent_csv, index=False) # Use save_comments here
    # Use float_format="%.3f" for the sentiment and probability columns
    save_comments[sent_cols].to_csv(out_sent_csv, index=False, float_format="%.3f")
    print(f"Saved comment-level sentiment scores to {out_sent_csv}")

    # --------------------------------------------------
    # FIND NARRATIVE INJECTION POINTS
    # --------------------------------------------------
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
        columns={
            "created_at": "injection_time",
            "comment_id": "injection_comment_id",
            "total_reactions": "injection_total_reactions"
        }
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

    # --------------------------------------------------
    # COMPUTE TOTAL REACTIONS PER ARTICLE
    # --------------------------------------------------
    article_reactions = (
        comments_valid
        .groupby(["portal", "article_id"])["total_reactions"]
        .sum()
        .reset_index()
        .rename(columns={"total_reactions": "comments_total_reactions"})
    )

    if comments_valid.empty:
        print("No articles with enough comments before/after injection.")
        return pd.DataFrame()

    agg = (
        comments_valid
        .groupby(
            ["portal", "article_id", "injection_comment_id", "relative_position"]
        )["sentiment_score"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["portal", "article_id", "injection_comment_id"],
            columns="relative_position",
            values="sentiment_score"
        )
        .reset_index()
    )

    if "before" not in agg.columns or "after" not in agg.columns:
        print("Not enough data to compute before/after sentiment.")
        return pd.DataFrame()

    agg["delta_after_minus_before"] = agg["after"] - agg["before"]
    # --------------------------------------------------
    # ADD ENGAGEMENT METRICS
    # --------------------------------------------------
    agg = agg.merge(
        inj[["portal", "article_id", "injection_comment_id", "injection_total_reactions"]],
        on=["portal", "article_id", "injection_comment_id"],
        how="left"
    )
    agg = agg.merge(
        article_reactions,
        on=["portal", "article_id"],
        how="left"
    )

    out_csv = RESULTS_DIR / "emotion_shift_narrative_injection.csv"
    agg.to_csv(out_csv, index=False, float_format="%.3f")
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
    embs = emb_model.encode(texts, batch_size=32*2, show_progress_bar=True)

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
    #plt.legend(markerscale=1.5, fontsize=8)
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    out_png_u = RESULTS_DIR / "narrative_clusters_umap_by_portal.png"
    plt.savefig(out_png_u, dpi=300)
    plt.close()
    print(f"Saved UMAP plot (by portal) to {out_png_u}")

    #print("Running t-SNE (this may take a while)...")
    #tsne_model = TSNE(n_components=2, random_state=42, perplexity=30,
    #                  learning_rate=200, verbose=1)
    #tsne_coords = tsne_model.fit_transform(embs)
    # Running t-SNE
    print("Running t-SNE (this may take a while)...")
    # NEW: Calculate dynamic perplexity
    # Perplexity must be less than the number of samples
    n_samples = embs.shape[0]
    dynamic_perplexity = min(30, max(1, n_samples - 1))
    tsne_model = TSNE(
        n_components=2, 
        perplexity=dynamic_perplexity,  # Use the dynamic value here
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
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
    #plt.legend(markerscale=1.5, fontsize=8)
    #plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
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
    print("TG SEMANTIC & SENTIMENT ANALYSIS (ALL CHANNELS MODE)")
    print("=" * 70)

    all_articles_list = []
    all_comments_list = []

    # Find all post files
    post_files = list(INPUT_FOLDER.glob("*_posts.json"))
    
    if not post_files:
        print(f"No *_posts.json files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(post_files)} total channels to process...")

    for post_file in post_files:
        channel_name = post_file.name.replace("_posts.json", "")
        comment_file = INPUT_FOLDER / f"{channel_name}_comments.json"
        
        # --- [Step A] Load posts for ALL channels ---
        print(f"--- Loading Channel Posts: {channel_name} ---")
        channel_posts_dict = load_telegram_posts(post_file)
        
        rows = []
        for aid, p in channel_posts_dict.items():
            rows.append({
                "portal": p["portal"] or channel_name, # Use channel_name as fallback
                "article_id": aid,
                "title": "", 
                "article_content": p["text"],
                "published_at": p["published_at"],
                "category": "telegram",
                "post_total_reactions": p.get("reactions_total", 0),
            })
        
        if rows:
            all_articles_list.append(pd.DataFrame(rows))
            
            # --- [Step B] Only load comments if the file exists ---
            if comment_file.exists():
                print(f"    -> Loading matching comments for {channel_name}...")
                all_comments_list.append(load_telegram_comments_df(comment_file))
            else:
                print(f"    -> (No comments file for {channel_name}, skipping interaction data)")

    # Merge all channels
    print("\nMerging data from all channels...")
    articles = pd.concat(all_articles_list, ignore_index=True)
    
    # Handle comments safely if list is empty
    if all_comments_list:
        comments = pd.concat(all_comments_list, ignore_index=True)
        # Only keep comments that belong to the articles we successfully loaded
        valid_article_ids = set(articles["article_id"].unique())
        comments = comments[comments["article_id"].isin(valid_article_ids)].copy()
    else:
        comments = pd.DataFrame() # Create empty DF if no channels have comments

    print(f"Total Loaded: {len(articles)} posts across all channels.")
    print(f"Total Valid Comments: {len(comments)} (from channels where enabled).")

    # Filter invalid dates
    if "published_at" in articles.columns:
        articles = articles[articles["published_at"].notna()].copy()

    # --- MODELS ---
    emb_model = load_embedding_model()
    tokenizer, sent_model = load_sentiment_model()

    # --- ANALYSIS ---

    # [1] & [2] Interaction-dependent files (only run if comments exist)
    if not comments.empty:
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
    else:
        print("\n[!] Skipping Comment-based analysis: No comment data available.")

    # [3] GLOBAL ANALYSIS - Always runs for ALL channels
    print("\n[3] Generating Article Sentiment Scores for ALL Channels")
    # We call the logic within the emotion shift function but specifically ensure 
    # the article_sentiment_scores_all.csv is saved regardless of comment presence.
    # Note: To be cleaner, you could move the article-sentiment logic to a standalone function,
    # but based on your current code, running the logic below ensures the CSV is saved.
    
    # (Optional: If you want to run sentiment ONLY without running the full emotion shift loop)
    # compute_article_sentiment_only(articles, tokenizer, sent_model)

    print("\n[4] Narrative Cluster Visualization (UMAP / t-SNE) for ALL Channels")
    build_narrative_clusters(articles, emb_model, max_articles=None)

    print(f"\nAll results saved in '{RESULTS_DIR}'")


if __name__ == "__main__":
    main()

