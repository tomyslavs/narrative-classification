import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message="Length of IterableDataset"
)

# --- 1. CONFIG & MODEL ---
INPUT_CSV = "Table1_Channel_Final_Research.csv"
DATA_DIR = Path(r"C:\tf\cursor_tg\20251121_to_20260120\separated_channels")
OUTPUT_CSV = "Article_Narrative_Frames_ZeroShot_work_pc_all_posts_batch8_5000ch_v2.csv"

# CANDIDATE_LABELS = ["Military Victory", "Economic Hardship", "Diplomatic Conflict", "Civilian Welfare", "Internal Unrest", "Foreign Intervention"]
CANDIDATE_LABELS = [
    "Military Success",     # Replaces 'Victory' (Success is broader and covers daily gains)
    "Economic Hardship",    # Replaces 'Panic' (Hardship covers inflation, prices, and systemic issues)
    "Internal Unrest",      # Replaces 'Civil' (Internal covers political instability + protests)
    "Foreign Intervention", # Crucial for "us vs them" or "foreign help/threat" narratives
    "Diplomatic Conflict",  # For news about sanctions, UN votes, and international pressure
    "Civilian Welfare"      # Essential for social-focused propaganda or crisis reporting
]
BATCH_SIZE = 8
MAX_CHARACTERS = 5000 # 2000 same as 5000, checked

print("Initializing mDeBERTa-v3...")
classifier = pipeline(
    "zero-shot-classification", 
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 
    device=0 if torch.cuda.is_available() else -1
)

# --- 2. DATA PREP ---
df_channels = pd.read_csv(INPUT_CSV)
master_posts_list = []

print("Loading and filtering posts...")
for _, row in df_channels.iterrows():
    post_file = DATA_DIR / f"{row['Channel Name']}_posts.json"
    if post_file.exists():
        try:
            posts = pd.read_json(post_file)
            
            # --- NEW FILTERING LOGIC ---
            # 1. Fill NaNs and ensure string type
            posts['text'] = posts['text'].fillna('').astype(str)
            
            # 2. Basic whitespace check
            posts = posts[posts['text'].str.strip() != ''].copy()
            
            # 3. Word count check (must be >= 3 words)
            # We split by whitespace and check the length of the resulting list
            posts = posts[posts['text'].apply(lambda x: len(x.split()) >= 3)].copy()
            
            if not posts.empty:
                posts['channel_name'] = row['Channel Name']
                posts['narrative_group'] = row['Narrative']
                master_posts_list.append(posts[['channel_name', 'narrative_group', 'message_id', 'text', 'views']])
        except: 
            continue

# Combine into one master DataFrame
master_df = pd.concat(master_posts_list, ignore_index=True)

# Truncate texts for the model
texts = master_df['text'].str.slice(0, MAX_CHARACTERS).tolist()
total_posts = len(texts)

# --- 3. BATCHED PROCESSING ---
print(f"Ready to process {total_posts} posts (after filtering).")

results_list = []

# Using the tqdm context manager for the progress bar
with tqdm(total=total_posts, desc="Classifying Narratives", unit="post") as pbar:
    for i in range(0, total_posts, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        # Note: pipeline can take a list for batching
        outputs = classifier(
            batch,
            CANDIDATE_LABELS,
            multi_label=True
        )

        # Handle single output vs list output from pipeline
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            score_map = {
                label: round(score, 4)
                for label, score in zip(out["labels"], out["scores"])
            }
            results_list.append(score_map)
            pbar.update(1)

# --- 4. SAVE ---
print("Merging and saving results...")
scores_df = pd.DataFrame(results_list)

# We drop the 'text' column from master_df to keep the CSV size manageable
# The indices will match because we filtered master_df BEFORE creating the 'texts' list
final_df = pd.concat([master_df.drop(columns=['text']), scores_df], axis=1)

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Complete! File saved to {OUTPUT_CSV}")