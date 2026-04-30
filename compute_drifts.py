# src/compute_drifts.py

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path: str) -> np.ndarray:
    """Load saved embeddings from a .npy file."""
    return np.load(path)

def compute_semantic_drift(emb_eng: np.ndarray, emb_trans: np.ndarray) -> np.ndarray:
    """Semantic drift = 1 - cosine_similarity between English and transliteration embeddings."""
    sims  = cosine_similarity(emb_eng, emb_trans)
    diag  = np.diag(sims)
    drift = 1.0 - diag
    return drift

def compute_emotion_drift(probs_eng: np.ndarray, probs_trans: np.ndarray) -> np.ndarray:
    """Emotion drift measured by Jensen–Shannon divergence between two distributions."""
    drifts = []
    for p, q in zip(probs_eng, probs_trans):
        jsd_dist = jensenshannon(p, q, base=2)
        jsd      = jsd_dist ** 2
        drifts.append(jsd)
    return np.array(drifts)

if __name__ == "__main__":
    df_path        = "data/processed/processed_data.csv"
    embeddings_dir = "embeddings"
    preds_path     = "results/translit_predictions.csv"
    output_path    = "results/drift_metrics.csv"

    df = pd.read_csv(df_path)
    print("Loaded processed data:", df.shape)

    # Load embeddings (file names you renamed)
    emb_eng   = load_embeddings(os.path.join(embeddings_dir, "english.npy"))
    emb_hindi = load_embeddings(os.path.join(embeddings_dir, "hindi.npy"))
    emb_tamil = load_embeddings(os.path.join(embeddings_dir, "tamil.npy"))
    emb_tel   = load_embeddings(os.path.join(embeddings_dir, "telugu.npy"))

    # Compute semantic drift for each transliteration
    df["hindi_semantic_drift"]  = compute_semantic_drift(emb_eng, emb_hindi)
    df["tamil_semantic_drift"]  = compute_semantic_drift(emb_eng, emb_tamil)
    df["telugu_semantic_drift"] = compute_semantic_drift(emb_eng, emb_tel)

    # Load prediction probabilities
    preds = pd.read_csv(preds_path)
    print("Prediction columns:", preds.columns.tolist())

    # Updated to use your actual column names from predictions
    eng_col_pos   = None  # We don't have english_prob_pos/neg in file
    translit_cols = {
        "hindi": ["hindi_translit_prob_neg", "hindi_translit_prob_pos"],
        "tamil": ["tamil_translit_prob_neg", "tamil_translit_prob_pos"],
        "telugu": ["telugu_translit_prob_neg", "telugu_translit_prob_pos"]
    }

    # For English, we only have the original manipulative_label not probabilities.
    # We'll treat English probability distribution as [1-label, label] -> [neg, pos]
    probs_eng = np.stack([
        1.0 - df["manipulative_label"].to_numpy(),
        df["manipulative_label"].to_numpy()
    ], axis=1)

    # Extract transliteration distributions
    probs_hi = preds[translit_cols["hindi"]].to_numpy()
    probs_ta = preds[translit_cols["tamil"]].to_numpy()
    probs_te = preds[translit_cols["telugu"]].to_numpy()

    # Compute emotion drift
    df["hindi_emotion_drift"]  = compute_emotion_drift(probs_eng, probs_hi)
    df["tamil_emotion_drift"]  = compute_emotion_drift(probs_eng, probs_ta)
    df["telugu_emotion_drift"] = compute_emotion_drift(probs_eng, probs_te)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved drift metrics to: {output_path}")

