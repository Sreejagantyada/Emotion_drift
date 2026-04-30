# src/analysis_drifts_effectsize.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def load_data(path="results/drift_metrics.csv"):
    return pd.read_csv(path)

def compute_summary(df, cols):
    summary = {col: {"mean": df[col].mean(),
                     "std":  df[col].std(),
                     "min":  df[col].min(),
                     "max":  df[col].max()}
               for col in cols}
    return pd.DataFrame(summary).T

def plot_box_by_language(df, sem_cols, emo_cols, langs):
    plt.figure(figsize=(14,6))
    # semantic
    plt.subplot(1,2,1)
    sns.boxplot(data=[df[col] for col in sem_cols])
    plt.xticks(range(len(langs)), langs)
    plt.title("Semantic Drift by Language")
    plt.ylabel("Semantic Drift")
    # emotion
    plt.subplot(1,2,2)
    sns.boxplot(data=[df[col] for col in emo_cols])
    plt.xticks(range(len(langs)), langs)
    plt.title("Emotion Drift by Language")
    plt.ylabel("Emotion Drift")
    plt.tight_layout()
    plt.show()

def scatter_sem_vs_emo(df, sem_col, emo_col, lang_label):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=df[sem_col], y=df[emo_col], alpha=0.4)
    plt.xlabel(f"{lang_label} Semantic Drift")
    plt.ylabel(f"{lang_label} Emotion Drift")
    plt.title(f"{lang_label} Semantic vs Emotion Drift")
    plt.show()

def cohen_d(mean1, std1, n1, mean2, std2, n2):
    """Compute Cohen's d for two independent groups."""
    # pooled sd
    pooled_sd = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_sd
    return d

def group_summary_by_label(df, label_col, drift_cols):
    return df.groupby(label_col)[drift_cols].agg(["mean", "std", "count"])

if __name__ == "__main__":
    df = load_data()

    sem_cols = ["hindi_semantic_drift", "tamil_semantic_drift", "telugu_semantic_drift"]
    emo_cols = ["hindi_emotion_drift",  "tamil_emotion_drift",  "telugu_emotion_drift"]
    langs    = ["Hindi", "Tamil", "Telugu"]

    print("=== Summary Statistics ===")
    print(compute_summary(df, sem_cols + emo_cols))

    plot_box_by_language(df, sem_cols, emo_cols, langs)

    for lang, sem_col, emo_col in zip(langs, sem_cols, emo_cols):
        scatter_sem_vs_emo(df, sem_col, emo_col, lang)

    # Effect size between languages – example: Telugu vs Hindi
    summary = compute_summary(df, ["telugu_semantic_drift", "hindi_semantic_drift",
                                   "telugu_emotion_drift",  "hindi_emotion_drift"])
    n = len(df)
    d_sem = cohen_d(summary.loc["telugu_semantic_drift","mean"],
                   summary.loc["telugu_semantic_drift","std"],   n,
                   summary.loc["hindi_semantic_drift","mean"],
                   summary.loc["hindi_semantic_drift","std"],    n)
    d_emo = cohen_d(summary.loc["telugu_emotion_drift","mean"],
                    summary.loc["telugu_emotion_drift","std"],    n,
                    summary.loc["hindi_emotion_drift","mean"],
                    summary.loc["hindi_emotion_drift","std"],     n)
    print(f"Cohen's d (Telugu vs Hindi) – semantic drift: {d_sem:.3f}")
    print(f"Cohen's d (Telugu vs Hindi) – emotion drift: {d_emo:.3f}")

    # Group by manipulative-label (if you have further emotion labels separate)
    if "manipulative_label" in df.columns:
        grouped = group_summary_by_label(df, "manipulative_label",
                                         sem_cols + emo_cols)
        print("\n=== Summary by manipulative_label ===")
        print(grouped)

    print("Analysis with effect size complete.")

