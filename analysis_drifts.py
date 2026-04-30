# src/analysis_drifts.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def load_data(path="results/drift_metrics.csv"):
    df = pd.read_csv(path)
    return df

def plot_distributions(df, drift_cols, language_names, kind="boxplot"):
    """
    Plot distributions of drift metrics by language.
    drift_cols: list of column names (strings)
    language_names: matching list of labels (e.g., ["Hindi","Tamil","Telugu"])
    """
    plt.figure(figsize=(12, 6))
    data = [df[col] for col in drift_cols]
    sns.boxplot(data=data)
    plt.xticks(range(len(language_names)), language_names)
    plt.ylabel("Drift value")
    plt.title(f"Distribution of {' / '.join(drift_cols)} across languages")
    plt.show()

def scatter_semantic_vs_emotion(df, sem_col, emo_col, lang_label):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=df[sem_col], y=df[emo_col], alpha=0.4)
    plt.xlabel(f"{lang_label} Semantic Drift")
    plt.ylabel(f"{lang_label} Emotion Drift")
    plt.title(f"{lang_label} Semantic vs Emotion Drift")
    plt.show()

def compute_summary(df, cols):
    summary = {}
    for col in cols:
        summary[col] = {
            "mean": df[col].mean(),
            "std":  df[col].std(),
            "min":  df[col].min(),
            "max":  df[col].max()
        }
    return pd.DataFrame(summary).T

def statistical_tests(df, col1, col2, label1, label2):
    """Perform independent two‐sample t‐test between two sets of drift values."""
    sample1 = df[col1].dropna()
    sample2 = df[col2].dropna()
    t_stat, p_val = ttest_ind(sample1, sample2, equal_var=False)  # Welch’s t-test
    print(f"T-test between {label1} ({col1}) and {label2} ({col2}):")
    print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
    return t_stat, p_val

if __name__ == "__main__":
    df = load_data()

    # 1) Summary statistics
    cols = ["hindi_semantic_drift", "tamil_semantic_drift", "telugu_semantic_drift",
            "hindi_emotion_drift",  "tamil_emotion_drift",  "telugu_emotion_drift"]
    summary_df = compute_summary(df, cols)
    print("Summary statistics:\n", summary_df)

    # 2) Plot semantic drift distributions
    plot_distributions(df,
                       ["hindi_semantic_drift", "tamil_semantic_drift", "telugu_semantic_drift"],
                       ["Hindi","Tamil","Telugu"])
    # 3) Plot emotion drift distributions
    plot_distributions(df,
                       ["hindi_emotion_drift", "tamil_emotion_drift", "telugu_emotion_drift"],
                       ["Hindi","Tamil","Telugu"])

    # 4) Scatter plots for each language
    scatter_semantic_vs_emotion(df, "hindi_semantic_drift",  "hindi_emotion_drift",  "Hindi")
    scatter_semantic_vs_emotion(df, "tamil_semantic_drift",  "tamil_emotion_drift",  "Tamil")
    scatter_semantic_vs_emotion(df, "telugu_semantic_drift", "telugu_emotion_drift", "Telugu")

    # 5) Statistical tests: compare drift across languages
    statistical_tests(df, "hindi_semantic_drift", "telugu_semantic_drift", "Hindi", "Telugu")
    statistical_tests(df, "hindi_emotion_drift",  "telugu_emotion_drift",  "Hindi", "Telugu")
    statistical_tests(df, "tamil_semantic_drift", "telugu_semantic_drift", "Tamil", "Telugu")
    statistical_tests(df, "tamil_emotion_drift",  "telugu_emotion_drift",  "Tamil", "Telugu")

    print("Analysis complete.")

