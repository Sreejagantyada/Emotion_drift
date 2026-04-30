# src/extract_embeddings.py

import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, XLMRobertaModel

def load_model(model_name_or_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model     = XLMRobertaModel.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return tokenizer, model

def extract_embeddings(texts, tokenizer, model, device,
                       batch_size=32, max_length=128):
    embeddings_list = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        inputs      = tokenizer(batch_texts,
                                truncation=True,
                                padding=True,
                                max_length=max_length,
                                return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embs = outputs.pooler_output.cpu().numpy()
            else:
                hidden_states = outputs.last_hidden_state.cpu().numpy()
                embs = np.mean(hidden_states, axis=1)
        embeddings_list.append(embs)
        # free up memory
        del inputs, outputs
        if 'hidden_states' in locals():
            del hidden_states
        torch.cuda.empty_cache()
    return np.vstack(embeddings_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path for embedding extraction")
    parser.add_argument("--processed_path",        type=str, required=True,
                        help="CSV file of processed data with sentence columns")
    parser.add_argument("--embeddings_dir",        type=str, required=True,
                        help="Directory where .npy embeddings will be saved")
    parser.add_argument("--device",                type=str, default="cuda",
                        help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--batch_size",            type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--max_length",            type=int, default=128,
                        help="Maximum token length (truncation)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Running embedding extraction on device:", device)
    tokenizer, model = load_model(args.model_name_or_path, device)

    df = pd.read_csv(args.processed_path)
    os.makedirs(args.embeddings_dir, exist_ok=True)

    for col in ["english_text", "hindi_translit", "tamil_translit", "telugu_translit"]:
        print(f"Extracting embeddings for column: {col}")
        texts = df[col].tolist()
        embs  = extract_embeddings(texts,
                                   tokenizer,
                                   model,
                                   device,
                                   batch_size=args.batch_size,
                                   max_length=args.max_length)
        save_path = os.path.join(args.embeddings_dir, f"{col}.npy")
        np.save(save_path, embs)
        print(f"Saved embeddings to: {save_path}")

    print("All embeddings extracted.")

