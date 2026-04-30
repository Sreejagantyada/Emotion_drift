import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification

def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_batch(tokenizer, model, texts, device, batch_size=64, max_length=128):
    probs_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        # free memory
        del inputs, outputs, logits
        torch.cuda.empty_cache()
    return np.vstack(probs_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",       type=str, required=True)
    parser.add_argument("--processed_path",  type=str, required=True)
    parser.add_argument("--output_path",     type=str, required=True)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--max_length",      type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    tokenizer, model = load_model(args.model_dir, device)
    
    df  = pd.read_csv(args.processed_path)
    import numpy as np
    
    for col in ["hindi_translit", "tamil_translit", "telugu_translit"]:
        print("Predicting for:", col)
        texts = df[col].tolist()
        probs = predict_batch(tokenizer, model, texts, device,
                              batch_size=args.batch_size,
                              max_length=args.max_length)
        df[f"{col}_prob_pos"] = probs[:,1]
        df[f"{col}_prob_neg"] = probs[:,0]
    
    df.to_csv(args.output_path, index=False)
    print("Finished saving predictions to", args.output_path)

