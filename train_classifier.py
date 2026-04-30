# src/train_classifier.py

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, pos_label=1)
    rec   = recall_score(labels, preds, pos_label=1)
    f1    = f1_score(labels, preds, pos_label=1)
    roc   = roc_auc_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist()
    }

def main(args):
    # Model name
    model_name = args.model_name_or_path

    # Load datasets
    train_df = pd.read_csv(args.train_path)
    val_df   = pd.read_csv(args.val_path)
    test_df  = pd.read_csv(args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Enable gradient checkpointing on model (for memory savings)
    model.gradient_checkpointing_enable()

    # Tokenize
    train_enc = tokenizer(list(train_df['english_text']), truncation=True, padding=True)
    val_enc   = tokenizer(list(val_df['english_text']),   truncation=True, padding=True)
    test_enc  = tokenizer(list(test_df['english_text']),  truncation=True, padding=True)

    train_dataset = TextDataset(train_enc, list(train_df['manipulative_label']))
    val_dataset   = TextDataset(val_enc,   list(val_df['manipulative_label']))
    test_dataset  = TextDataset(test_enc,  list(test_df['manipulative_label']))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                    = output_dir,
        num_train_epochs              = args.epochs,
        per_device_train_batch_size   = args.batch_size,
        per_device_eval_batch_size    = args.batch_size,
        gradient_accumulation_steps   = args.grad_accum_steps,
        learning_rate                 = args.learning_rate,
        weight_decay                  = args.weight_decay,
        warmup_steps                  = args.warmup_steps,
        fp16                          = True,
        evaluation_strategy           = args.eval_strategy,
        save_strategy                 = args.save_strategy,
        load_best_model_at_end        = True,
        metric_for_best_model         = "precision",   # focus on precision
        gradient_checkpointing        = True,          # memory-saving
        seed                          = args.seed
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics
    )

    trainer.train()

    preds_output = trainer.predict(test_dataset)
    metrics     = preds_output.metrics
    print("Test metrics:", metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("Model & tokenizer saved to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",          type=str, required=True)
    parser.add_argument("--val_path",            type=str, required=True)
    parser.add_argument("--test_path",           type=str, required=True)
    parser.add_argument("--model_name_or_path",  type=str, default="FacebookAI/xlm-roberta-base",
                        help="Model identifier from Hugging Face")
    parser.add_argument("--output_dir",           type=str, default="models/classifier_tuned",
                        help="Directory to save the model and tokenizer")
    parser.add_argument("--epochs",              type=int,   default=4,   help="Number of training epochs")
    parser.add_argument("--batch_size",          type=int,   default=1,   help="Batch size per device")
    parser.add_argument("--grad_accum_steps",     type=int,   default=16,  help="Gradient accumulation steps")
    parser.add_argument("--learning_rate",        type=float, default=5e-6, help="Initial learning rate")
    parser.add_argument("--weight_decay",         type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps",         type=int,   default=100, help="Warm-up steps for scheduler")
    parser.add_argument("--eval_strategy",        type=str,   default="epoch", help="Evaluation strategy during training")
    parser.add_argument("--save_strategy",        type=str,   default="epoch", help="Model checkpoint save strategy")
    parser.add_argument("--seed",                 type=int,   default=42,   help="Random seed")
    args = parser.parse_args()
    main(args)

