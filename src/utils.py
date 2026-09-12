import os
from datasets import Dataset, ClassLabel

SEED = 42


def stratified_sample(dataset, n_total, seed=SEED):
    df = dataset.to_pandas()
    n_classes = df["label"].nunique()
    n_per_class = n_total // n_classes
    sampled = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=n_per_class, random_state=seed)
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def to_classlabel_dataset(df, label_names):
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds.cast_column("label", ClassLabel(names=label_names))


def train_val_split(dataset, val_fraction=0.1, seed=SEED):
    split = dataset.train_test_split(
        test_size=val_fraction, stratify_by_column="label", seed=seed
    )
    return split["train"], split["test"]


def get_label_maps(raw_dataset):
    label_names = raw_dataset["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in id2label.items()}
    return label_names, id2label, label2id


def make_tokenize_fn(tokenizer, max_length=256):
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)
    return tokenize_function


def tokenize_dataset(dataset, tokenizer, max_length=256):
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    return dataset.map(tokenize_fn, batched=True)


def cache_path(cache_dir, name, train_size=None):
    if train_size is not None:
        return os.path.join(cache_dir, f"{name}_{train_size}")
    return os.path.join(cache_dir, name)


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def reset_vram():
    import torch
    torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb():
    import torch
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def make_compute_metrics():
    import numpy as np
    import evaluate

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    return compute_metrics