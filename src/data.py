import os
from datasets import load_dataset, load_from_disk
from src.utils import (
    stratified_sample,
    to_classlabel_dataset,
    train_val_split,
    get_label_maps,
    tokenize_dataset,
    cache_path,
)

DATASET_NAME = "fancyzhx/ag_news"
CACHE_DIR = "./data"
TEST_SIZE = 1500


def load_raw_dataset():
    return load_dataset(DATASET_NAME)


def get_experiment_data(train_size, tokenizer, raw_dataset=None, force_rebuild=False):
    train_path = cache_path(CACHE_DIR, "train", train_size)
    val_path = cache_path(CACHE_DIR, "val", train_size)
    test_path = cache_path(CACHE_DIR, "test")

    all_cached = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    if all_cached and not force_rebuild:
        train_ds = load_from_disk(train_path)
        val_ds = load_from_disk(val_path)
        test_ds = load_from_disk(test_path)
        raw_dataset = raw_dataset or load_raw_dataset()
        label_names, id2label, label2id = get_label_maps(raw_dataset)
        return {
            "train": train_ds, "val": val_ds, "test": test_ds,
            "label_names": label_names, "id2label": id2label, "label2id": label2id,
        }

    raw_dataset = raw_dataset or load_raw_dataset()
    label_names, id2label, label2id = get_label_maps(raw_dataset)

    train_df = stratified_sample(raw_dataset["train"], train_size)
    train_full = to_classlabel_dataset(train_df, label_names)
    train_ds, val_ds = train_val_split(train_full)
    train_ds = tokenize_dataset(train_ds, tokenizer)
    val_ds = tokenize_dataset(val_ds, tokenizer)

    if os.path.exists(test_path) and not force_rebuild:
        test_ds = load_from_disk(test_path)
    else:
        test_df = stratified_sample(raw_dataset["test"], TEST_SIZE)
        test_full = to_classlabel_dataset(test_df, label_names)
        test_ds = tokenize_dataset(test_full, tokenizer)
        test_ds.save_to_disk(test_path)

    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)

    return {
        "train": train_ds, "val": val_ds, "test": test_ds,
        "label_names": label_names, "id2label": id2label, "label2id": label2id,
    }