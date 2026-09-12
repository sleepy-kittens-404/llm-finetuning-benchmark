"""
Smoke test — checks the modules import and run end to end.
Not a substitute for the real experiments, just a fast sanity check.
Uses a tiny sample size and 1 epoch so it finishes in a minute or two.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from src.model import MODEL_NAME, load_model
from src.data import get_experiment_data
from src.train import run_experiment, epochs_for_size
from src.utils import count_trainable_params

TINY_SIZE = 40  # small enough to run fast, still divisible by 4 classes


def test_data_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bundle = get_experiment_data(train_size=TINY_SIZE, tokenizer=tokenizer, force_rebuild=True)

    assert "input_ids" in bundle["train"].column_names
    assert "input_ids" in bundle["val"].column_names
    assert "input_ids" in bundle["test"].column_names
    assert len(bundle["label_names"]) == 4
    print("data pipeline ok:", len(bundle["train"]), len(bundle["val"]), len(bundle["test"]))


def test_model_loading():
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    label2id = {v: k for k, v in id2label.items()}

    for method in ["full_ft", "lora", "qlora"]:
        model = load_model(method, id2label, label2id)
        trainable, total = count_trainable_params(model)
        print(f"{method}: trainable {trainable:,} / total {total:,}")

        if method == "full_ft":
            assert trainable == total
        else:
            assert trainable < total * 0.05  # LoRA/QLoRA should be a small fraction

        del model


def test_epochs_for_size():
    assert epochs_for_size(500) == 5
    assert epochs_for_size(5000) == 3
    assert epochs_for_size(50000) == 2


def test_full_experiment_run_lora():
    # smallest, fastest method+size combo, just to prove run_experiment works end to end
    row = run_experiment("lora", TINY_SIZE)
    assert "test_accuracy" in row
    assert "test_f1" in row
    assert row["trainable_pct"] < 5
    print("run_experiment smoke test result:", row)


if __name__ == "__main__":
    test_data_pipeline()
    test_model_loading()
    test_epochs_for_size()
    test_full_experiment_run_lora()
    print("\nall smoke tests passed")