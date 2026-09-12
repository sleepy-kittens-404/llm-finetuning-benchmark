import time
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding, AutoTokenizer

from src.model import load_model, MODEL_NAME
from src.data import get_experiment_data
from src.utils import count_trainable_params, reset_vram, get_peak_vram_mb, make_compute_metrics

# per-method hyperparams that actually differ
METHOD_CONFIG = {
    "full_ft": dict(learning_rate=2e-5),
    "lora":    dict(learning_rate=2e-4),
    "qlora":   dict(learning_rate=2e-4),
}

# epoch count scales with train_size instead of a fixed lookup for two sizes
# small datasets overfit fast so they get more epochs, larger ones fewer
def epochs_for_size(train_size):
    if train_size <= 1000:
        return 5
    elif train_size <= 10000:
        return 3
    else:
        return 2


def build_training_args(method, train_size):
    cfg = METHOD_CONFIG[method]
    return TrainingArguments(
        output_dir=f"./results/{method}-{train_size}",
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs_for_size(train_size),
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        logging_steps=10,
        metric_for_best_model="f1",
        seed=42,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
    )


def run_experiment(method, train_size, tokenizer=None):
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    bundle = get_experiment_data(train_size=train_size, tokenizer=tokenizer)
    model = load_model(method, bundle["id2label"], bundle["label2id"])

    training_args = build_training_args(method, train_size)
    compute_metrics = make_compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bundle["train"],
        eval_dataset=bundle["val"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    reset_vram()
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    peak_vram = get_peak_vram_mb()
    trainable, total = count_trainable_params(model)

    test_results = trainer.evaluate(bundle["test"])

    return {
        "method": method,
        "train_size": train_size,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "peak_vram_mb": peak_vram,
        "time_s": elapsed,
        "test_accuracy": test_results["eval_accuracy"],
        "test_f1": test_results["eval_f1"],
    }


def run_all_experiments(methods=("full_ft", "lora", "qlora"), train_sizes=(500, 5000)):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []
    for train_size in train_sizes:
        for method in methods:
            print(f"\n=== Running {method} on {train_size} samples ===")
            row = run_experiment(method, train_size, tokenizer=tokenizer)
            print(row)
            results.append(row)
    return results