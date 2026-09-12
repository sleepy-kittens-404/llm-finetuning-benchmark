import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "./results/results.csv"
CHARTS_DIR = "./results/charts"

METHOD_LABELS = {"full_ft": "Full FT", "lora": "LoRA", "qlora": "QLoRA"}
COLORS = {"full_ft": "#d64545", "lora": "#4577d6", "qlora": "#45a065"}


def load_results(path=RESULTS_CSV):
    return pd.read_csv(path)


def _grouped_bar(df, value_col, ylabel, title, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    train_sizes = sorted(df["train_size"].unique())
    methods = ["full_ft", "lora", "qlora"]

    x = range(len(train_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].sort_values("train_size")
        values = [sub[sub["train_size"] == s][value_col].values[0] for s in train_sizes]
        positions = [xi + (i - 1) * width for xi in x]
        ax.bar(positions, values, width, label=METHOD_LABELS[method], color=COLORS[method])

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in train_sizes])
    ax.set_xlabel("Training set size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_vram(df):
    return _grouped_bar(df, "peak_vram_mb", "Peak VRAM (MB)", "Peak VRAM by method and train size", "vram.png")


def plot_time(df):
    return _grouped_bar(df, "time_s", "Training time (s)", "Training time by method and train size", "time.png")


def plot_accuracy(df):
    return _grouped_bar(df, "test_accuracy", "Test accuracy", "Test accuracy by method and train size", "accuracy.png")


def plot_trainable_params(df):
    return _grouped_bar(df, "trainable_params", "Trainable parameters", "Trainable parameters by method", "trainable_params.png")


def plot_efficiency_scatter(df):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = {500: "o", 5000: "^"}
    for _, row in df.iterrows():
        ax.scatter(
            row["peak_vram_mb"], row["test_accuracy"],
            s=140, color=COLORS[row["method"]], marker=markers[row["train_size"]],
            edgecolors="black", linewidths=0.8,
            label=f"{METHOD_LABELS[row['method']]} ({row['train_size']})",
        )

    ax.set_xlabel("Peak VRAM (MB)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. VRAM cost (circle=500 samples, triangle=5000)")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "efficiency_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(epoch_log_path="./results/epoch_log.csv"):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    log = pd.read_csv(epoch_log_path)
    train_sizes = sorted(log["train_size"].unique())
    methods = ["full_ft", "lora", "qlora"]

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(6 * len(train_sizes), 5), sharey=True)
    if len(train_sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, train_sizes):
        for method in methods:
            sub = log[(log["method"] == method) & (log["train_size"] == size)].sort_values("epoch")
            ax.plot(sub["epoch"], sub["val_loss"], marker="o", color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(f"Validation loss — {size} training samples")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.legend()

    fig.tight_layout()
    out_path = os.path.join(CHARTS_DIR, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_charts(path=RESULTS_CSV):
    df = load_results(path)
    paths = [
        plot_vram(df),
        plot_time(df),
        plot_accuracy(df),
        plot_trainable_params(df),
        plot_efficiency_scatter(df),
        plot_training_curves(),
    ]
    return paths


if __name__ == "__main__":
    paths = generate_all_charts()
    for p in paths:
        print("saved:", p)