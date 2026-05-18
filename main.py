import argparse
import os
import numpy as np
import torch.nn as nn
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test


def run_experiment(dataset_name, dataset_dir, models_to_train, patch_size, num_epochs, batch_size, ratio, device):
    """Run training + testing for all models on a given dataset. Returns results and histories."""
    image, gt, labels = load_mat_hsi(dataset_name, dataset_dir)
    num_classes = len(labels)
    seed = 202401

    dataset_results = {}
    dataset_histories = {}

    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name.upper()}  |  Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        np.random.seed(seed)

        # Split dataset
        trainval_gt, test_gt = sample_gt(gt, ratio, seed)
        train_gt, val_gt = sample_gt(trainval_gt, 0.5, seed)
        del trainval_gt

        train_set = HSIDataset(image, train_gt, patch_size=patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size, drop_last=False, shuffle=False)

        # Load model
        model = get_model(model_name, dataset_name, patch_size)
        model = model.to(device)

        # Optimizer and scheduler
        optimizer, scheduler = load_scheduler(model_name, model)

        # Loss
        criterion = nn.CrossEntropyLoss()

        # Train
        model_dir = f"./checkpoints/{model_name}/{dataset_name}/0"
        try:
            history = train(model, optimizer, criterion, train_loader, val_loader,
                            num_epochs, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print("Training interrupted.")
            history = {'train_loss': [], 'val_acc': []}

        dataset_histories[model_name] = history

        # Test
        probabilities = test(model, model_dir, image, patch_size, num_classes, device)
        predictions = np.argmax(probabilities, axis=-1)

        # Metrics
        run_metrics = metrics(predictions, test_gt, n_classes=num_classes)
        dataset_results[model_name] = run_metrics

        print(f"\nResults for {model_name.upper()} on {dataset_name.upper()}:")
        show_results(run_metrics, label_values=labels)

    return dataset_results, dataset_histories, labels


def plot_training_curves(all_histories, output_dir):
    """Plot training loss and validation accuracy curves for all models across datasets."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(all_histories.keys())
    num_datasets = len(datasets)

    # --- Figure 1: Training Loss ---
    fig, axes = plt.subplots(1, num_datasets, figsize=(7 * num_datasets, 5))
    if num_datasets == 1:
        axes = [axes]

    colors = {'gscvit': '#e74c3c', 'hybridsn': '#3498db'}
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        for model_name, history in all_histories[dataset_name].items():
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'],
                    label=model_name.upper(), color=colors.get(model_name, None), linewidth=2)
        ax.set_title(f'Training Loss — {dataset_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Figure 2: Validation Accuracy ---
    fig, axes = plt.subplots(1, num_datasets, figsize=(7 * num_datasets, 5))
    if num_datasets == 1:
        axes = [axes]

    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        for model_name, history in all_histories[dataset_name].items():
            epochs = range(1, len(history['val_acc']) + 1)
            val_acc_pct = [v * 100 for v in history['val_acc']]
            ax.plot(epochs, val_acc_pct,
                    label=model_name.upper(), color=colors.get(model_name, None), linewidth=2)
        ax.set_title(f'Validation Accuracy — {dataset_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'validation_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_bar_chart(all_results, output_dir):
    """Plot grouped bar charts comparing OA, AA, F1, Kappa across models and datasets."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(all_results.keys())
    metric_keys = ['Accuracy', 'AA', 'F1', 'Kappa']
    metric_labels = ['OA (%)', 'AA (%)', 'F1 (%)', 'Kappa (%)']
    model_names = list(all_results[datasets[0]].keys())
    colors = {'gscvit': '#e74c3c', 'hybridsn': '#3498db'}

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(5 * len(metric_keys), 5))

    x = np.arange(len(datasets))
    width = 0.3

    for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[i]
        for j, model_name in enumerate(model_names):
            values = [all_results[ds][model_name][mk] for ds in datasets]
            bars = ax.bar(x + j * width, values, width,
                          label=model_name.upper(), color=colors.get(model_name, None))
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(ml, fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Model Comparison Across Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_class_accuracy(all_results, all_labels, output_dir):
    """Plot per-class accuracy bar charts for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(all_results.keys())
    model_names = list(all_results[datasets[0]].keys())
    colors = {'gscvit': '#e74c3c', 'hybridsn': '#3498db'}

    for dataset_name in datasets:
        labels = all_labels[dataset_name]
        n_classes = len(labels)

        fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.8), 6))
        x = np.arange(n_classes)
        width = 0.35

        for j, model_name in enumerate(model_names):
            class_acc = all_results[dataset_name][model_name]['class acc']
            ax.bar(x + j * width, class_acc, width,
                   label=model_name.upper(), color=colors.get(model_name, None), alpha=0.85)

        ax.set_title(f'Per-Class Accuracy — {dataset_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'class_accuracy_{dataset_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSC-ViT: Hyperspectral Image Classification")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./results")
    opts = parser.parse_args()

    device = torch.device(f"cuda:{opts.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_to_train = ["gscvit", "hybridsn"]
    dataset_list = ["ip", "sa"]

    all_results = {}
    all_histories = {}
    all_labels = {}

    # ---- Run experiments on each dataset ----
    for dataset_name in dataset_list:
        print(f"\n{'#'*60}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'#'*60}")

        results, histories, labels = run_experiment(
            dataset_name=dataset_name,
            dataset_dir=opts.dataset_dir,
            models_to_train=models_to_train,
            patch_size=opts.patch_size,
            num_epochs=opts.epoch,
            batch_size=opts.bs,
            ratio=opts.ratio,
            device=device,
        )
        all_results[dataset_name] = results
        all_histories[dataset_name] = histories
        all_labels[dataset_name] = labels

    # ---- Generate Plots ----
    print(f"\n{'='*60}")
    print("  GENERATING PLOTS")
    print(f"{'='*60}")

    plot_training_curves(all_histories, opts.output_dir)
    plot_comparison_bar_chart(all_results, opts.output_dir)
    plot_class_accuracy(all_results, all_labels, opts.output_dir)

    # ---- Final Summary Table ----
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<10} {'Model':<12} | {'OA (%)':>8} | {'AA (%)':>8} | {'F1 (%)':>8} | {'Kappa':>8}")
    print("-" * 70)
    for dataset_name in dataset_list:
        for model_name in models_to_train:
            r = all_results[dataset_name][model_name]
            print(f"{dataset_name.upper():<10} {model_name.upper():<12} | "
                  f"{r['Accuracy']:>8.2f} | {r['AA']:>8.2f} | {r['F1']:>8.2f} | {r['Kappa']:>8.2f}")

    print(f"\nAll plots saved to: {os.path.abspath(opts.output_dir)}/")