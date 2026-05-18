import os
import torch
import argparse
import numpy as np
from utils.dataset import load_mat_hsi
from models.get_model import get_model
from train import test
from utils.utils import metrics, show_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSC-ViT Evaluation — Indian Pines Dataset")
    parser.add_argument("--model", type=str, default="gscvit", choices=["gscvit", "hybridsn"])
    parser.add_argument("--dataset_name", type=str, default="ip")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--weights", type=str, default="./checkpoints/gscvit/ip/0")
    opts = parser.parse_args()

    device = torch.device(f"cuda:{opts.device}" if torch.cuda.is_available() else "cpu")

    print(f"Dataset   : {opts.dataset_name}")
    print(f"Patch size: {opts.patch_size}")
    print(f"Model     : {opts.model}")

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)
    num_classes = len(labels)

    # Load model and weights
    model = get_model(opts.model, opts.dataset_name, opts.patch_size)
    weights_path = os.path.join(opts.weights, "model_best.pth")
    print(f"Loading weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Run inference across the full image
    probabilities = test(model, opts.weights, image, opts.patch_size, num_classes, device=device)
    prediction = np.argmax(probabilities, axis=-1)

    # Compute metrics: OA, AA, F1, Kappa
    run_results = metrics(prediction, gt, n_classes=num_classes)
    show_results(run_results, label_values=labels)
