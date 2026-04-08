
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import argparse
from model import get_vit_model
from utils import get_cifar100_loaders, get_cifar100_classes


def test(model_path, use_lora=False, rank=4, alpha=4, dropout=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_cifar100_loaders(batch_size=64)
    classes = get_cifar100_classes()

    model = get_vit_model(
        num_classes=100,
        use_lora=use_lora,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * 100
    class_total = [0] * 100

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

    overall_acc = 100. * correct / total
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")

    # Class-wise accuracy
    class_acc = [100. * class_correct[i] / class_total[i] for i in range(100)]

    # Plot class-wise histogram
    plt.figure(figsize=(20, 6))
    plt.bar(range(100), class_acc)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Class-wise Test Accuracy | Overall: {overall_acc:.2f}%")
    plt.tight_layout()
    plt.savefig("classwise_accuracy.png")
    plt.show()
    print("Histogram saved!")

    # Log to WandB
    wandb.init(project="DLops-Assignment-5", name="test_results")
    wandb.log({
        "overall_test_accuracy": overall_acc,
        "classwise_accuracy_histogram": wandb.Image("classwise_accuracy.png")
    })
    wandb.finish()

    return overall_acc, class_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    test(args.model_path, args.use_lora, args.rank, args.alpha, args.dropout)
