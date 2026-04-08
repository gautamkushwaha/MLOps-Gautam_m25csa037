import optuna
import torch
import wandb
import os
from model import get_vit_model
from utils import get_cifar100_loaders
import torch.nn as nn
import torch.optim as optim

# Load once globally
train_loader, val_loader, _ = get_cifar100_loaders(batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    rank = trial.suggest_categorical('rank', [2, 4, 8])
    alpha = trial.suggest_categorical('alpha', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    model = get_vit_model(
        num_classes=100,
        use_lora=True,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    # Only 2 epochs per trial
    model.train()
    for epoch in range(2):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = 100. * correct / total
    print(f"Trial {trial.number} | Rank={rank} Alpha={alpha} "
          f"Dropout={dropout:.3f} LR={lr:.5f} | Val Acc={val_acc:.2f}%")
    return val_acc


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    print("\n🎉 Optuna Done!")
    print(f"Best Val Accuracy: {study.best_value:.2f}%")
    print(f"Best Params: {study.best_params}")

    wandb.init(project="DLops-Assignment-5", name="optuna_search")
    wandb.log({
        "best_val_accuracy": study.best_value,
        "best_rank": study.best_params['rank'],
        "best_alpha": study.best_params['alpha'],
        "best_dropout": study.best_params['dropout'],
        "best_lr": study.best_params['lr']
    })
    wandb.finish()

if __name__ == "__main__":
    main()