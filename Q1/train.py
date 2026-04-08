
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import argparse
from model import get_vit_model, get_trainable_param_count
from utils import get_cifar100_loaders

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def train(config):
    # Init WandB
    run_name = f"lora_r{config['rank']}_a{config['alpha']}" if config['use_lora'] else "no_lora"
    
    wandb.init(
        project="DLops-Assignment-5",
        name=run_name,
        config=config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_cifar100_loaders(batch_size=config['batch_size'])

    # Model
    model = get_vit_model(
        num_classes=100,
        use_lora=config['use_lora'],
        lora_rank=config['rank'],
        lora_alpha=config['alpha'],
        lora_dropout=config['dropout']
    )
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_acc = 0
    trainable_params = get_trainable_param_count(model)
    print(f"Trainable parameters: {trainable_params:,}")

    # Log gradient updates on LoRA weights
    if config['use_lora']:
        wandb.watch(model, log="gradients", log_freq=100)

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_name = f"best_model_{run_name}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"  ✅ Best model saved: {save_name}")

    print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
    wandb.finish()
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    config = vars(args)
    train(config)
