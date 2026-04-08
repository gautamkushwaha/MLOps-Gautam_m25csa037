
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import wandb


def get_resnet18(num_classes=10):
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def get_test_loader(batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def fgsm_attack(image, epsilon, gradient):
    perturbed = image + epsilon * gradient.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed


def denormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def evaluate_fgsm(model, test_loader, epsilon, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        gradient = images.grad.data
        perturbed = fgsm_attack(images, epsilon, gradient)

        with torch.no_grad():
            outputs = model(perturbed)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


def save_comparison_images(model, test_loader, epsilons, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Get one batch
    images, labels = next(iter(test_loader))
    images, labels = images[:5].to(device), labels[:5].to(device)

    fig, axes = plt.subplots(len(epsilons) + 1, 5, figsize=(15, 3 * (len(epsilons) + 1)))

    # Original images
    for i in range(5):
        img = denormalize(images[i].cpu()).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original\nLabel: {labels[i].item()}")
        axes[0, i].axis('off')

    # Adversarial images for each epsilon
    for eps_idx, epsilon in enumerate(epsilons):
        imgs = images.clone().requires_grad_(True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        gradient = imgs.grad.data
        perturbed = fgsm_attack(imgs, epsilon, gradient)

        for i in range(5):
            img = denormalize(perturbed[i].detach().cpu()).permute(1, 2, 0).numpy()
            axes[eps_idx + 1, i].imshow(img)
            axes[eps_idx + 1, i].set_title(f"ε={epsilon}")
            axes[eps_idx + 1, i].axis('off')

    plt.tight_layout()
    plt.savefig("fgsm_scratch_comparison.png", dpi=100)
    plt.show()
    print("✅ Comparison image saved!")


def main():
    wandb.init(project="DLops-Assignment-5", name="fgsm_scratch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("resnet18_clean.pth", map_location=device))
    print("✅ Model loaded!")

    test_loader = get_test_loader(batch_size=128)

    # Test clean accuracy first
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    clean_acc = 100. * correct / total
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # Test FGSM for different epsilons
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
    results = {}

    for epsilon in epsilons:
        acc = evaluate_fgsm(model, test_loader, epsilon, device)
        results[epsilon] = acc
        print(f"ε={epsilon} | Adversarial Accuracy: {acc:.2f}% | Drop: {clean_acc - acc:.2f}%")

        wandb.log({
            "epsilon": epsilon,
            "adversarial_accuracy_scratch": acc,
            "accuracy_drop_scratch": clean_acc - acc
        })

    # Save comparison images
    save_comparison_images(model, test_loader, epsilons, device)

    # Log image to wandb
    wandb.log({
        "fgsm_scratch_comparison": wandb.Image("fgsm_scratch_comparison.png")
    })

    print("\n📊 Summary:")
    print(f"{'Epsilon':<10} {'Adv Accuracy':<15} {'Accuracy Drop'}")
    print("-" * 40)
    print(f"{'Clean':<10} {clean_acc:.2f}%")
    for eps, acc in results.items():
        print(f"{eps:<10} {acc:.2f}%{'':<9} {clean_acc - acc:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
