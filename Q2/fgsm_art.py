

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


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
    return DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


def get_raw_test_data():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


def main():
    wandb.init(project="DLops-Assignment-5", name="fgsm_art")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = get_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load("resnet18_clean.pth", map_location=device))
    model.eval()
    print("✅ Model loaded!")

    # Wrap model with ART
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    )

    # Get raw test data (unnormalized)
    x_test, y_test = get_raw_test_data()
    print(f"Test data shape: {x_test.shape}")

    # Clean accuracy
    predictions = classifier.predict(x_test)
    clean_acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test) * 100
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    # FGSM attack for different epsilons
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
    results = {}

    for epsilon in epsilons:
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        x_adv = attack.generate(x=x_test)

        predictions = classifier.predict(x_adv)
        adv_acc = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test) * 100
        results[epsilon] = adv_acc

        print(f"ε={epsilon} | Adversarial Accuracy: {adv_acc:.2f}% | Drop: {clean_acc - adv_acc:.2f}%")

        wandb.log({
            "epsilon": epsilon,
            "adversarial_accuracy_art": adv_acc,
            "accuracy_drop_art": clean_acc - adv_acc
        })

    # Save comparison images
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    x_adv = attack.generate(x=x_test[:10])

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    classes = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    for i in range(10):
        # Original
        axes[0, i].imshow(x_test[i].transpose(1, 2, 0))
        axes[0, i].set_title(f"Original\n{classes[y_test[i]]}", fontsize=8)
        axes[0, i].axis('off')

        # Adversarial
        axes[1, i].imshow(np.clip(x_adv[i].transpose(1, 2, 0), 0, 1))
        pred = np.argmax(classifier.predict(x_adv[i:i+1]))
        axes[1, i].set_title(f"Adv (ART)\n{classes[pred]}", fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle("FGSM with IBM ART (ε=0.1)", fontsize=12)
    plt.tight_layout()
    plt.savefig("fgsm_art_comparison.png", dpi=100)
    plt.show()
    print("✅ Comparison image saved!")

    # Log to wandb
    wandb.log({
        "fgsm_art_comparison": wandb.Image("fgsm_art_comparison.png")
    })

    # Final summary
    print("\n📊 Summary:")
    print(f"{'Epsilon':<10} {'Scratch Acc':<15} {'ART Acc':<15} {'ART Drop'}")
    print("-" * 55)
    scratch_results = {0.01: 46.10, 0.05: 35.68, 0.1: 29.40, 0.2: 23.06, 0.3: 19.40}
    print(f"{'Clean':<10} {93.16:.2f}%")
    for eps, acc in results.items():
        scratch = scratch_results.get(eps, 0)
        print(f"{eps:<10} {scratch:.2f}%{'':<9} {acc:.2f}%{'':<9} {clean_acc - acc:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
