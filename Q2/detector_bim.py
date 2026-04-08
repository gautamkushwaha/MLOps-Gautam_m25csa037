
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wandb
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier


def get_resnet34_detector():
    model = resnet34(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 2)
    return model


def get_resnet18_classifier():
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def get_raw_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=False)

    x_train, y_train = next(iter(train_loader))
    x_test, y_test = next(iter(test_loader))

    return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()


def main():
    wandb.init(project="DLops-Assignment-5", name="detector_bim")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet18 classifier
    classifier_model = get_resnet18_classifier().to(device)
    classifier_model.load_state_dict(torch.load("resnet18_clean.pth", map_location=device))
    classifier_model.eval()
    print("✅ ResNet18 classifier loaded!")

    # Wrap with ART
    criterion = nn.CrossEntropyLoss()
    optimizer_cls = optim.SGD(classifier_model.parameters(), lr=0.01)

    art_classifier = PyTorchClassifier(
        model=classifier_model,
        loss=criterion,
        optimizer=optimizer_cls,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    )

    # Get data
    print("Loading data...")
    x_train, y_train, x_test, y_test = get_raw_data()

    # Generate BIM adversarial examples
    print("Generating BIM adversarial examples...")
    bim_attack = BasicIterativeMethod(
        estimator=art_classifier,
        eps=0.1,
        eps_step=0.01,
        max_iter=40
    )

    x_train_adv = bim_attack.generate(x=x_train)
    x_test_adv = bim_attack.generate(x=x_test)
    print("✅ BIM adversarial examples generated!")

    # Create binary detection dataset
    x_train_combined = np.concatenate([x_train, x_train_adv], axis=0)
    y_train_binary = np.array([0] * len(x_train) + [1] * len(x_train_adv))

    x_test_combined = np.concatenate([x_test, x_test_adv], axis=0)
    y_test_binary = np.array([0] * len(x_test) + [1] * len(x_test_adv))

    # Shuffle
    train_idx = np.random.permutation(len(x_train_combined))
    x_train_combined = x_train_combined[train_idx]
    y_train_binary = y_train_binary[train_idx]

    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train_combined)
    y_train_tensor = torch.LongTensor(y_train_binary)
    x_test_tensor = torch.FloatTensor(x_test_combined)
    y_test_tensor = torch.LongTensor(y_test_binary)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Train detector
    detector = get_resnet34_detector().to(device)
    criterion_det = nn.CrossEntropyLoss()
    optimizer_det = optim.Adam(detector.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_det, T_max=20)

    print("\nTraining BIM detector...")
    best_acc = 0

    for epoch in range(1, 21):
        # Train
        detector.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_det.zero_grad()
            outputs = detector(images)
            loss = criterion_det(outputs, labels)
            loss.backward()
            optimizer_det.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # Evaluate
        detector.eval()
        test_correct, test_total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = detector(images)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100. * test_correct / test_total
        scheduler.step()

        print(f"Epoch {epoch}/20 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Detection Acc: {test_acc:.2f}%")

        wandb.log({
            "epoch": epoch,
            "bim_train_loss": train_loss,
            "bim_train_accuracy": train_acc,
            "bim_detection_accuracy": test_acc
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(detector.state_dict(), "detector_bim.pth")
            print(f"  ✅ Best detector saved! Detection Acc: {best_acc:.2f}%")

    print(f"\n🎉 BIM Detector Done! Best Detection Accuracy: {best_acc:.2f}%")
    print(f"Target was: ≥ 70% | Achieved: {best_acc:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
