#!/usr/bin/env python3


import logging
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from FPRNet.FingerprintDataset import FingerprintDataset
from FPRNet.FPRNet import FPRNet
from step1 import generate_randomized_dataset, load_dataset

# pyright: basic


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    if torch.cuda.is_available():
        logging.info("Training model with CUDA acceleration enabled")
        device = torch.device("cuda")
    else:
        logging.warning("Training model with CPU, this may be slow!")
        device = torch.device("cpu")

    _ = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        _ = model.train()
        running_loss = 0.0

        for reference_image, comparison_image, labels in train_loader:
            # Send to the device
            reference_image = reference_image.to(device)
            comparison_image = comparison_image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(reference_image, comparison_image).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        _ = model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for reference_image, comparison_image, labels in val_loader:
                reference_image = reference_image.to(device)
                comparison_image = comparison_image.to(device)
                labels = labels.to(device)

                outputs = model(reference_image, comparison_image).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
            + f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    training_directory = Path.cwd().joinpath("workdir", "train")
    initial_training_dataset = load_dataset(training_directory)
    randomized_training_dataset = generate_randomized_dataset(initial_training_dataset)
    training_dataset = FingerprintDataset(randomized_training_dataset)

    train_size = int(0.8 * len(training_dataset))
    val_size = len(training_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        training_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = FPRNet()
    train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
