import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

from config import config
from utils.dataset import SkinLesionDataset, get_transforms
from models.vit_model import create_model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Create model
        self.model = create_model(config.num_classes).to(self.device)
        print(f"Using device: {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Prepare data
        self.setup_data()

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def setup_data(self):
        print("Loading dataset...")

        # Load full dataset
        full_dataset = SkinLesionDataset(
            csv_file=self.config.csv_path,
            image_dir=self.config.image_dir,
            image_size=self.config.image_size,
            class_names=self.config.class_names
        )

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Apply transforms
        train_transform, val_transform = get_transforms(self.config.image_size)
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            if batch_idx % 10 == 0:
                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                current_acc = accuracy_score(all_labels, all_preds)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=list(self.config.class_names.keys())))

        return epoch_loss, epoch_acc

    def plot_training_history(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def train(self):
        best_val_acc = 0.0

        print("Starting training...")
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate_epoch()

            # Update scheduler
            self.scheduler.step()

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_model.pth")
                print(f"New best model saved with val_acc: {val_acc:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}.pth")

        # Plot training history
        self.plot_training_history()

        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

    def save_model(self, filename):
        os.makedirs("saved_models", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'history': self.history
        }, f"saved_models/{filename}")
        print(f"Model saved as saved_models/{filename}")


def main():
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()