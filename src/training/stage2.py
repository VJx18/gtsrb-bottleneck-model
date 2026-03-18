import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
from src.config.config import Config
from src.data.dataset import get_dataloaders
from src.models.concept_predictor import ConceptPredictor
from src.models.label_predictor import LabelPredictor
from src.models.cbm_model import CBMModel
from src.utils.stopper import EarlyStopper


def train_label_predictor(config=Config()):
    
    # config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Stage 2 on: {device}")

    # 1. Daten laden
    train_loader, val_loader, test_loader = get_dataloaders(config)
    num_concepts = train_loader.dataset.num_concepts
    num_classes = config.dataset.num_classes

    # 2. Stage 1 Modell (Concept Predictor) laden
    print("Loading pretrained Concept Predictor...")
    concept_model = ConceptPredictor(num_concepts=num_concepts, dropout=config.model.dropout)

    checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_concept_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Please run Task 2 training first.")

    concept_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # WICHTIG: Concept Predictor einfrieren!
    # Wir wollen nicht, dass er sich verändert, die Konzepte sollen stabil bleiben.
    for param in concept_model.parameters():
        param.requires_grad = False

    concept_model.eval()  # Setzt Dropout/BatchNorm auf Eval Modus

    # 3. Stage 2 Modell (Label Predictor) initialisieren
    label_model = LabelPredictor(num_concepts=num_concepts, num_classes=num_classes, dropout=config.model.dropout)

    # 4. Kombiniertes CBM Modell erstellen
    cbm_model = CBMModel(concept_model, label_model).to(device)

    # 5. Training Setup
    # Wir optimieren NUR die Parameter des Label Predictors
    optimizer = optim.Adam(cbm_model.label_predictor.parameters(), lr=config.training.lr)

    # calculate weights
    label_counts = torch.bincount(torch.tensor([label for _, (_ ,label) in train_loader.dataset]))
    weights = 1.0 / (label_counts + 1e-8)
    weights = weights.to(device)

    # CrossEntropyLoss für Multi-Class Classification (Verkehrsschilder)
    criterion = nn.CrossEntropyLoss(weight=weights)

    early_stopper = EarlyStopper(patience=config.training.patience)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print(f"\nStage 2 Training started: Label Predictor ({config.training.epochs} epochs)")

    for epoch in range(config.training.epochs):
        cbm_model.train()  # Label Predictor train mode, Concept Predictor bleibt gefroren
        # (Hinweis: cbm_model.train() setzt alle submodule auf train.
        # Da wir requires_grad=False gesetzt haben, werden CP Gewichte trotzdem nicht geupdated,
        # aber Dropout im CP wäre aktiv. Oft will man CP im Eval mode lassen.
        # Im strikten CBM Sinne lassen wir CP komplett im eval mode:)
        cbm_model.concept_predictor.eval()
        

        running_loss = 0.0
        correct_labels = 0
        total_labels = 0

        for i, (images, (concept_targets, labels)) in enumerate(train_loader):
            cbm_model.label_predictor.train()
            images, labels = images.to(device), labels.to(device)
            # concept_targets brauchen wir hier nicht zwingend für den Loss,
            # da wir labels vorhersagen wollen.

            optimizer.zero_grad()

            # Forward Pass durch das ganze CBM
            _, label_logits = cbm_model(images)

            # Loss berechnen (Vorhergesagte Labels vs. Echte Labels)
            loss = criterion(label_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # calculate train accuracy (but only for every 50th Batch to save time)
            if ((i + 1) % 50 == 0):
                with torch.no_grad():
                    # deactivates dropout -> important for correct Train Acc!
                    cbm_model.label_predictor.eval()
                    _, predicted = torch.max(label_logits.data, 1)
                    total_labels += labels.size(0)
                    correct_labels += (predicted == labels).sum().item()
                    print(f"Batch {i+1} / {len(train_loader)} | Train acc: {correct_labels / total_labels:.4f}")

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {i + 1} | Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_labels / total_labels
        train_losses.append(epoch_train_loss)
        train_accuracies.append(train_acc)

        # --- Validation ---
        cbm_model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, (concept_targets, labels) in val_loader:
                images, labels = images.to(device), labels.to(device)

                _, label_logits = cbm_model(images)
                val_loss = criterion(label_logits, labels)

                running_val_loss += val_loss.item()

                _, predicted = torch.max(label_logits.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(
            f"Epoch {epoch + 1} Result: Train Loss={epoch_train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss={epoch_val_loss:.4f} (Acc: {epoch_val_acc:.2f}%)")

        if early_stopper.early_stop(epoch_val_loss, cbm_model.label_predictor):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Speichern
    print("\nTraining finished. Saving best CBM model...")
    # Wir laden den besten Zustand des Label Predictors zurück
    early_stopper.load_best_model(cbm_model.label_predictor)

    save_path = os.path.join(config.training.checkpoint_dir, "best_cbm_model.pth")
    # Wir speichern das ganze CBM Model (inklusive CP weights)
    torch.save(cbm_model.state_dict(), save_path)
    print(f"Full CBM model saved to: {save_path}")

    # Save History
    with open(os.path.join(config.training.checkpoint_dir, "history_cbm.json"), 'w') as f:
        json.dump({'train_loss': train_losses, 'val_loss': val_losses, 'train_acc': train_accuracies, 'val_acc': val_accuracies}, f)

    # plot train and validation loss curves
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss Curves - Label Predictor")

    ax2.plot(train_accuracies, label="Training Accuracy")
    ax2.plot(val_accuracies, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy Curves - Label Predictor")

    plt.tight_layout()
    loss_curve_path = os.path.join(config.training.checkpoint_dir, "loss_curves_label.png")
    plt.savefig(loss_curve_path)
    plt.close()

if __name__ == "__main__":
    train_label_predictor()
