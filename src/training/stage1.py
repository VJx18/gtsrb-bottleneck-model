import torch
import torch.nn as nn
import torch.optim as optim
from src.config.config import Config
from src.data.dataset import get_dataloaders
from src.models.concept_predictor import ConceptPredictor
from src.utils.stopper import EarlyStopper
import os
import json
import matplotlib.pyplot as plt

def train_concept_predictor(config=Config()):
   
    # config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")
    print("Loading GTSRB Data")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Instead of just batch count, show the actual number of images
    print(f"Dataset Loaded: {len(train_loader.dataset)} training images")
    print(f"Dataset Loaded: {len(val_loader.dataset)} validation images")

    # initialize model: Stage 1
    # Finding out num_concepts from the dataset logic 
    num_concepts = train_loader.dataset.num_concepts
    print(f"Model will predict {num_concepts} individual concepts")

    model = ConceptPredictor(num_concepts=num_concepts, dropout=config.model.dropout)
    model = model.to(device)

    # training setup 
    # We use BCEWithLogitsLoss because concepts are binary
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    
    # Using EarlyStopper
    early_stopper = EarlyStopper(patience=config.training.patience)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nStage 1 training started: Concept Predictor ({config.training.epochs} epochs)")
    for epoch in range(config.training.epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for i, (images, (concepts, labels)) in enumerate(train_loader):
            images, concepts = images.to(device), concepts.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, concepts)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation for concepts per batch
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_train += concepts.numel()
            correct_train += (preds == concepts).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Current Batch Loss: {loss.item():.4f}")

        #epoch summary
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train # Hamming Accuracy

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        #validation stage
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, (concepts, labels) in val_loader:
                images, concepts = images.to(device), concepts.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, concepts)
                running_val_loss += val_loss.item()

                # Accuracy calculation
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_val += concepts.numel()
                correct_val += (preds == concepts).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = correct_val / total_val # Hamming Accuracy

        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"epoch {epoch+1} done: Train Loss = {epoch_train_loss:.4f} (Train Acc: {epoch_train_acc:.2f}%), Val Loss = {epoch_val_loss:.4f} (Val Acc: {epoch_val_acc:.2f}%)")

        #early stopping
        if early_stopper.early_stop(epoch_val_loss, model):
            print(f"Early stopping at epoch {epoch+1} to prevent overfitting")
            break

    #saving weights
    print("\nTraining finished.Loading best model state found")
    early_stopper.load_best_model(model)

    if not os.path.exists(config.training.checkpoint_dir):
        os.makedirs(config.training.checkpoint_dir)
        
    save_path = os.path.join(config.training.checkpoint_dir, "best_concept_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Stage 1 weights saved to: {save_path}")

    # Save history for the final report plots
    with open(os.path.join(config.training.checkpoint_dir, "history.json"), 'w') as f:
        json.dump({'train': train_losses, 'val': val_losses}, f)

    # plot train and validation loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss Curves - Concept Predictor")

    ax2.plot(train_accs, label="Training Accuracy")
    ax2.plot(val_accs, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy Curves - Concept Predictor")

    plt.tight_layout()
    loss_curve_path = os.path.join(config.training.checkpoint_dir, "loss_curves_concept.png")
    plt.savefig(loss_curve_path)
    plt.close()

    return model

if __name__ == "__main__":
    train_concept_predictor()