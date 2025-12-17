import torch
import torch.nn as nn
import torch.optim as optim
from src.config.config import Config
from src.data.dataset import get_dataloaders
from src.models.concept_predictor import ConceptPredictor
from src.utils.stopper import EarlyStopper
import os
import json

def train_concept_predictor():
   
    config = Config()
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

    print(f"\nStage 1 training started: Concept Predictor ({config.training.epochs} epochs)")
    for epoch in range(config.training.epochs):
        model.train()
        running_loss = 0.0

        for i, (images, (concepts, labels)) in enumerate(train_loader):
            images, concepts = images.to(device), concepts.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, concepts)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Current Batch Loss: {loss.item():.4f}")

        #epoch summary
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        #validation stage
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, (concepts, labels) in val_loader:
                images, concepts = images.to(device), concepts.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, concepts)
                running_val_loss += val_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"epoch{epoch+1} done: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

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

    return model

if __name__ == "__main__":
    train_concept_predictor()