import torch
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from src.config.config import Config
from src.data.dataset import get_dataloaders
from src.models.concept_predictor import ConceptPredictor

class evaluation:

    @staticmethod
    def evaluate_concept(dataloader, model, num_labels):

        device = next(model.parameters()).device

        precision = MultilabelPrecision(num_labels=num_labels, average=None).to(device)
        recall = MultilabelRecall(num_labels=num_labels, average=None).to(device)
        f1 = MultilabelF1Score(num_labels=num_labels, average=None).to(device)
        
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for image, (concept_vector, label) in dataloader:
                image, concept_vector = image.to(device), concept_vector.to(device)
                logits = model(image)
                pred_concept = (torch.sigmoid(logits) > 0.5).float()

                precision.update(pred_concept, concept_vector)
                recall.update(pred_concept, concept_vector)
                f1.update(pred_concept, concept_vector)

                all_preds.append(pred_concept.cpu())
                all_targets.append(concept_vector.cpu())

        precision = precision.compute()
        recall = recall.compute()
        f1 = f1.compute()

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        accuracy = (all_preds == all_targets).float().mean(dim=0)

        return precision, recall, f1, accuracy


def evaluate_concept_predictor(checkpoint_path="./experiments/checkpoints/best_concept_model.pth"):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")#MPS for local Mac testing
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_concepts = checkpoint['fc2.weight'].shape[0]

    model = ConceptPredictor(num_concepts=num_concepts, dropout=config.model.dropout)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    print(f"\nEvaluating {num_concepts} concepts")
   

    precision, recall, f1, accuracy = evaluation.evaluate_concept(val_loader, model, num_concepts)
    print(f"\nOverall Accuracy: {accuracy.mean():.4f}")
    print(f"\nPer-Concept Accuracy:")
    
    for i, acc in enumerate(accuracy):
        print(f"Concept {i:2d}: {acc:.4f}")

    best_idx = accuracy.argmax().item()
    worst_idx = accuracy.argmin().item()
     # print all metrics
    print(f"Best Concept: {best_idx} (Accuracy: {accuracy[best_idx]:.4f})")
    print(f"Worst Concept: {worst_idx} (Accuracy: {accuracy[worst_idx]:.4f})")
    print(f"\nAverage Precision: {precision.mean():.4f}")
    print(f"Average Recall: {recall.mean():.4f}")
    print(f"Average F1 Score: {f1.mean():.4f}")
    return {
        'precision': precision.cpu().numpy(),
        'recall': recall.cpu().numpy(),
        'f1': f1.cpu().numpy(),
        'accuracy': accuracy.cpu().numpy()
    }

if __name__ == "__main__":
    evaluate_concept_predictor()