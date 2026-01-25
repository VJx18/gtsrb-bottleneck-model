import json
import torch
import os
import pandas as pd
import numpy as np
from src.config.config import Config
from src.models.concept_predictor import ConceptPredictor
from src.models.cbm_model import CBMModel
from src.models.label_predictor import LabelPredictor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_cbm_model(config=Config(), num_examples=8, test_loader=None):

    if test_loader is None:
        raise TypeError("Dataloader is of Type None")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")#MPS for local Mac testing
    print(f"Using device: {device}")

    output_dir = "./experiments/checkpoints"
    checkpoint_path = config.training.checkpoint_dir
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_concepts = config.model.num_concepts

    # load CBM Model
    concept_predictor = ConceptPredictor(num_concepts=num_concepts, dropout=config.model.dropout)
    label_predictor = LabelPredictor(num_concepts=num_concepts, num_classes=config.dataset.num_classes, dropout=config.model.dropout)
    model = CBMModel(concept_predictor, label_predictor)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()
    all_concept_preds = []
    all_concept_targets = []
    all_label_preds = []
    all_label_targets = []
    example_data = [] # für die Visualisierung

    print("\n Starting CBM Model Evaluation...\n")

    with torch.no_grad():
        for batch_indx, (images, (concept_vectors, labels)) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # predict concepts and labels
            concept_logits, label_logits = model(images)

            # get predicted concepts and true concepts
            concept_preds = (torch.sigmoid(concept_logits) > 0.5).cpu()
            all_concept_preds.append(concept_preds)
            all_concept_targets.append(concept_vectors.cpu())

            # get predicted labels and true labels
            label_preds = (torch.softmax(label_logits, dim=1)).cpu().argmax(dim=1)
            all_label_preds.append(label_preds)
            all_label_targets.append(labels)

            # shows progress (how many batches are processed)
            if (batch_indx + 1) % 20 == 0:
                    print(f"Batch {batch_indx+1}/{len(test_loader)} processed.")

            # collect example data for visualization
            if len(example_data) < num_examples:
                for i in range(images.size(0)):
                    if len(example_data) >= num_examples:
                        break
                    example_data.append({
                        "image": images[i].cpu(),
                        "true_concepts": concept_vectors[i].cpu().tolist(),
                        "predicted_concepts": concept_preds[i].tolist(),
                        "true_label": int(labels[i].item()),
                        "predicted_label": int(label_preds[i].item())
                    })
    
    print("\n Evaluation completed. Computing and plotting metrics...\n")

    all_concept_preds   = torch.cat(all_concept_preds, axis=0)
    all_concept_targets = torch.cat(all_concept_targets, axis=0)
    all_label_preds     = torch.cat(all_label_preds, axis=0)
    all_label_targets   = torch.cat(all_label_targets, axis=0)

    # read concept csv file for concept and label names
    concept_df = pd.read_csv(config.dataset.concept_csv)

    # concept metrics
    concept_acc = accuracy_score(all_concept_targets, all_concept_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_concept_targets, all_concept_preds, average=None, zero_division=0)

    # get concept names
    concept_names = concept_df.columns.tolist()[2:]  # skip first two columns (class_id and class_name)
    
    concept_metrics = {
        "overall_accuracy": float(concept_acc),
        "per concept": [
            {
            "concept_name": concept_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i])
            } for i in range(len(concept_names))
        ]
    }

    # label metrics
    label_acc = accuracy_score(all_label_targets, all_label_preds)

    # get names of traffic signs
    class_names = concept_df["class_name"].tolist()

    label_report = classification_report(all_label_targets, all_label_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_label_targets, all_label_preds)

    # plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Traffic Sign Classes")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # visualize example images
    for idx, ex in enumerate(example_data):
        fig = plt.figure(figsize=(14, 6))
        img = ex["image"].cpu().permute(1, 2, 0).numpy()

        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img)

        ax1.set_title(f"True: {ex['true_label']} | Pred: {ex['predicted_label']}")
        ax1.axis('off')
        
        # plotting true and predicted concepts
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')

        # get true and predicted concepts
        indices_true = [index for index, value in enumerate(ex['true_concepts']) if value == 1.0]
        indices_pred = [index for index, value in enumerate(ex['predicted_concepts']) if value == 1.0]
        all_indices = set(indices_true + indices_pred)
        selected_concepts = [concept_names[i] for i in all_indices]

        # create dataframe
        data = {
            "True Concepts": [ex['true_concepts'][i] for i in all_indices],
            "Predicted Concepts": [ex['predicted_concepts'][i] for i in all_indices]
        }
        df = pd.DataFrame(index=selected_concepts, data=data)
        df = df.astype(int)

        # Farben: blau bei 1, weiß bei 0
        cell_colors = [['#ADD8E6' if val == 1 else 'white' 
                        for val in row] 
                       for row in df.values]
        
        table = ax2.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='center',cellColours=cell_colors, colWidths=[0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.4, 1.8)  # Spaltenbreite und Zeilenhöhe vergrößern

        # Rahmen und Styling verbessern
        table.auto_set_column_width(col=list(range(len(df.columns))))

        ax2.set_title("True vs Predicted Concepts")
        
        plt.tight_layout()
        ex_path = os.path.join(output_dir, f"example_{idx+1}.png")
        plt.savefig(ex_path)
        plt.close()

        # ── Zusammenfassung ─────────────────────────────────────────
    results = {
        "label_accuracy": float(label_acc),
        "label_classification_report": label_report,
        "concept_metrics": concept_metrics,
        "num_test_samples": len(all_label_targets),
        "confusion_matrix_path": cm_path,
        "example_plots": [os.path.join(output_dir, f"example_{i+1}.png") for i in range(len(example_data))]
    }

    # JSON speichern
    with open(os.path.join(output_dir, "cbm_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate_cbm_model()