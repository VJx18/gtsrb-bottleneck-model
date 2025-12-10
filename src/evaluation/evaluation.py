import torch
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

class evaluation:

    def evaluate_concept(dataloader, model, num_labels):

        device = next(model.parameters()).device

        precision = MultilabelPrecision(num_labels=num_labels, average=None).to(device)
        recall = MultilabelRecall(num_labels=num_labels, average=None).to(device)
        f1 = MultilabelF1Score(num_labels=num_labels, average=None).to(device)
        
        # predict all concepts
        model.eval()
        with torch.no_grad():
            for batchindx, (image, (concept_vector, label)) in enumerate(dataloader):
                image, concept_vector = image.to(device), concept_vector.to(device)
                logits = model(image)
                print(batchindx)

                # set all values over 0.5 to 1 and all others to 0
                pred_concept = (torch.sigmoid(logits)>0.5).float()

                precision.update(pred_concept, concept_vector)
                recall.update(pred_concept, concept_vector)
                f1.update(pred_concept, concept_vector)

        precision = precision.compute()
        recall = recall.compute()
        f1 = f1.compute()

        # print all metrics
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        return precision, recall, f1