import torch
import torch.nn as nn


class CBMModel(nn.Module):
    def __init__(self, concept_predictor, label_predictor):
        """
        Args:
            concept_predictor: Das trainierte Modell aus Task 2
            label_predictor: Das neue Modell für Task 3
        """
        super(CBMModel, self).__init__()
        self.concept_predictor = concept_predictor
        self.label_predictor = label_predictor

    def forward(self, x):
        # 1. Bild -> Konzept Logits (Stage 1)
        concept_logits = self.concept_predictor(x)

        # 2. Logits -> Wahrscheinlichkeiten (Bottleneck)
        # Wir nutzen Sigmoid, da wir binäre Konzepte haben (Ja/Nein)
        concept_probs = torch.sigmoid(concept_logits)

        # 3. Wahrscheinlichkeiten -> Label Logits (Stage 2)
        label_logits = self.label_predictor(concept_probs)

        # Wir geben beides zurück, um später sowohl Konzepte als auch Labels
        # auswerten zu können (Wichtig für den Report!)
        return concept_logits, label_logits
    