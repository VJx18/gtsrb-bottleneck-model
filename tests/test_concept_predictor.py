import pytest
import torch
import torch.nn as nn
from src.models.concept_predictor import ConceptPredictor   # Pfad anpassen

@pytest.fixture
def concept_model():
    return ConceptPredictor(num_concepts=15, dropout=0.3)

@pytest.fixture
def sample_input():
    return torch.randn(4, 3, 32, 32)

def test_concept_predictor_output_shape(concept_model, sample_input):
    out = concept_model(sample_input)
    assert out.shape == (4, 15), f"Erwartet (batch, 15), bekam {out.shape}"

def test_output_dtype(concept_model, sample_input):
    out = concept_model(sample_input)
    assert out.dtype == torch.float32

def test_device_preservation(concept_model, sample_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = concept_model.to(device)
    inp = sample_input.to(device)
    out = model(inp)
    assert out.device == inp.device

def test_dropout_is_active_in_train_mode(concept_model, sample_input):
    concept_model.train()
    
    # Mehrere Forward-Pässe → Dropout sollte Varianz erzeugen
    outs = [concept_model(sample_input) for _ in range(3)]
    outs = torch.stack(outs)                    # shape: (3, bs, 15)
    
    # Varianz über Dropout-Dimension sollte > 0 sein (bei p=0.3)
    var = outs.var(dim=0).mean()
    assert var > 1e-4, "Dropout scheint nicht aktiv zu sein (zu wenig Varianz)"

def test_dropout_is_inactive_in_eval_mode(concept_model, sample_input):
    concept_model.eval()
    
    with torch.no_grad():
        out1 = concept_model(sample_input)
        out2 = concept_model(sample_input)
    
    # In eval sollte das Ergebnis deterministisch sein
    assert torch.allclose(out1, out2, atol=1e-6)

