import pytest
import torch
from src.models.label_predictor import LabelPredictor

@pytest.fixture
def label_model():
    return LabelPredictor(num_concepts=15, num_classes=43, dropout=0.3)

@pytest.fixture
def sample_concepts():
    return torch.randn(8, 15)   # batch=8, 15 Konzepte

def test_label_predictor_output_shape(label_model, sample_concepts):
    out = label_model(sample_concepts)
    assert out.shape == (8, 43), f"Erwartet (batch, 43), bekam {out.shape}"

def test_label_predictor_dtype(label_model, sample_concepts):
    out = label_model(sample_concepts)
    assert out.dtype == torch.float32

def test_dropout_behavior_in_train(label_model, sample_concepts):
    label_model.train()
    outs = torch.stack([label_model(sample_concepts) for _ in range(3)])
    var = outs.var(dim=0).mean()
    assert var > 1e-5, "Dropout scheint in train() nicht zu wirken"

def test_dropout_disabled_in_eval(label_model, sample_concepts):
    label_model.eval()
    with torch.no_grad():
        out1 = label_model(sample_concepts)
        out2 = label_model(sample_concepts)
    assert torch.equal(out1, out2) 

def test_parameter_count_sensible(label_model):
    total_params = sum(p.numel() for p in label_model.parameters() if p.requires_grad)
    # Grobe Schätzung: ~ 15→256 + 256→128 + 128→43 ≈ 45k–50k Parameter
    assert 40000 < total_params < 70000, f"Parameter-Anzahl unerwartet: {total_params}"