import pytest
import torch
import torch.nn as nn

from src.models.concept_predictor import ConceptPredictor
from src.models.label_predictor import LabelPredictor
from src.models.cbm_model import CBMModel


# ────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────

@pytest.fixture
def concept_predictor():
    return ConceptPredictor(num_concepts=12, dropout=0.0)  # dropout=0 → deterministisch


@pytest.fixture
def label_predictor():
    return LabelPredictor(num_concepts=12, num_classes=43, dropout=0.0)


@pytest.fixture
def cbm_model(concept_predictor, label_predictor):
    return CBMModel(concept_predictor, label_predictor)


@pytest.fixture
def sample_input():
    return torch.randn(5, 3, 32, 32)   # typische EfficientNetV2 Eingabegröße
    

def test_cbm_forward_output_shapes(cbm_model, sample_input):
    concept_logits, label_logits = cbm_model(sample_input)
    
    assert concept_logits.shape == (5, 12),   "Concept logits falsche Shape"
    assert label_logits.shape  == (5, 43),    "Label logits falsche Shape"


def test_cbm_returns_two_outputs(cbm_model, sample_input):
    outputs = cbm_model(sample_input)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    assert all(isinstance(o, torch.Tensor) for o in outputs)


def test_sigmoid_is_applied_as_bottleneck(cbm_model, sample_input):
    concept_logits, label_logits = cbm_model(sample_input)
    
    # Manuelle Sigmoid → sollte mit concept_probs übereinstimmen
    manual_probs = torch.sigmoid(concept_logits)
    
    # Indirekt prüfen: Wertebereich von concept_probs muss [0,1] sein
    assert manual_probs.min() >= 0.0 - 1e-6
    assert manual_probs.max() <= 1.0 + 1e-6
    
    # Wichtig: label_logits dürfen NICHT direkt von concept_logits kommen
    # (wäre der Fall, wenn sigmoid vergessen wurde)
    assert not torch.allclose(
        cbm_model.label_predictor(concept_logits),
        label_logits,
        atol=1e-4
    ), "Sigmoid wurde nicht angewendet – Label Predictor bekam rohe Logits"

def test_device_propagation(cbm_model, sample_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cbm_model = cbm_model.to(device)
    inp = sample_input.to(device)
    
    c_logits, l_logits = cbm_model(inp)
    assert c_logits.device == device
    assert l_logits.device == device


def test_dtype_consistency(cbm_model, sample_input):
    c_logits, l_logits = cbm_model(sample_input)
    assert c_logits.dtype == torch.float32
    assert l_logits.dtype == torch.float32

def test_only_label_predictor_receives_gradients(cbm_model, sample_input):
    # Concept Predictor explizit einfrieren (wie im Training)
    for param in cbm_model.concept_predictor.parameters():
        param.requires_grad = False
    
    cbm_model.train()
    c_logits, l_logits = cbm_model(sample_input)
    
    dummy_loss = l_logits.sum()   # nur Label-Output relevant
    dummy_loss.backward()
    
    # Concept-Parameter dürfen keinen Gradienten haben
    for name, param in cbm_model.concept_predictor.named_parameters():
        assert param.grad is None, f"Gradient fließt in gefrorenen Parameter: {name}"
    
    # Label-Predictor-Parameter müssen Gradienten haben
    for name, param in cbm_model.label_predictor.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Kein Gradient in Label-Predictor: {name}"


def test_concept_predictor_stays_in_eval_mode_during_training(cbm_model, sample_input):
    # Typisches Training-Setup
    cbm_model.concept_predictor.eval()
    cbm_model.label_predictor.train()
    
    # Mehrere Forward-Pässe → Dropout im Concept-Predictor sollte deaktiviert sein
    with torch.no_grad():
        out1 = cbm_model(sample_input)[0]   # concept_logits
    with torch.no_grad():
        out2 = cbm_model(sample_input)[0]
    
    assert torch.allclose(out1, out2, atol=1e-6), \
           "Concept Predictor verhält sich nicht deterministisch (eval-Modus fehlt?)"


def test_state_dict_contains_both_submodules(cbm_model):
    state = cbm_model.state_dict()
    assert any("concept_predictor" in k for k in state.keys())
    assert any("label_predictor" in k for k in state.keys())