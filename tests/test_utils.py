from src.utils.stopper import EarlyStopper

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    """Minimales Modell nur für Early-Stopping-Tests"""
    def __init__(self):
        super().__init__()
        # Nur ein Parameter, damit state_dict() funktioniert
        self.dummy_param = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return x

def test_early_stopping():
    es = EarlyStopper(patience=6, min_delta=0.001)
    dummy_model = DummyModel()

    losses = [0.5, 0.4, 0.35, 0.351, 0.34, 0.339, 0.398, 0.397, 0.396, 0.395, 0.394]
    expected_counters = [0, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6]
    should_stop = []

    for i, loss in enumerate(losses):
        stop = es.early_stop(loss, model=dummy_model)
        should_stop.append(stop)
        assert es.counter == expected_counters[i], f"expected counter: {expected_counters[i]} but got: {es.counter}, index: {i}"

    assert should_stop == [False, False, False, False, False, False, False, False, False, False, True], f"did not stop: {should_stop}"