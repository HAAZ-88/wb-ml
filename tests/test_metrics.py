from src.metrics import tpr_gap, dp_gap
import numpy as np

def test_gaps():
    y_true = np.array([1,1,0,0])
    y_pred = np.array([1,0,0,1])
    s = np.array([0,0,1,1])
    assert tpr_gap(y_true, y_pred, s) >= 0
    assert dp_gap(y_pred, s) >= 0
