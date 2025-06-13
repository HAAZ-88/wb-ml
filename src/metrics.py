import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def tpr_gap(y_true, y_pred, s):
    ref = (y_pred[y_true==1]==1).mean()
    gaps=[]
    for g in np.unique(s):
        mask=(s==g)&(y_true==1)
        tpr_g = (y_pred[mask]==1).mean() if mask.any() else 0
        gaps.append(abs(tpr_g-ref))
    return max(gaps)

def dp_gap(y_pred, s):
    ref = (y_pred==1).mean()
    gaps=[]
    for g in np.unique(s):
        mask = s==g
        dp = (y_pred[mask]==1).mean()
        gaps.append(abs(dp-ref))
    return max(gaps)
