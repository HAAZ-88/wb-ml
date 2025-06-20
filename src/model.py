import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.losses import build_sample_weights

class WBLogisticModel:
    """
    Regresión logística con ponderación de bienestar.

    Args
    ----
    lambda_norm : float
        Peso del término de bienestar (λ_norm).
    disadvantaged_groups : list[int]
        IDs de los grupos desfavorecidos.
    C : float
        Parámetro de regularización L2 de LogisticRegression.
    max_iter : int
        Iteraciones del optimizador.
    """
    def __init__(self, lambda_norm=0.0, disadvantaged_groups=[1],
                 C=1.0, max_iter=1000):
        self.lambda_norm = lambda_norm
        self.disadvantaged_groups = disadvantaged_groups
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter,
                                       solver="lbfgs"))
        ])

    def fit(self, X, y, s):
        """Entrena con sample_weight basado en λ_norm."""
        w = build_sample_weights(y, s,
                                 lambda_norm=self.lambda_norm,
                                 disadvantaged_groups=self.disadvantaged_groups)
        self.pipeline.fit(X, y, clf__sample_weight=w)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        # probabilidad de clase positiva
        return self.pipeline.predict_proba(X)[:, 1]

