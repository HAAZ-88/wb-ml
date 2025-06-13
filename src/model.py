
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.losses import build_sample_weights

class WBLogisticModel:
    def __init__(self, lambda_norm=0.0, disadvantaged_groups=[1], C=1.0, max_iter=1000):
        """
        Modelo WB‑ML basado en regresión logística + sample weights.

        Args:
            lambda_norm: peso del término de bienestar
            disadvantaged_groups: lista de grupos g desfavorecidos
            C: parámetro de regularización de LogisticRegression
            max_iter: iteraciones máximas del optimizador
        """
        self.lambda_norm = lambda_norm
        self.disadvantaged_groups = disadvantaged_groups
        
        # Pipeline: escalado + clasificador
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter))
        ])
    
    def fit(self, X, y, s):
        """
        Ajusta el modelo con sample weights adecuados.

        Args:
            X: ndarray de características
            y: ndarray de etiquetas
            s: ndarray de variable sensible
        """
        sample_weight = build_sample_weights(
            y, s, self.lambda_norm, self.disadvantaged_groups
        )
        
        self.pipeline.fit(X, y, clf__sample_weight=sample_weight)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]  # probabilidad de clase positiva
