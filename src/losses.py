import numpy as np

def build_sample_weights(y, s, lambda_norm, disadvantaged_groups=[1]):
    """
    Construye el vector sample_weight para optimizar 𝓛_CE + λ_norm * 𝓛_WB.

    Args:
        y: ndarray de etiquetas reales (shape (N,))
        s: ndarray de variable sensible (shape (N,))
        lambda_norm: float, valor de λ_norm
        disadvantaged_groups: lista de grupos g considerados desfavorecidos

    Returns:
        sample_weight: ndarray de pesos (shape (N,))
    """
    sample_weight = np.ones_like(y, dtype=float)
    
    # Identifica ejemplos a ponderar
    mask = np.isin(s, disadvantaged_groups) & (y == 1)
    
    # Aplica el incremento ponderado
    sample_weight[mask] += lambda_norm
    
    return sample_weight
