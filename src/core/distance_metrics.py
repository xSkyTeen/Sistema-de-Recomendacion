# src/core/distance_metrics.py

import numpy as np
from scipy.spatial.distance import cosine, euclidean

def euclidean_distance(vec1, vec2):
    """
    Calcula la distancia euclidiana entre dos vectores.
    Args:
        vec1 (np.array): Primer vector.
        vec2 (np.array): Segundo vector.
    Returns:
        float: Distancia euclidiana.
    """
    return euclidean(vec1, vec2)

def cosine_similarity(vec1, vec2):
    """
    Calcula la similitud del coseno entre dos vectores.
    Args:
        vec1 (np.array): Primer vector.
        vec2 (np.array): Segundo vector.
    Returns:
        float: Similitud del coseno (valor entre -1 y 1).
    """
    return 1 - cosine(vec1, vec2) # 1 - coseno para obtener un valor de similitud (0 a 1)

# Puedes añadir más métricas según necesites (ej. Manhattan, Jaccard)