# src/core/bk_tree.py

import unicodedata

def normalize_string(text):
    """
    Normaliza una cadena de texto para búsqueda:
    - Elimina espacios al inicio/final.
    - Convierte a minúsculas.
    - Elimina acentos y caracteres diacríticos.
    - Elimina caracteres no alfanuméricos (excepto espacios que se manejan con strip).
    """
    if not isinstance(text, str):
        return "" # Manejar entradas no string o vacías

    text = text.strip().lower()
    # Normalizar a NFD y eliminar caracteres diacríticos (acentos, etc.)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Eliminar caracteres no alfanuméricos (manteniendo espacios para la distancia de Levenshtein)
    # Se debe ser cuidadoso con qué caracteres se eliminan aquí, ya que afecta la distancia de Levenshtein.
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip()


def levenshtein_distance(s1, s2):
    """
    Calcula la distancia de Levenshtein (distancia de edición) entre dos cadenas.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Inputs must be strings for Levenshtein distance calculation.")

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class BKTreeNode:
    """Clase para un nodo del BK-Tree."""
    def __init__(self, word):
        if not isinstance(word, str) or not word:
            raise ValueError("BKTreeNode word must be a non-empty string.")
        self.word = word # La palabra almacenada en el nodo (ya normalizada)
        self.children = {}  # Diccionario: distancia -> nodo hijo

class BKTree:
    """Clase para el Burkhard-Keller Tree (BK-Tree)."""
    def __init__(self):
        self.root = None

    def insert(self, word):
        """
        Inserta una palabra normalizada en el BK-Tree.
        La palabra debe ser normalizada antes de ser pasada a este método.
        """
        if not word: # No insertar cadenas vacías
            return

        if not self.root:
            self.root = BKTreeNode(word)
            return

        current = self.root
        while True:
            dist = levenshtein_distance(word, current.word)
            if dist == 0: # La palabra ya existe en el árbol
                return
            if dist in current.children:
                current = current.children[dist]
            else:
                current.children[dist] = BKTreeNode(word)
                break

    def search(self, query, tolerance):
        """
        Busca palabras en el BK-Tree que estén dentro de una distancia máxima de la palabra de consulta.
        La palabra de consulta debe ser normalizada antes de ser pasada a este método.
        Retorna una lista de palabras normalizadas que coinciden.
        """
        results = set() # Usar un conjunto para evitar duplicados, luego convertir a lista
        if not self.root:
            return list(results)

        def _recursive_search(node, query_norm, tolerance):
            if node is None:
                return

            dist = levenshtein_distance(query_norm, node.word)
            if dist <= tolerance:
                results.add(node.word) # Añadir la palabra normalizada del nodo

            # Poda: solo visita los hijos que podrían contener resultados
            for child_distance, child_node in node.children.items():
                if abs(dist - child_distance) <= tolerance:
                    _recursive_search(child_node, query_norm, tolerance) # Llama recursivamente

        _recursive_search(self.root, query, tolerance)
        return list(results) # Convertir a lista al final
