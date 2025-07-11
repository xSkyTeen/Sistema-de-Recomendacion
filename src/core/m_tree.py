# src/core/m_tree.py

import numpy as np
import pickle
from collections import deque
from src.core.distance_metrics import euclidean_distance  # O la métrica que elijas
from config import M_TREE_BRANCHING_FACTOR, M_TREE_MIN_CHILDREN


class MTreeNode:
    """
    Representa un nodo en el M-tree. Puede ser una hoja o un nodo interno.
    """

    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # Para nodos hoja: (item_id, vector_caracteristicas)
        # Para nodos internos: (routing_object_vector, routing_object_id, covering_radius, child_node)
        self.parent = None

    def add_entry(self, entry):
        self.entries.append(entry)

    def is_full(self):
        return len(self.entries) > M_TREE_BRANCHING_FACTOR


class MTree:
    """
    Implementación básica de un M-tree.
    """

    def __init__(self, distance_metric=euclidean_distance):
        self.root = MTreeNode(is_leaf=True)
        self.distance_metric = distance_metric
        self.item_map = {}  # Mapea item_id a vector_caracteristicas (para acceso rápido)

    def insert(self, item_id, features_vector):
        """
        Inserta un nuevo ítem en el M-tree.
        (Implementación simplificada, el M-tree real tiene lógica compleja de división y re-inserción)
        """
        self.item_map[item_id] = features_vector
        self._insert_recursive(self.root, item_id, features_vector)

    def _insert_recursive(self, node, item_id, features_vector):
        if node.is_leaf:
            node.add_entry((item_id, features_vector))
            if node.is_full():
                self._split_node(node)
        else:
            # Encontrar el mejor subárbol para insertar
            # Esto implica calcular distancias a los objetos de enrutamiento y radios de cobertura
            best_child = self._find_best_child(node, features_vector)
            self._insert_recursive(best_child, item_id, features_vector)
            # Actualizar radios de cobertura y objetos de enrutamiento en el camino de regreso
            self._adjust_path(node, features_vector)

    def _find_best_child(self, node, features_vector):
        """
        Encuentra el hijo más adecuado para la inserción.
        Basado en la mínima expansión del radio de cobertura.
        """
        min_expansion = float('inf')
        best_child = None
        for routing_obj_vec, _, covering_radius, child_node in node.entries:
            dist = self.distance_metric(features_vector, routing_obj_vec)
            # Si el nuevo punto está dentro del radio actual, la expansión es 0
            # De lo contrario, es la diferencia entre la nueva distancia y el radio actual
            expansion = max(0, dist - covering_radius)
            if expansion < min_expansion:
                min_expansion = expansion
                best_child = child_node
            elif expansion == min_expansion:
                # Criterio de desempate: elegir el que tenga el radio de cobertura más pequeño
                if covering_radius < best_child.entries[0][2]:  # Asumiendo que el radio está en la entrada
                    best_child = child_node
        return best_child

    def _split_node(self, node):
        """
        Divide un nodo lleno en dos.
        (Lógica compleja: selección de pivotes, distribución de entradas, creación de nuevo padre)
        """
        # Implementación de la división de nodos (ej. algoritmo de división cuadrática o lineal)
        # Esto es crucial para el equilibrio del árbol.
        pass  # Placeholder

    def _adjust_path(self, node, features_vector):
        """
        Ajusta los radios de cobertura de los objetos de enrutamiento en el camino hacia la raíz.
        """
        # Actualiza el radio de cobertura del objeto de enrutamiento que apunta a este nodo
        # para asegurar que contenga el nuevo punto.
        pass  # Placeholder

    def find_k_nearest(self, query_vector, k):
        """
        Encuentra los k vecinos más cercanos a un vector de consulta.
        Utiliza una búsqueda de prioridad (cola de prioridad) para explorar el árbol.
        """
        if not self.root.entries:
            return []

        # Cola de prioridad para la búsqueda (distancia, item_id)
        # Usamos heapq para mantener los k elementos más cercanos encontrados hasta ahora
        # La cola almacenará (-distancia, item_id) para un max-heap que simule un min-heap
        # de los k elementos más cercanos.
        priority_queue = []  # Almacena (distancia_a_query, nodo/entrada)

        # Resultado final: (distancia, item_id)
        results = []

        # Inicializar con la raíz
        # Para nodos internos, la distancia es la distancia mínima posible a la región del nodo
        # Para nodos hoja, la distancia es la distancia real al punto

        # Iniciar la búsqueda desde la raíz
        # La cola de prioridad almacenará tuplas (distancia_minima_posible_a_region, nodo)
        # O (distancia_real_a_item, item_id)

        # Para simplificar, una búsqueda BFS o DFS para empezar, luego optimizar con heap.

        # BFS simplificado para demostración
        q = deque([self.root])

        # Lista temporal para almacenar todos los resultados encontrados con sus distancias
        all_found_items = []

        while q:
            current_node = q.popleft()

            if current_node.is_leaf:
                for item_id, item_vector in current_node.entries:
                    dist = self.distance_metric(query_vector, item_vector)
                    all_found_items.append((dist, item_id))
            else:
                for routing_obj_vec, routing_obj_id, covering_radius, child_node in current_node.entries:
                    # Aplicar la desigualdad triangular para podar
                    # Calcular la distancia del query al objeto de enrutamiento
                    dist_query_to_routing = self.distance_metric(query_vector, routing_obj_vec)

                    # Si el radio de cobertura del hijo + la distancia del query al objeto de enrutamiento
                    # es menor que la distancia del k-ésimo vecino actual,
                    # o si el query está dentro del radio de cobertura,
                    # entonces vale la pena explorar este hijo.

                    # Simplificación: explorar todos los hijos para esta demostración
                    q.append(child_node)

        # Ordenar todos los ítems encontrados por distancia y tomar los k primeros
        all_found_items.sort(key=lambda x: x[0])
        results = all_found_items[:k]

        return results

    def save_index(self, filepath):
        """Guarda el M-tree en un archivo."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_index(filepath):
        """Carga el M-tree desde un archivo."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Nota: Una implementación completa del M-tree es compleja e incluye:
# - Algoritmos de selección de pivotes para la división de nodos.
# - Algoritmos de división de nodos (ej. división cuadrática, división lineal).
# - Lógica de re-inserción tras la división.
# - Manejo de radios de cobertura y distancias al objeto de enrutamiento.
# Para este proyecto, puedes usar una implementación simplificada o buscar una librería existente.