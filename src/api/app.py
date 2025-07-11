import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import re
import random
import io

# Importa el módulo de configuración desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

# Importa tus módulos personalizados desde src/core y src/data_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.m_tree import MTree
from src.core.bk_tree import BKTree, normalize_string, levenshtein_distance
from src.data_processing.data_loader import load_movielens_data, preprocess_data, \
    load_movie_titles  # Importar load_movie_titles

# Importaciones para procesamiento de datos y PCA
from sklearn.decomposition import PCA

# --- Configuración de la aplicación Flask ---
app = Flask(__name__,
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))
CORS(app)  # Habilita CORS para todas las rutas

# --- Variables Globales para Modelos y Datos Cargados ---
# Usar rutas definidas en config.py
MODELS_DIR = config.MODELS_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR  # Para cargar los CSVs

# Nombres de archivos de config
MOVIES_CSV_NAME = config.MOVIES_CSV
RATINGS_CSV_NAME = config.RATINGS_CSV
M_TREE_INDEX_FILE = config.M_TREE_INDEX_FILE
BK_TREE_INDEX_FILE = config.BK_TREE_INDEX_FILE
MOVIE_METADATA_FILE = 'movie_metadata.pkl'  # Nombre del archivo para guardar los metadatos
ITEM_EMBEDDINGS_FILE = 'item_embeddings.pkl'  # Nombre del archivo para guardar los embeddings del M-Tree
ALL_GENRES_FILE = 'all_genres.pkl'  # Nombre del archivo para guardar la lista de géneros
NORMALIZED_TITLE_MAP_FILE = 'normalized_title_to_id.pkl'  # Mapeo de título normalizado a ID
NORMALIZED_TITLE_TO_ORIGINAL_FILE = 'normalized_title_to_original.pkl'  # Mapeo de título normalizado a título original

# Archivos para el mapa de visualización (basado en embeddings de género)
MOVIE_COORDINATES_2D_FILE = 'movie_coordinates_2d_genre.pkl'  # Nuevo nombre para evitar conflicto
PCA_MODEL_FILE = 'pca_model_genre.pkl'  # Nuevo nombre para evitar conflicto

# Parámetros de recomendación
DEFAULT_K_RECOMMENDATIONS = config.DEFAULT_K_RECOMMENDATIONS

# Instancias de los modelos (inicializadas a None)
m_tree_instance = None
bk_tree_instance = None
movie_metadata = {}
item_embeddings = {}  # Embeddings basados en géneros para el M-Tree
all_genres_list = []
normalized_title_to_id = {}  # Mapeo de título normalizado a movie_id
normalized_title_to_original_title = {}  # Mapeo de título normalizado a título original

# Variables para el mapa de visualización (basadas en embeddings de género)
movie_coordinates_2d = {}  # Coordenadas 2D para la visualización del mapa
pca_model = None


# --- Funciones de Utilidad ---
# load_movielens_data, preprocess_data, parse_year_from_title se asumen importadas o definidas
# aquí si no se importan desde data_loader.py. Para esta versión, se importan.

def euclidean_distance(v1, v2):
    """Calcula la distancia euclidiana entre dos arrays de numpy (vectores)."""
    if v1 is None or v2 is None:
        raise ValueError("Los vectores no pueden ser None para el cálculo de la distancia euclidiana.")
    return np.linalg.norm(v1 - v2)


def load_or_build_movie_m_tree():
    """
    Carga todas las instancias de modelos (M-Tree, BK-Tree, PCA para visualización)
    desde el disco o las construye y guarda si no se encuentran o hay un error.
    """
    global m_tree_instance, bk_tree_instance, movie_metadata, item_embeddings, all_genres_list
    global normalized_title_to_id, normalized_title_to_original_title
    global movie_coordinates_2d, pca_model

    # Asegura que el directorio de modelos exista
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Rutas completas a los archivos de modelos
    m_tree_path = os.path.join(MODELS_DIR, M_TREE_INDEX_FILE)
    bk_tree_path = os.path.join(MODELS_DIR, BK_TREE_INDEX_FILE)
    movie_metadata_path = os.path.join(MODELS_DIR, MOVIE_METADATA_FILE)
    item_embeddings_path = os.path.join(MODELS_DIR, ITEM_EMBEDDINGS_FILE)
    all_genres_path = os.path.join(MODELS_DIR, ALL_GENRES_FILE)
    normalized_title_map_path = os.path.join(MODELS_DIR, NORMALIZED_TITLE_MAP_FILE)
    normalized_title_to_original_path = os.path.join(MODELS_DIR, NORMALIZED_TITLE_TO_ORIGINAL_FILE)

    # Rutas para componentes de visualización
    coordinates_path = os.path.join(MODELS_DIR, MOVIE_COORDINATES_2D_FILE)
    pca_path = os.path.join(MODELS_DIR, PCA_MODEL_FILE)

    # Lista de todos los archivos de modelos que deben existir para una carga completa
    all_model_files = [
        m_tree_path, bk_tree_path, movie_metadata_path, item_embeddings_path,
        all_genres_path, normalized_title_map_path, normalized_title_to_original_path,
        coordinates_path, pca_path
    ]

    # Comprobar si todos los archivos existen para intentar cargar
    all_files_exist = all(os.path.exists(p) for p in all_model_files)

    if all_files_exist:
        try:
            print("Intentando cargar todos los modelos y datos existentes...")
            with open(m_tree_path, 'rb') as f:
                m_tree_instance = pickle.load(f)
            with open(bk_tree_path, 'rb') as f:
                bk_tree_instance = pickle.load(f)
            with open(movie_metadata_path, 'rb') as f:
                movie_metadata = pickle.load(f)
            with open(item_embeddings_path, 'rb') as f:
                item_embeddings = pickle.load(f)
            with open(all_genres_path, 'rb') as f:
                all_genres_list = pickle.load(f)
            with open(normalized_title_map_path, 'rb') as f:
                normalized_title_to_id = pickle.load(f)
            with open(normalized_title_to_original_path, 'rb') as f:
                normalized_title_to_original_title = pickle.load(f)
            with open(coordinates_path, 'rb') as f:
                movie_coordinates_2d = pickle.load(f)
            with open(pca_path, 'rb') as f:
                pca_model = pickle.load(f)

            print("Todos los modelos y datos cargados exitosamente.")
            return  # Salir si todos los datos se cargaron

        except Exception as e:
            print(f"Error al cargar modelos/datos existentes: {e}. Forzando la reconstrucción de todos los modelos...")
            # Limpiar variables para forzar la reconstrucción
            m_tree_instance = None;
            bk_tree_instance = None;
            movie_metadata = {}
            item_embeddings = {};
            all_genres_list = []
            normalized_title_to_id = {};
            normalized_title_to_original_title = {}
            movie_coordinates_2d = {};
            pca_model = None

    # --- Reconstruir Modelos si no se cargaron o si ocurrió un error ---
    print("Construyendo nuevas instancias de M-Tree, BK-Tree y PCA para visualización...")

    # 1. Cargar y preprocesar los datos de películas
    movies_df, ratings_df = load_movielens_data()
    if movies_df is None:
        print("No se pudieron cargar los datos de películas. El sistema de películas no estará disponible.")
        return

    item_embeddings, all_genres_list = preprocess_data(movies_df, ratings_df)

    # parse_year_from_title se define aquí o se asume importada si es una función auxiliar
    def parse_year_from_title_local(title):
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None

    movie_metadata = {
        row['movieId']: {'title': row['title'], 'genres': row['genres'],
                         'year': parse_year_from_title_local(row['title'])}
        for index, row in movies_df.iterrows()
    }

    # 2. Construcción del M-Tree (para recomendaciones basadas en géneros)
    m_tree_instance = MTree(distance_metric=euclidean_distance)
    print("Insertando elementos en el M-Tree...")
    for movie_id, embedding in item_embeddings.items():
        m_tree_instance.insert(movie_id, embedding)
    print("M-Tree construido.")

    # 3. Construcción del BK-Tree (para búsqueda difusa de títulos / distancia de edición)
    bk_tree_instance = BKTree()
    print("Insertando títulos normalizados en el BK-Tree...")
    # Usar la lógica de construcción del BK-Tree del ejemplo funcional
    for _, row in movies_df.iterrows():  # Usar movies_df original
        original_title = row['title']
        movie_id = row['movieId']

        normalized_title = normalize_string(original_title)  # Usar la función normalize_string importada

        if normalized_title and normalized_title not in normalized_title_to_id:  # Evitar duplicados
            bk_tree_instance.insert(normalized_title)
            normalized_title_to_id[normalized_title] = movie_id
            normalized_title_to_original_title[normalized_title] = original_title
    print("BK-Tree construido.")

    # 4. Generar Coordenadas 2D para el Mapa D3.js (basadas en embeddings de género)
    print("Generando coordenadas 2D para el mapa D3.js (basadas en géneros)...")
    embeddings_for_pca = []
    movie_ids_for_pca = []
    for movie_id, embedding in item_embeddings.items():  # Usar item_embeddings (géneros)
        embeddings_for_pca.append(embedding)
        movie_ids_for_pca.append(movie_id)

    if embeddings_for_pca:
        embeddings_np = np.array(embeddings_for_pca)
        n_components = min(2, embeddings_np.shape[1])
        if n_components < 2:
            print(
                "Advertencia: No hay suficientes dimensiones para PCA 2D con embeddings de género. Omitiendo la generación de coordenadas 2D.")
            movie_coordinates_2d = {}
            pca_model = None
        else:
            pca_model = PCA(n_components=n_components)
            reduced_embeddings = pca_model.fit_transform(embeddings_np)
            for i, movie_id in enumerate(movie_ids_for_pca):
                movie_coordinates_2d[movie_id] = reduced_embeddings[i].tolist()  # Guardar como lista
            print("Coordenadas 2D generadas.")
    else:
        print("Omitiendo la generación de coordenadas 2D para el mapa.")
        movie_coordinates_2d = {}
        pca_model = None

    # 5. Guardar todos los modelos y datos construidos
    try:
        with open(m_tree_path, 'wb') as f:
            pickle.dump(m_tree_instance, f)
        with open(bk_tree_path, 'wb') as f:
            pickle.dump(bk_tree_instance, f)
        with open(movie_metadata_path, 'wb') as f:
            pickle.dump(movie_metadata, f)
        with open(item_embeddings_path, 'wb') as f:
            pickle.dump(item_embeddings, f)
        with open(all_genres_path, 'wb') as f:
            pickle.dump(all_genres_list, f)
        with open(normalized_title_map_path, 'wb') as f:
            pickle.dump(normalized_title_to_id, f)
        with open(normalized_title_to_original_path, 'wb') as f:
            pickle.dump(normalized_title_to_original_title, f)

        # Guardar componentes de visualización (PCA)
        if pca_model is not None:
            with open(coordinates_path, 'wb') as f: pickle.dump(movie_coordinates_2d, f)
            with open(pca_path, 'wb') as f: pickle.dump(pca_model, f)

        print("Todos los modelos y datos guardados exitosamente.")
    except Exception as e:
        print(f"Error al guardar modelos/datos: {e}")


# --- Rutas de Flask ---

@app.route('/')
def index():
    """Sirve el archivo HTML principal de la aplicación."""
    return render_template('index.html')


@app.route('/movies', methods=['GET'])
def get_movies():
    """Devuelve la lista completa de películas con sus metadatos."""
    if not movie_metadata:
        return jsonify({"error": "Datos de películas no cargados."}), 500

    movies_list = []
    for movie_id, data in movie_metadata.items():
        movies_list.append({
            "id": movie_id,
            "title": data['title'],
            "genres": data['genres'],
            "year": data['year']
        })
    movies_list.sort(key=lambda x: x['title'].lower())

    return jsonify({"movies": movies_list, "available_genres": all_genres_list})


@app.route('/recommend', methods=['GET'])
def recommend_movies():
    """
    Endpoint para obtener recomendaciones de películas basadas en un ID de película de referencia.
    Utiliza el M-Tree y embeddings basados en géneros.
    """
    item_id = request.args.get('item_id', type=int)
    if not item_id:
        return jsonify({"error": "Por favor, proporciona un 'item_id'."}), 400

    if m_tree_instance is None:
        return jsonify({"error": "El M-Tree no está inicializado. Asegúrate de que los datos estén cargados."}), 500

    if item_id not in movie_metadata:
        return jsonify({"error": f"Película con ID {item_id} no encontrada."}), 404

    query_embedding = item_embeddings.get(item_id)
    if query_embedding is None:
        return jsonify({"error": f"Embedding para la película ID {item_id} no encontrado."}), 500

    try:
        neighbors = m_tree_instance.find_k_nearest(query_embedding, DEFAULT_K_RECOMMENDATIONS + 1)
    except Exception as e:
        print(f"Error durante la búsqueda k-NN del M-Tree: {e}")
        return jsonify({"error": "Error durante la búsqueda de recomendaciones."}), 500

    recommendations = []
    for dist, rec_id in neighbors:
        if rec_id == item_id:  # Saltar la película de consulta
            continue
        rec_info = movie_metadata.get(rec_id)
        if rec_info:
            recommendations.append({
                "id": rec_id,
                "title": rec_info['title'],
                "genres": rec_info['genres'],
                "similarity_score": 1 / (1 + dist)  # Una simple inversión para la similitud (mayor es mejor)
            })
        if len(recommendations) >= DEFAULT_K_RECOMMENDATIONS:
            break

    query_item_info = movie_metadata.get(item_id)

    return jsonify({
        "query_item": {
            "id": item_id,
            "title": query_item_info['title'],
            "genres": query_item_info['genres']
        },
        "recommendations": recommendations
    })


@app.route('/search_by_title', methods=['GET'])
def search_by_title():
    """
    Endpoint para búsqueda por título utilizando solo BK-Tree (distancia de edición).
    Parámetros:
        query (str): La cadena de búsqueda del título.
        max_distance (int, opcional): Distancia máxima de Levenshtein para BK-Tree. Por defecto 2.
    """
    if bk_tree_instance is None:
        return jsonify({"error": "El sistema BK-Tree no está inicializado. Por favor, reinicia el servidor."}), 500

    query_string = request.args.get('query', type=str)
    max_dist = request.args.get('max_distance', type=int, default=2)  # Usar 2 como default, como en tu código original

    if not query_string:
        return jsonify({"error": "El parámetro 'query' es requerido."}), 400

    print(f"Búsqueda por título (solo BK-Tree) para '{query_string}' (Distancia Levenshtein máxima: {max_dist})...")

    normalized_query = normalize_string(query_string)

    # Perform the search in the BK-Tree
    bk_results_normalized_titles = bk_tree_instance.search(normalized_query, max_dist)

    # Format the results for the response
    formatted_results = []
    for normalized_title_from_bk in bk_results_normalized_titles:
        # Retrieve original title and ID using the global mappings
        movie_id = normalized_title_to_id.get(normalized_title_from_bk)
        original_title = normalized_title_to_original_title.get(normalized_title_from_bk)

        if movie_id and original_title:
            # Calculate distance using the original query (normalized) and the found normalized title.
            distance_to_show = levenshtein_distance(normalized_query, normalized_title_from_bk)
            genres = movie_metadata.get(movie_id, {}).get('genres', '')  # Obtener géneros para mostrar
            year = movie_metadata.get(movie_id, {}).get('year', None)  # Obtener año

            formatted_results.append({
                "title": original_title,  # Display the original title
                "id": movie_id,
                "distance": distance_to_show,
                "genres": genres,  # Incluir géneros
                "year": year  # Incluir año
            })

    # Sort by Levenshtein distance
    formatted_results.sort(key=lambda x: x['distance'])

    return jsonify({
        "query": query_string,
        "max_distance": max_dist,
        "results": formatted_results
    })


@app.route('/plot_recommendations', methods=['POST'])
def plot_recommendations():
    """
    Generates a 2D scatter plot of the query and recommended movies
    in a PCA-reduced space.
    """
    data = request.json
    query_item_id = data.get('query_item_id')
    recommended_item_ids = data.get('recommended_item_ids', [])

    if not query_item_id or not recommended_item_ids:
        return jsonify({"error": "Missing 'query_item_id' or 'recommended_item_ids'."}), 400

    current_embeddings = item_embeddings
    current_metadata = movie_metadata

    plot_embeddings = []
    plot_labels = []
    plot_colors = []
    plot_sizes = []

    # Add the query movie
    if query_item_id in current_embeddings:
        plot_embeddings.append(current_embeddings[query_item_id])
        plot_labels.append(current_metadata[query_item_id]['title'])
        plot_colors.append('red')  # Red for the query item
        plot_sizes.append(150)  # Larger for the query item
    else:
        return jsonify({"error": f"Query movie with ID {query_item_id} not found for plotting."}), 404

    # Add recommended movies
    for rec_id in recommended_item_ids:
        if rec_id in current_embeddings:
            plot_embeddings.append(current_embeddings[rec_id])
            plot_labels.append(current_metadata[rec_id]['title'])
            plot_colors.append('blue')  # Blue for recommended movies
            plot_sizes.append(100)  # Standard size for recommended
        else:
            print(f"Warning: Recommended movie with ID {rec_id} not found for plotting.")

    # Add a sample of other movies (rest of the database)
    other_item_ids = [mid for mid in current_embeddings.keys() if
                      mid != query_item_id and mid not in recommended_item_ids]
    sample_size = min(200, len(other_item_ids))  # Sample up to 200 additional movies
    sampled_other_ids = random.sample(other_item_ids, sample_size)

    for other_id in sampled_other_ids:
        plot_embeddings.append(current_embeddings[other_id])
        plot_labels.append(current_metadata[other_id]['title'])
        plot_colors.append('lightgray')  # Light gray for other movies
        plot_sizes.append(50)  # Smaller for other movies

    if not plot_embeddings:
        return jsonify({"error": "No valid items to plot."}), 400

    plot_embeddings_np = np.array(plot_embeddings)

    # Apply PCA to reduce to 2D
    n_features = plot_embeddings_np.shape[1]
    n_samples = plot_embeddings_np.shape[0]

    if n_samples < 2 or n_features < 2:
        return jsonify({"error": "Not enough samples or dimensions to create a meaningful 2D plot."}), 400

    n_components = min(2, n_features)

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(plot_embeddings_np)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                         c=plot_colors, s=plot_sizes, alpha=0.8)

    for i, txt in enumerate(plot_labels):
        if i == 0 or (i > 0 and i <= len(recommended_item_ids) and i <= 5):  # Annotate query and top 5 recommendations
            ax.annotate(txt,
                        (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        ha='left',
                        fontsize=9,
                        color='black' if plot_colors[i] == 'blue' else 'red',
                        fontweight='bold' if plot_colors[i] == 'red' else 'normal')

    ax.set_title(f"Alcance de las Recomendaciones de Películas (Reducción PCA a 2D)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Componente Principal 1", fontsize=12)
    ax.set_ylabel("Componente Principal 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Película de Consulta', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Películas Recomendadas', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Otras Películas', markerfacecolor='lightgray', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='svg', bbox_inches='tight')
    plt.close(fig)

    img_bytes.seek(0)
    svg_data = img_bytes.getvalue().decode('utf-8')

    return jsonify({"svg": svg_data})


@app.route('/movie_map_data', methods=['GET'])
def get_movie_map_data():
    """
    Endpoint para obtener los datos de películas (ID, título, géneros, año, coordenadas 2D)
    para la visualización interactiva del mapa de similitud.
    Ahora incluye los géneros para colorear los puntos.
    """
    if not movie_coordinates_2d or not pca_model:
        return jsonify({
                           "error": "Datos del mapa de películas no disponibles. Los embeddings o PCA podrían no haberse generado."}), 500

    map_data = []
    for movie_id, coords in movie_coordinates_2d.items():
        if movie_id in movie_metadata:
            movie_info = movie_metadata[movie_id]
            # Unir los géneros en una cadena para facilitar el mapeo de colores en D3.js
            genres_list = movie_info['genres'].split('|')
            map_data.append({
                "id": movie_id,
                "title": movie_info['title'],
                "genres": genres_list,  # Devolver como lista para D3.js
                "year": movie_info['year'],
                "x": float(coords[0]),
                "y": float(coords[1])
            })
    return jsonify(map_data)


# --- Inicialización de la Aplicación Flask ---
@app.before_request
def before_first_request():
    """
    Esta función se ejecuta una vez cuando llega la primera solicitud al servidor.
    Es un buen lugar para cargar o construir modelos pesados.
    """
    # Solo cargar/construir si las instancias de los modelos no están ya inicializadas
    if not m_tree_instance or not bk_tree_instance:
        load_or_build_movie_m_tree()


if __name__ == '__main__':
    app.run(host=config.API_HOST, port=config.API_PORT, debug=True)
