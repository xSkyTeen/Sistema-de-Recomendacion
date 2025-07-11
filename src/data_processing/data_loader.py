# src/data_processing/data_loader.py

import pandas as pd
import numpy as np
import os
import pickle
import re  # Importar re para la función de limpieza de títulos de fallback
from src.core.bk_tree import normalize_string  # Importar la función de normalización

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, \
    MOVIES_CSV, RATINGS_CSV, EMBEDDINGS_FILE, MOVIES_CLEANED_CSV


def load_movielens_data():
    """
    Carga los datasets de MovieLens.
    Descarga los archivos movies.csv y ratings.csv de MovieLens (ej. ml-latest-small)
    y colócalos en la carpeta data/raw/.
    """
    movies_path = os.path.join(RAW_DATA_DIR, MOVIES_CSV)
    ratings_path = os.path.join(RAW_DATA_DIR, RATINGS_CSV)

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print(f"Error: Archivos de MovieLens no encontrados en {RAW_DATA_DIR}.")
        print(
            "Por favor, descarga 'ml-latest-small.zip' de https://grouplens.org/datasets/movielens/ y extrae 'movies.csv' y 'ratings.csv' en la carpeta data/raw/.")
        return None, None

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    print(f"Cargadas {len(movies_df)} películas y {len(ratings_df)} calificaciones.")
    return movies_df, ratings_df


def preprocess_data(movies_df, ratings_df=None):
    """
    Preprocesa los datos de películas (ej. combinar, limpiar).
    Para este ejemplo, nos centraremos en los géneros para crear características simples.
    """
    # Intentar cargar embeddings existentes si ya fueron procesados
    embeddings_path = os.path.join(PROCESSED_DATA_DIR, EMBEDDINGS_FILE)
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, 'rb') as f:
                genre_vectors = pickle.load(f)
            # Reconstruir all_genres_list si solo se cargan los embeddings
            all_genres = set()
            for genres_str in movies_df['genres']:
                for genre in genres_str.split('|'):
                    if genre != '(no genres listed)':
                        all_genres.add(genre)
            all_genres = sorted(list(all_genres))
            print(f"Cargados vectores de género existentes para {len(genre_vectors)} películas.")
            return genre_vectors, all_genres
        except Exception as e:
            print(f"Error al cargar embeddings existentes: {e}. Reprocesando datos de películas.")

    # Si no existen o hay error al cargar, procesar desde cero
    all_genres = set()
    for genres_str in movies_df['genres']:
        for genre in genres_str.split('|'):
            if genre != '(no genres listed)':
                all_genres.add(genre)
    all_genres = sorted(list(all_genres))
    print(f"Géneros únicos encontrados: {len(all_genres)}")

    genre_vectors = {}
    for index, row in movies_df.iterrows():
        movie_id = row['movieId']
        genres_list = row['genres'].split('|')

        vector = np.zeros(len(all_genres))
        for i, genre in enumerate(all_genres):
            if genre in genres_list:
                vector[i] = 1
        genre_vectors[movie_id] = vector

    print(f"Generados vectores de género para {len(genre_vectors)} películas.")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    with open(embeddings_path, 'wb') as f:
        pickle.dump(genre_vectors, f)
    print(f"Vectores de características de películas guardados en {embeddings_path}")

    return genre_vectors, all_genres


def load_movie_titles():
    """
    Extrae y retorna una lista de todos los títulos de películas limpios y normalizados
    desde el archivo movies_cleaned.csv.
    """
    cleaned_movies_path = os.path.join(RAW_DATA_DIR, MOVIES_CLEANED_CSV)

    if not os.path.exists(cleaned_movies_path):
        print(
            f"Advertencia: {MOVIES_CLEANED_CSV} no encontrado. Asegúrate de ejecutar scripts/clean_movie_titles.py primero.")
        # Fallback: intenta cargar el original movies.csv y limpiar/normalizar en memoria
        movies_df, _ = load_movielens_data()
        if movies_df is not None:
            # Esta lógica de limpieza debe coincidir con la de clean_movie_titles.py
            def clean_title_fallback(title):
                cleaned_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
                if cleaned_title.endswith(', The'):
                    cleaned_title = 'The ' + cleaned_title[:-5]
                elif cleaned_title.endswith(', A'):
                    cleaned_title = 'A ' + cleaned_title[:-3]
                elif cleaned_title.endswith(', An'):
                    cleaned_title = 'An ' + cleaned_title[:-4]
                return cleaned_title.strip()

            # Aplicar la limpieza y luego la normalización, filtrando vacíos
            cleaned_titles = movies_df['title'].apply(clean_title_fallback).tolist()
            normalized_titles = [normalize_string(t) for t in cleaned_titles if
                                 normalize_string(t)]  # Filter out empty strings
            return normalized_titles
        return []

    cleaned_movies_df = pd.read_csv(cleaned_movies_path)
    # Normalizar los títulos al cargarlos para el BK-Tree, filtrando vacíos
    normalized_titles = [normalize_string(title) for title in cleaned_movies_df['title'].tolist() if
                         normalize_string(title)]  # Filter out empty strings
    return normalized_titles


if __name__ == '__main__':
    # Este bloque se ejecuta si corres data_loader.py directamente
    movies_df, ratings_df = load_movielens_data()
    if movies_df is not None and ratings_df is not None:
        movie_vectors, all_genres = preprocess_data(movies_df, ratings_df)
        print("\nPreprocesamiento de datos de películas completado.")

        movie_titles_normalized = load_movie_titles()
        print(f"Cargados {len(movie_titles_normalized)} títulos de películas limpios y normalizados.")