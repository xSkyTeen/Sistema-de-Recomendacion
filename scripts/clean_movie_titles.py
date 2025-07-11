# scripts/clean_movie_titles.py

import pandas as pd
import re
import os

# Asegúrate de que la ruta a config.py sea correcta
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import RAW_DATA_DIR, MOVIES_CSV, MOVIES_CLEANED_CSV


def clean_title(title):
    """
    Limpia un título de película eliminando el año entre paréntesis
    y cualquier espacio o puntuación extra al final.
    Ej: "Toy Story (1995)" -> "Toy Story"
    Ej: "American President, The (1995)" -> "American President, The"
    """
    # Eliminar el año entre paréntesis al final del título
    cleaned_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

    # Manejar el caso de "American President, The" -> "The American President"
    # Esto es un patrón común en MovieLens
    if cleaned_title.endswith(', The'):
        cleaned_title = 'The ' + cleaned_title[:-5]
    elif cleaned_title.endswith(', A'):
        cleaned_title = 'A ' + cleaned_title[:-3]
    elif cleaned_title.endswith(', An'):
        cleaned_title = 'An ' + cleaned_title[:-4]

    return cleaned_title.strip()  # Asegurar que no queden espacios al inicio/final


def create_cleaned_movies_csv():
    """
    Lee movies.csv, limpia los títulos y guarda un nuevo CSV.
    """
    movies_path = os.path.join(RAW_DATA_DIR, MOVIES_CSV)
    cleaned_movies_path = os.path.join(RAW_DATA_DIR, MOVIES_CLEANED_CSV)

    if not os.path.exists(movies_path):
        print(f"Error: {MOVIES_CSV} no encontrado en {RAW_DATA_DIR}.")
        print("Asegúrate de que tus archivos de MovieLens estén en la carpeta correcta.")
        return

    print(f"Cargando {MOVIES_CSV} desde {movies_path}...")
    movies_df = pd.read_csv(movies_path)

    print("Limpiando títulos de películas...")
    # Aplicar la función de limpieza a la columna 'title'
    movies_df['cleaned_title'] = movies_df['title'].apply(clean_title)

    # Crear un nuevo DataFrame con movieId y el título limpio
    # Mantener el título original también si es necesario para otros fines,
    # pero para el BK-Tree usaremos 'cleaned_title'.
    cleaned_df = movies_df[['movieId', 'cleaned_title']].copy()
    cleaned_df.rename(columns={'cleaned_title': 'title'}, inplace=True)  # Renombrar para consistencia

    print(f"Guardando títulos limpios en {cleaned_movies_path}...")
    cleaned_df.to_csv(cleaned_movies_path, index=False)
    print("Proceso de limpieza de títulos completado.")
    print(f"Primeras 5 entradas del nuevo CSV:")
    print(cleaned_df.head())


if __name__ == '__main__':
    create_cleaned_movies_csv()
