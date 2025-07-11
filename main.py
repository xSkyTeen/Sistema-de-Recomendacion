# main.py

from src.api.app import app, load_or_build_movie_m_tree
from config import API_HOST, API_PORT

if __name__ == '__main__':
    print("Iniciando Sistema de Recomendación...")
    # Cargar o construir el M-tree antes de iniciar el servidor Flask
    load_or_build_movie_m_tree()
    print(f"Servidor Flask ejecutándose en http://{API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=True)