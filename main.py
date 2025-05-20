#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motor de búsqueda de películas por descripción (overview)
usando Sentence-Transformers y FAISS.
"""
import os
import pickle
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import faiss  # pip install faiss-cpu (o faiss-gpu)
from sentence_transformers import (
    SentenceTransformer,
)  # pip install sentence-transformers

# ─────────────────────────── Configuración ─────────────────────────────── #
DATA_FILE = "tmdb_movies.db"
DB_TABLE_NAME = "movies"
ID_COLUMN = "movie_id"  # PK de la tabla movies
TITLE_COLUMN = "title"  # Columna para mostrar el título
OVERVIEW_COLUMN = "overview"  # Columna con el texto para los embeddings

# Modelo de Sentence Transformers (puedes elegir otros)
# paraphrase-multilingual-MiniLM-L12-v2 es bueno y ligero.
# all-mpnet-base-v2 es muy potente para inglés.
# Si tus overviews están en varios idiomas, el multilingual es mejor.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Nombres de archivo para el motor serializado
ENGINE_PKL_FILE = "st_overview_engine.pkl"
METRICS_TXT_FILE = "st_overview_metrics.txt"

TOP_K_RESULTS = 5


# ─────────────────── Normalización de Texto (Simple) ───────────────────── #
def normalize_text_for_embedding(text: str) -> str:
    """Normalización básica para texto antes de pasarlo al Sentence Transformer."""
    if not text or not isinstance(text, str):
        return ""
    # Los Sentence Transformers suelen manejar bien mayúsculas/minúsculas,
    # pero convertir a minúsculas y strip es una práctica común y segura.
    # No se necesita eliminación agresiva de puntuación o stopwords.
    return text.strip().lower()


# ─────────────────── Construcción del Motor (1ª Vez) ──────────────────── #
def build_engine():
    print(
        f"⚙️ Construyendo motor de búsqueda con Sentence-Transformers (basado en '{OVERVIEW_COLUMN}')..."
    )
    t_start = time.perf_counter()

    movie_ids_list = []
    titles_for_display_list = []
    overviews_for_embedding_list = []

    try:
        with sqlite3.connect(DATA_FILE) as conn:
            conn.row_factory = sqlite3.Row  # Acceder a columnas por nombre
            cursor = conn.cursor()

            query = f"SELECT {ID_COLUMN}, {TITLE_COLUMN}, {OVERVIEW_COLUMN} FROM {DB_TABLE_NAME}"
            print(f"Ejecutando consulta: {query}")
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                print(
                    f"❌ No se encontraron filas en la tabla '{DB_TABLE_NAME}'. El motor no puede ser construido."
                )
                return

            for row in rows:
                movie_id = row[ID_COLUMN]
                title = (
                    str(row[TITLE_COLUMN])
                    if row[TITLE_COLUMN]
                    else "Título Desconocido"
                )
                overview = str(row[OVERVIEW_COLUMN]) if row[OVERVIEW_COLUMN] else ""

                # Solo incluir películas con un overview no vacío para el embedding
                if overview.strip():
                    movie_ids_list.append(movie_id)
                    titles_for_display_list.append(title)
                    overviews_for_embedding_list.append(
                        normalize_text_for_embedding(overview)
                    )
                # else:
                #     print(f"Advertencia: Película '{title}' (ID: {movie_id}) no tiene overview. Omitiendo del índice.")

    except sqlite3.Error as e:
        print(f"❌ Error de SQLite al leer datos: {e}")
        return

    if not overviews_for_embedding_list:
        print(
            "❌ No se encontraron overviews válidos para procesar. El motor no puede ser construido."
        )
        return

    print(
        f"Procesando {len(overviews_for_embedding_list)} overviews para embeddings..."
    )

    # Cargar modelo Sentence Transformer
    print(f"Cargando modelo: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Generar embeddings
    print("Generando embeddings para los overviews (esto puede tardar)...")
    embeddings = model.encode(
        overviews_for_embedding_list,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalizar a longitud unitaria es bueno para similitud coseno/IP
    ).astype(np.float32)

    # Crear índice FAISS
    print("Creando índice FAISS...")
    index_dimension = embeddings.shape[1]
    # IndexFlatIP es bueno para vectores normalizados (producto interno = similitud coseno)
    faiss_index = faiss.IndexFlatIP(index_dimension)
    faiss_index.add(embeddings)
    print(f"Índice FAISS creado con {faiss_index.ntotal} vectores.")

    # Guardar todo lo necesario
    engine_data = {
        "movie_ids": movie_ids_list,
        "titles_for_display": titles_for_display_list,
        "model_name": MODEL_NAME,
        # No guardamos los embeddings crudos si el índice FAISS es lo que usaremos,
        # pero guardamos el índice serializado.
        # Para IndexFlatIP simple, reconstruirlo es rápido. Guardaremos los embeddings.
        "embeddings_for_rebuild": embeddings,
    }

    with open(ENGINE_PKL_FILE, "wb") as f_out:
        pickle.dump(engine_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    dt_build = time.perf_counter() - t_start
    metrics_content = (
        f"Motor de Búsqueda por Overview (Sentence-Transformers)\n"
        f"Películas con overview indexadas: {len(titles_for_display_list)}\n"
        f"Modelo de embedding: {MODEL_NAME}\n"
        f"Dimensión del vector: {index_dimension}\n"
        f"Tiempo total de construcción: {dt_build:.2f} s\n"
        f"Tamaño del archivo {ENGINE_PKL_FILE}: {Path(ENGINE_PKL_FILE).stat().st_size / 1e6:.2f} MB\n"
    )
    Path(METRICS_TXT_FILE).write_text(metrics_content, encoding="utf-8")
    print(
        f"✅ Motor guardado en {ENGINE_PKL_FILE} ({dt_build:.1f}s). Métricas en {METRICS_TXT_FILE}"
    )


# ───────────────────── Carga del Motor ────────────────────── #
def load_engine():
    print(f"Cargando motor de búsqueda desde {ENGINE_PKL_FILE}...")
    if not os.path.exists(ENGINE_PKL_FILE):
        print(f"❌ Archivo del motor '{ENGINE_PKL_FILE}' no encontrado.")
        return None, None, None, None

    with open(ENGINE_PKL_FILE, "rb") as f_in:
        engine_data = pickle.load(f_in)

    model = SentenceTransformer(engine_data["model_name"])

    # Reconstruir índice FAISS desde los embeddings guardados
    embeddings_for_rebuild = engine_data["embeddings_for_rebuild"]
    index_dimension = embeddings_for_rebuild.shape[1]
    faiss_index = faiss.IndexFlatIP(index_dimension)
    faiss_index.add(embeddings_for_rebuild)

    print(f"Índice FAISS reconstruido con {faiss_index.ntotal} vectores.")

    return (
        model,
        faiss_index,
        engine_data["movie_ids"],
        engine_data["titles_for_display"],
    )


# ───────────────────── Búsqueda Semántica ─────────────────────── #
def search_by_description(
    user_description: str,
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    movie_ids_list: list,
    titles_for_display_list: list,
    k: int = TOP_K_RESULTS,
):
    if not user_description.strip():
        print("⚠️ La descripción está vacía.")
        return []

    normalized_description = normalize_text_for_embedding(user_description)

    # Generar embedding para la descripción del usuario
    query_embedding = model.encode(
        [normalized_description],  # model.encode espera una lista de textos
        convert_to_numpy=True,
        show_progress_bar=False,  # No es necesario para una sola consulta
        normalize_embeddings=True,
    ).astype(np.float32)

    # Buscar en FAISS
    # D son las distancias (o similitudes, para IndexFlatIP son productos internos)
    # I son los índices de los vectores más similares en la matriz original de embeddings
    distances, indices = faiss_index.search(query_embedding, k)

    results = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            idx_in_original_list = indices[0][i]
            if (
                idx_in_original_list < 0
            ):  # FAISS puede devolver -1 si no encuentra suficientes
                continue

            # La similitud es el producto interno (ya que los vectores están normalizados)
            # FAISS para IndexFlatIP devuelve el producto interno, que es la similitud coseno
            # Si se usara IndexFlatL2, distances sería la distancia L2 al cuadrado.
            similarity_score = distances[0][i]

            # Filtrar por umbral de similitud si se desea
            # if similarity_score < 0.3: # Ajusta este umbral
            #     continue

            results.append(
                {
                    "id": movie_ids_list[idx_in_original_list],
                    "title": titles_for_display_list[idx_in_original_list],
                    "similarity": float(similarity_score),
                }
            )

    return results


# ───────────────────── Loop Interactivo Principal ────────────────── #
def main_interactive_loop():
    if not os.path.exists(DATA_FILE):
        print(
            f"❌ Error Crítico: El archivo de base de datos '{DATA_FILE}' no se encuentra."
        )
        return

    model, faiss_index, movie_ids, titles_display = None, None, None, None

    if not os.path.exists(ENGINE_PKL_FILE):
        build_engine()  # Intentar construir si no existe

    # Intentar cargar después de construir o si ya existía
    load_attempt = load_engine()
    if load_attempt is None or any(item is None for item in load_attempt):
        print("❌ No se pudo cargar o construir el motor de búsqueda. Terminando.")
        return

    model, faiss_index, movie_ids, titles_display = load_attempt

    print(
        f"\n🎬 Motor Sentence-Transformer listo · {len(titles_display)} películas indexadas (por overview)."
    )
    print(f"   Modelo: {MODEL_NAME}")
    print(
        "   Escribe una breve descripción de una película (basada en su sinopsis/overview)"
    )
    print("   o escribe 'apagar sistema' para salir.")

    while True:
        user_query = input("\nDescripción: ").strip()
        if not user_query:
            continue
        if user_query.lower() == "apagar sistema":
            print("👋 Hasta pronto.")
            break

        search_results = search_by_description(
            user_query, model, faiss_index, movie_ids, titles_display
        )

        if search_results:
            print("\nResultados encontrados:")
            for r in search_results:
                print(
                    f" • {r['title']} (ID: {r['id']} · Similitud: {r['similarity']:.4f})"
                )
        else:
            print("No se encontraron películas similares para tu descripción.")


if __name__ == "__main__":
    main_interactive_loop()
