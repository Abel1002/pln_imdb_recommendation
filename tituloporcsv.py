# buscador perfecto solo por titulo

# main.py
import os
import pickle
import re
import difflib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# configuración
# --------------------------------------------------------------------------- #
DATA_FILE = "movies.csv"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PKL_FILE = "model.pkl"  # motor serializado
METRICS_TXT = "metrics.txt"  # métricas de la 1.ª generación
TOP_K = 5  # nº de resultados a mostrar


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def normaliza(t: str) -> str:
    """Baja a minúsculas y quita el año entre paréntesis."""
    t = re.sub(r"\(\d{4}\)$", "", t).strip().lower()
    return t


def fuzzy_suggestion(query_norm, titles_norm) -> str:
    """Devuelve un título muy parecido en texto o la query original."""
    matches = difflib.get_close_matches(query_norm, titles_norm, n=1, cutoff=0.8)
    return matches[0] if matches else query_norm


# --------------------------------------------------------------------------- #
# 1ª vez: construir y guardar el motor
# --------------------------------------------------------------------------- #
def build_engine() -> None:
    print("⚙️  Construyendo motor de búsqueda… (solo ocurrirá una vez)")
    t0 = time.perf_counter()

    # 1) leer CSV
    df = pd.read_csv(DATA_FILE)
    titles_orig = df["title"].tolist()
    titles_norm = [normaliza(t) for t in titles_orig]

    # 2) modelo y embeddings
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        titles_norm, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    # 3) índice FAISS (Inner-Product exacto)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # 4) serializar todo (embeddings ≈ 5 MB, índice ≈ igual)
    with open(PKL_FILE, "wb") as f:
        pickle.dump(
            dict(
                titles_orig=titles_orig,
                titles_norm=titles_norm,
                embeddings=emb,  # guardamos emb → exactitud idéntica
                model_name=MODEL_NAME,
            ),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # métricas informativas
    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas: {len(titles_orig)}\n"
        f"Dim vector: {emb.shape[1]}\n"
        f"Tiempo total: {dt:.2f} s\n"
        f"Tamaño model.pkl: {Path(PKL_FILE).stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8",
    )
    print(f"✅ Motor guardado en {PKL_FILE} ({dt:.1f}s).")


# --------------------------------------------------------------------------- #
# carga rápida
# --------------------------------------------------------------------------- #
def load_engine():
    data = pickle.load(open(PKL_FILE, "rb"))
    index = faiss.IndexFlatIP(data["embeddings"].shape[1])
    index.add(data["embeddings"])  # reconstruir índice en RAM
    model = SentenceTransformer(data["model_name"])  # se carga desde caché local
    return index, model, data["titles_orig"], data["titles_norm"]


# --------------------------------------------------------------------------- #
# búsqueda
# --------------------------------------------------------------------------- #
# --- Búsqueda ---
def buscar(query, index, model, titles_orig, titles_clean, k=5):
    """
    Realiza una búsqueda semántica de películas.
    Intenta corregir typos antes de la búsqueda semántica.
    """
    # Preprocesar la consulta
    q = normaliza(query)
    original_q_normalized = q  # Guardamos la consulta normalizada original

    # Sugerir corrección de typo si hay un título muy similar textualmente
    # Usamos los títulos normalizados para la comparación textual.
    # Bajamos el cutoff para ser más tolerantes con múltiples typos.
    # Nota: Esto puede aumentar sugerencias incorrectas para queries que NO son títulos.
    # Experimenta con este valor (ej: 0.6, 0.7, 0.75)
    suggestion = difflib.get_close_matches(
        q, titles_clean, n=1, cutoff=0.7
    )  # <-- CAMBIO AQUÍ

    if suggestion:
        corrected_query_normalized = suggestion[0]
        # Solo usamos la sugerencia si es diferente de la consulta normalizada original
        if corrected_query_normalized != original_q_normalized:
            # Encontrar el título original correspondiente a la sugerencia normalizada para mostrar al usuario
            try:
                original_suggested_title = titles_orig[
                    titles_clean.index(corrected_query_normalized)
                ]
                print(
                    f"(💡 ¿Quizás quisiste decir: '{original_suggested_title}'? Buscando '{corrected_query_normalized}')"
                )
            except ValueError:
                # Esto no debería pasar si la sugerencia viene de titles_clean, pero por seguridad
                print(
                    f"(💡 ¿Quizás quisiste decir: '{corrected_query_normalized}'? Buscando '{corrected_query_normalized}')"
                )

            q = corrected_query_normalized  # Usamos la sugerencia corregida para la búsqueda semántica
        # Else: Si la sugerencia es la misma que la consulta original (normalizada),
        # significa que la consulta ya es un título válido o muy similar a uno.
        # En este caso, proceed with the original (normalized) query.
        else:
            print(
                "(✅ La consulta parece correcta o es muy similar a un título existente. Realizando búsqueda semántica.)"
            )

    else:
        # No close textual match found (even with lower cutoff), proceed with semantic search on the original normalized query.
        print(
            "(🔎 No se encontró una corrección textual cercana. Realizando búsqueda semántica con la consulta original.)"
        )

    # Obtener embedding de la consulta usando el modelo cargado
    # Aseguramos que la query_vec sea float32 para FAISS
    query_vec = model.encode(
        [q], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    # Buscar los k más similares en el índice FAISS
    distances, idx = index.search(query_vec, k)

    # Retornar títulos originales correspondientes a los índices encontrados
    # idx[0] contiene los índices para la primera (y única) query vector
    results = [titles_orig[i] for i in idx[0]]
    return results


# --- El resto del código (normaliza, build_engine, load_engine, main) permanece igual ---
# Asegúrate de reemplazar SOLO la función 'buscar' en tu archivo main.py con esta versión.


# --------------------------------------------------------------------------- #
# programa interactivo
# --------------------------------------------------------------------------- #
def main():
    if not os.path.exists(PKL_FILE):
        build_engine()

    index, model, titles_orig, titles_norm = load_engine()
    print("🎬 Motor listo. Escribe un título o ‘apagar sistema’ para salir.")

    while True:
        q = input("\n¿Película que buscas?: ").strip()
        if not q:
            continue
        if q.lower() == "apagar sistema":
            print("👋 Hasta pronto.")
            break

        for r in buscar(q, index, model, titles_orig, titles_norm):
            print(" •", r)


if __name__ == "__main__":
    main()
