import os
import pickle
import re
import difflib
from pathlib import Path
import time
import numpy as np

# import pandas as pd # Ya no es necesario para leer el archivo de datos principal
import sqlite3  # <--- AÑADIDO para SQLite
import faiss
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# configuración
# --------------------------------------------------------------------------- #
DATA_FILE = "tmdb_movies.db"  # <--- CAMBIADO a .db
# Nuevas constantes para la base de datos
DB_TABLE_NAME = "movies"  # Nombre de tu tabla según la imagen
DB_TITLE_COLUMN = "title"  # Nombre de la columna de títulos según la imagen

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PKL_FILE = "model_db_title_only.pkl"  # Nuevo nombre para el pkl para evitar sobrescribir el anterior
METRICS_TXT = "metrics_db_title_only.txt"  # Nuevo nombre para las métricas
TOP_K = 5  # nº de resultados a mostrar


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def normaliza(t: str) -> str:
    """Baja a minúsculas y quita el año entre paréntesis al final."""
    # Aseguramos que t sea string antes de cualquier operación
    text_str = str(t)
    # Usamos r"\s*\(\d{4}\)$" para coincidir con (YYYY) al final, con o sin espacio antes.
    # Si prefieres la regex original de tu script CSV era r"\(\d{4}\)$"
    text_str = re.sub(r"\s*\(\d{4}\)$", "", text_str).strip().lower()
    return text_str


def fuzzy_suggestion(query_norm, titles_norm) -> str:
    """Devuelve un título muy parecido en texto o la query original."""
    matches = difflib.get_close_matches(query_norm, titles_norm, n=1, cutoff=0.8)
    return matches[0] if matches else query_norm


# --------------------------------------------------------------------------- #
# 1ª vez: construir y guardar el motor
# --------------------------------------------------------------------------- #
def build_engine() -> None:
    print(
        "⚙️  Construyendo motor de búsqueda (desde DB, solo título)… (solo ocurrirá una vez)"
    )
    t0 = time.perf_counter()

    titles_orig_list = []
    try:
        conn = sqlite3.connect(DATA_FILE)
        cursor = conn.cursor()

        # 1) leer títulos de la base de datos SQLite
        query = f"SELECT {DB_TITLE_COLUMN} FROM {DB_TABLE_NAME}"
        print(f"Ejecutando consulta: {query} en {DATA_FILE}")
        cursor.execute(query)

        # Extraer títulos, asegurándose de que no sean None
        # y convirtiéndolos a string por si acaso.
        titles_orig_list = [
            str(row[0]) for row in cursor.fetchall() if row[0] is not None
        ]
        conn.close()

        if not titles_orig_list:
            print(
                f"❌ No se encontraron títulos en la columna '{DB_TITLE_COLUMN}' de la tabla '{DB_TABLE_NAME}'."
            )
            print("Verifica la configuración DB_TABLE_NAME y DB_TITLE_COLUMN.")
            return  # Salir si no hay títulos

    except sqlite3.Error as e:
        print(f"❌ Error al leer la base de datos SQLite: {e}")
        print(
            f"Asegúrate de que el archivo '{DATA_FILE}' existe y que la tabla '{DB_TABLE_NAME}' con la columna '{DB_TITLE_COLUMN}' es correcta."
        )
        if "conn" in locals() and conn:  # Intenta cerrar la conexión si estaba abierta
            conn.close()
        return  # Salir en caso de error

    print(f"Cargados {len(titles_orig_list)} títulos de la base de datos.")
    titles_norm_list = [normaliza(t) for t in titles_orig_list]

    # 2) modelo y embeddings
    print(f"Cargando modelo SentenceTransformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Generando embeddings para los títulos normalizados...")
    emb = model.encode(
        titles_norm_list,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # 3) índice FAISS (Inner-Product exacto)
    print("Creando índice FAISS...")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # 4) serializar todo
    print(f"Guardando motor en {PKL_FILE}...")
    with open(PKL_FILE, "wb") as f:
        pickle.dump(
            dict(
                titles_orig=titles_orig_list,  # Guardamos la lista original de títulos
                titles_norm=titles_norm_list,  # Guardamos la lista normalizada de títulos
                embeddings=emb,
                model_name=MODEL_NAME,
            ),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # métricas informativas
    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas (desde DB): {len(titles_orig_list)}\n"
        f"Dim vector: {emb.shape[1]}\n"
        f"Tiempo total: {dt:.2f} s\n"
        f"Tamaño {PKL_FILE}: {Path(PKL_FILE).stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8",
    )
    print(f"✅ Motor guardado en {PKL_FILE} ({dt:.1f}s).")


# --------------------------------------------------------------------------- #
# carga rápida
# --------------------------------------------------------------------------- #
def load_engine():
    # Esta función no cambia, ya que carga el .pkl preprocesado.
    print(f"Cargando motor desde {PKL_FILE}...")
    data = pickle.load(open(PKL_FILE, "rb"))

    embeddings_array = data["embeddings"]
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)

    model = SentenceTransformer(data["model_name"])

    # Devuelve los nombres de variables como en tu script original
    return index, model, data["titles_orig"], data["titles_norm"]


# --------------------------------------------------------------------------- #
# búsqueda
# --------------------------------------------------------------------------- #
# La función buscar que me proporcionaste en el último código es robusta.
# La usaré aquí, ya que es la que tenías en tu "buscador perfecto solo por titulo"
# con la corrección de 'cutoff' que mencionaste.
def buscar(
    query, index, model, titles_orig_arg, titles_clean_arg, k=TOP_K
):  # Usar TOP_K de config
    """
    Realiza una búsqueda semántica de películas.
    Intenta corregir typos antes de la búsqueda semántica.
    """
    q_normalized_query = normaliza(query)  # Normaliza la consulta del usuario
    original_q_normalized_for_suggestion_check = q_normalized_query

    suggestion_list = difflib.get_close_matches(
        q_normalized_query,
        titles_clean_arg,
        n=1,
        cutoff=0.7,  # cutoff como en tu script
    )

    if suggestion_list:
        corrected_query_normalized = suggestion_list[0]
        if corrected_query_normalized != original_q_normalized_for_suggestion_check:
            try:
                # Encuentra el título original correspondiente a la sugerencia para mostrar
                original_suggested_title = titles_orig_arg[
                    titles_clean_arg.index(corrected_query_normalized)
                ]
                print(
                    f"(💡 ¿Quizás quisiste decir: '{original_suggested_title}'? Buscando '{corrected_query_normalized}')"
                )
            except ValueError:
                print(
                    f"(💡 ¿Quizás quisiste decir: '{corrected_query_normalized}'? Buscando '{corrected_query_normalized}')"
                )
            q_normalized_query = corrected_query_normalized  # Usar la sugerencia para la búsqueda semántica
        else:
            print(
                "(✅ La consulta parece correcta o es muy similar a un título existente. Realizando búsqueda semántica.)"
            )
    else:
        print(
            "(🔎 No se encontró una corrección textual cercana. Realizando búsqueda semántica con la consulta original.)"
        )

    # Obtener embedding de la consulta (original normalizada o sugerencia normalizada)
    query_vec = model.encode(
        [q_normalized_query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    distances, idx_results = index.search(query_vec, k)
    results = [titles_orig_arg[i] for i in idx_results[0]]
    return results


# --------------------------------------------------------------------------- #
# programa interactivo
# --------------------------------------------------------------------------- #
def main():
    # Antes de construir o cargar, verificar si la DB existe
    if not os.path.exists(DATA_FILE):
        print(f"🔴 Error: El archivo de base de datos '{DATA_FILE}' no se encontró.")
        print("Por favor, asegúrate de que el archivo está en la ubicación correcta.")
        return

    if not os.path.exists(PKL_FILE):
        build_engine()
        # Comprobar si build_engine creó el archivo (pudo haber fallado)
        if not os.path.exists(PKL_FILE):
            print(
                "🔴 El motor no pudo ser construido. Revisa los mensajes de error anteriores."
            )
            return

    # Los nombres de variable devueltos por load_engine coinciden con el script original
    index_faiss, model_st, titles_original_for_display, titles_normalized_for_search = (
        load_engine()
    )

    print(
        f"🎬 Motor (desde DB, solo '{DB_TITLE_COLUMN}') listo. Escribe un título o ‘apagar sistema’ para salir."
    )

    while True:
        user_query = input("\n¿Película que buscas?: ").strip()
        if not user_query:
            continue
        if user_query.lower() == "apagar sistema":
            print("👋 Hasta pronto.")
            break

        # Pasar los argumentos correctos a buscar()
        found_titles = buscar(
            user_query,
            index_faiss,
            model_st,
            titles_original_for_display,
            titles_normalized_for_search,
        )
        if found_titles:
            for r_title in found_titles:
                print(" •", r_title)
        else:
            print("   No se encontraron resultados para tu búsqueda.")


if __name__ == "__main__":
    main()
