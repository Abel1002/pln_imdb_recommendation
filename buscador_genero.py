#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Buscador de películas por TÍTULO o por GÉNERO (BD TMDB SQLite).

Tablas:
  movies        (movie_id PK, title)
  movie_genres  (movie_id FK, genre_id FK)
  genres        (id PK, name)
"""
# --------------------------------------------------------------------------- #
import os, re, difflib, pickle, sqlite3, time
from pathlib import Path
from typing import List, Tuple, Set

import faiss                       # pip install faiss-cpu
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# ------------------------------ Configuración ------------------------------ #
DATA_FILE   = "tmdb_movies.db"
PKL_FILE    = "model_db_title_and_genres.pkl"
METRICS_TXT = "metrics_db_title_and_genres.txt"
MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

# ----------------------------- Utilidades ---------------------------------- #
def normaliza(txt: str) -> str:
    """Minúsculas + quita '(1999)' final."""
    return re.sub(r"\s*\(\d{4}\)$", "", str(txt)).strip().lower()

def parse_genres_query(query: str, known: Set[str]) -> List[str]:
    """
    Extrae géneros de la cadena del usuario admitiendo:
        Action Comedy
        Action and Comedy
        Action, Comedy
        "science fiction" & Drama
    Devuelve una lista en minúsculas.
    """
    q = query.lower()
    q = re.sub(r"[,&]", " ", q)       # , &  → espacio
    q = re.sub(r"\band\b", " ", q)    # 'and' → espacio
    q = " ".join(q.split())           # normaliza espacios

    found = []
    tmp   = q
    # recorrer géneros conocidos de mayor a menor longitud ("science fiction" antes que "war")
    for g in sorted(known, key=len, reverse=True):
        pat = r"\b" + re.escape(g) + r"\b"
        if re.search(pat, tmp):
            found.append(g)
            tmp = re.sub(pat, " ", tmp, count=1)

    return found

def format_genres_output(movie_genres: str, wanted: List[str]) -> str:
    """Marca géneros coincidentes en verde y los adicionales en rojo."""
    wanted_set = {normaliza(g) for g in wanted}
    out = []
    for g in [s.strip() for s in movie_genres.split(",") if s.strip()]:
        out.append(f"{GREEN if normaliza(g) in wanted_set else RED}{g}{RESET}")
    return ", ".join(out)

def looks_like_genre_query(q: str, known: Set[str]) -> bool:
    """Heurística: ¿la consulta únicamente contiene géneros conocidos?"""
    return bool(parse_genres_query(q, known))

# ---------------------- Lectura títulos + géneros -------------------------- #
SQL_JOIN = """
SELECT  m.title,
        GROUP_CONCAT(g.name, ', ') AS genres
FROM    movies       AS m
JOIN    movie_genres AS mg ON m.movie_id = mg.movie_id
JOIN    genres       AS g  ON mg.genre_id = g.id
GROUP BY m.movie_id
"""

def fetch_from_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute(SQL_JOIN)
    rows = cur.fetchall()
    titles = [r[0] for r in rows]
    genres = [r[1] or "" for r in rows]

    cur.execute("SELECT DISTINCT name FROM genres")
    all_genres = {r[0].lower() for r in cur.fetchall()}
    conn.close()
    return titles, genres, all_genres

# -------------------------- Construcción motor ----------------------------- #
def build_engine():
    print("⚙️  Construyendo motor (títulos + géneros)…")
    t0 = time.perf_counter()

    titles_orig, genres_orig, all_genres = fetch_from_db(DATA_FILE)
    if not titles_orig:
        print("❌ No se han encontrado películas en la BD."); return

    titles_norm = [normaliza(t) for t in titles_orig]

    print(f"· {len(titles_orig):,} títulos • Cargando modelo {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    emb   = model.encode(titles_norm, convert_to_numpy=True,
                         normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

    with open(PKL_FILE, "wb") as f:
        pickle.dump(dict(titles_orig=titles_orig,
                         titles_norm=titles_norm,
                         genres_orig=genres_orig,
                         all_genres=list(all_genres),
                         embeddings=emb,
                         model_name=MODEL_NAME), f)

    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas: {len(titles_orig)}\nDim vector: {emb.shape[1]}\n"
        f"Tiempo: {dt:.2f}s\nPickle: {Path(PKL_FILE).stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8")
    print(f"✅ Motor listo en {dt:.1f}s.")

# ---------------------------- Carga rápida --------------------------------- #
def load_engine():
    data  = pickle.load(open(PKL_FILE, "rb"))
    index = faiss.IndexFlatIP(data["embeddings"].shape[1]); index.add(data["embeddings"])
    model = SentenceTransformer(data["model_name"])
    return (index, model, data["titles_orig"], data["titles_norm"],
            data["genres_orig"], set(data["all_genres"]))

# ----------------------- Búsqueda por título ------------------------------- #
def buscar_por_titulo(q: str, index, model,
                      titles_orig, titles_norm, genres_orig) -> List[Tuple[str,str]]:
    qn = normaliza(q)
    sug = difflib.get_close_matches(qn, titles_norm, n=1, cutoff=0.7)
    if sug and sug[0] != qn:
        print(f"(💡 Quizás quisiste decir: '{titles_orig[titles_norm.index(sug[0])]}')")
        qn = sug[0]

    q_vec = model.encode([qn], convert_to_numpy=True,
                         normalize_embeddings=True).astype("float32")
    _, idx = index.search(q_vec, TOP_K)
    return [(titles_orig[i], genres_orig[i]) for i in idx[0]]

# ---------------------- Búsqueda por géneros ------------------------------- #
def buscar_por_genero(query: str,
                      titles_orig: List[str], genres_orig: List[str],
                      all_genres: Set[str]) -> List[Tuple[str,str]]:

    wanted = set(parse_genres_query(query, all_genres))
    if not wanted:
        return []

    exact, superset, partial = [], [], []

    for t, g_str in zip(titles_orig, genres_orig):
        movie_set = {normaliza(x) for x in g_str.split(",")}
        if not movie_set:
            continue
        if movie_set == wanted:
            exact.append((t, g_str))
        elif wanted <= movie_set:
            superset.append((t, g_str))
        elif movie_set & wanted:
            partial.append((t, g_str))

    return (exact + superset + partial)[:TOP_K]

# ---------------------------- CLI interactivo ------------------------------ #
def main():
    if not os.path.exists(DATA_FILE):
        print(f"🔴 Falta '{DATA_FILE}'."); return
    if not os.path.exists(PKL_FILE):
        build_engine()
        if not os.path.exists(PKL_FILE): return

    (index, model, titles_orig, titles_norm,
     genres_orig, all_genres) = load_engine()

    print("🎬 Motor listo. Ejemplos:")
    print("   Matrix")
    print("   Action Comedy            (≈ genero)")
    print("   genero: Horror, Thriller")
    print("Escribe 'apagar sistema' para salir.")

    while True:
        q = input("\nConsulta: ").strip()
        if not q: continue
        if q.lower() == "apagar sistema":
            print("👋 Adiós."); break

        # ¿Es género?
        if q.lower().startswith("genero:"):
            g_query = q[7:].strip()
            res = buscar_por_genero(g_query, titles_orig, genres_orig, all_genres)
        elif looks_like_genre_query(q, all_genres):
            g_query = q
            res = buscar_por_genero(g_query, titles_orig, genres_orig, all_genres)
        else:
            g_query = None
            res = buscar_por_titulo(q, index, model,
                                    titles_orig, titles_norm, genres_orig)

        if not res:
            print("   Sin resultados."); continue

        if g_query is None:
            print(f"Top {len(res)} por título:")
            for t, g in res:
                print(f" • {t}  (Géneros: {g})")
        else:
            print(f"Top {len(res)} por género:")
            wanted_list = parse_genres_query(g_query, all_genres)
            for t, g in res:
                print(f" • {t}  ({format_genres_output(g, wanted_list)})")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
