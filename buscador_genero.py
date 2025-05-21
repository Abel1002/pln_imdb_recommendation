#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Buscador de películas por GÉNERO (basado únicamente en tablas de TMDB).

Tablas
------
movies        (movie_id PK, title)
movie_genres  (movie_id FK, genre_id FK)
genres        (id PK,    name)

Ranking devuelto
----------------
1.  Películas cuyo set de géneros == set buscado
2.  Películas que contienen todos los géneros buscados y alguno más
3.  Películas que contienen al menos uno de los géneros buscados
"""

# --------------------------------------------------------------------------- #
import os, re, sqlite3, time, pickle
from pathlib import Path
from typing import List, Tuple, Set

# ------------------------------ Configuración ------------------------------ #
DB_FILE     = "tmdb_movies.db"
PKL_FILE    = "genre_engine.pkl"
METRICS_TXT = "genre_engine_metrics.txt"
TOP_K       = 5

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

# ----------------------------- Utilidades ---------------------------------- #
def normaliza(txt: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(txt)).strip().lower()

def parse_genres_query(query: str, catalogo: Set[str]) -> List[str]:
    """Extrae los géneros introducidos por el usuario (robusto a ‘and’, ‘,’, ‘&’)."""
    q = query.lower()
    q = re.sub(r"[,&]", " ", q)
    q = re.sub(r"\band\b", " ", q)
    q = " ".join(q.split())

    encontrados, tmp = [], q
    for g in sorted(catalogo, key=len, reverse=True):
        pat = r"\b" + re.escape(g) + r"\b"
        if re.search(pat, tmp):
            encontrados.append(g)
            tmp = re.sub(pat, " ", tmp, count=1)
    return encontrados

def colores(g_str: str, buscados: Set[str]) -> str:
    out = []
    for g in [x.strip() for x in g_str.split(",") if x.strip()]:
        out.append(f"{GREEN if normaliza(g) in buscados else RED}{g}{RESET}")
    return ", ".join(out)

# --------------------- Cargar datos de la base de datos -------------------- #
SQL = """
SELECT  m.title, GROUP_CONCAT(g.name, ', ')
FROM    movies m
JOIN    movie_genres mg ON m.movie_id = mg.movie_id
JOIN    genres g        ON mg.genre_id = g.id
GROUP BY m.movie_id
"""

def leer_bd(db: str):
    conn = sqlite3.connect(db)
    cur  = conn.cursor()
    cur.execute(SQL)
    rows   = cur.fetchall()
    titles = [r[0] for r in rows]
    genres = [r[1] or "" for r in rows]

    cur.execute("SELECT DISTINCT name FROM genres")
    catalogo = {r[0].lower() for r in cur.fetchall()}
    conn.close()
    return titles, genres, catalogo

# -------------------------- Construir / cargar motor ----------------------- #
def construir_motor():
    print("⚙️  Construyendo motor de géneros…")
    t0 = time.perf_counter()
    titles, genres, catalogo = leer_bd(DB_FILE)
    with open(PKL_FILE, "wb") as f:
        pickle.dump(dict(titles=titles,
                         genres=genres,
                         catalogo=list(catalogo)), f)
    dt = time.perf_counter() - t0
    Path(METRICS_TXT).write_text(
        f"Películas: {len(titles)}\nTiempo: {dt:.2f}s\nPickle: {Path(PKL_FILE).stat().st_size/1e6:.1f} MB\n",
        encoding="utf-8")
    print(f"✅ Motor guardado ({dt:.1f}s).")

def cargar_motor():
    d = pickle.load(open(PKL_FILE, "rb"))
    return d["titles"], d["genres"], set(d["catalogo"])

# --------------------------- Lógica de búsqueda ---------------------------- #
def buscar_por_genero(query: str,
                      titles: List[str], genres: List[str],
                      catalogo: Set[str]) -> List[Tuple[str,str]]:

    buscados = set(parse_genres_query(query, catalogo))
    if not buscados:                           # nada reconocido → sin resultados
        return []

    exact, superset, parcial = [], [], []

    for t, g_str in zip(titles, genres):
        peli = {normaliza(x) for x in g_str.split(",") if x.strip()}
        if not peli: continue

        if peli == buscados:
            exact.append((t, g_str))
        elif buscados <= peli:
            superset.append((t, g_str))
        elif peli & buscados:
            parcial.append((t, g_str))

    return (exact + superset + parcial)[:TOP_K]

# ------------------------------ CLI simple --------------------------------- #
def main():
    if not os.path.exists(DB_FILE):
        print(f"🔴 Falta la BD '{DB_FILE}'."); return
    if not os.path.exists(PKL_FILE):
        construir_motor()

    titles, genres, catalogo = cargar_motor()

    print("🎬 Buscador por género listo. Ejemplos:")
    print("   Action Comedy")
    print("   genero: Horror, Thriller")
    print("Escribe 'apagar' para salir.")

    while True:
        q = input("\nConsulta: ").strip()
        if not q: continue
        if q.lower() == "apagar":
            print("👋 Adiós"); break

        if q.lower().startswith("genero:"):
            q = q[7:].strip()

        res = buscar_por_genero(q, titles, genres, catalogo)
        if not res:
            print("   Sin resultados."); continue

        buscados = set(parse_genres_query(q, catalogo))
        print(f"Top {len(res)}:")
        for t, g_str in res:
            print(f" • {t}  ({colores(g_str, buscados)})")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
