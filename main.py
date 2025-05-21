#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agente jefe que enruta la consulta del usuario al motor adecuado
(título, géneros o descripción).
"""
import os
import re
import difflib
import pickle
import time
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer

# ---------- constantes ----------------------------------------------------- #
PKL_TITLE    = "model_db_title_only.pkl"
PKL_GENRE    = "genre_engine.pkl"
PKL_OVERVIEW = "st_overview_engine.pkl"
DATA_FILE    = "tmdb_movies.db" 
TOP_K        = 5

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# ---------- utilidades genéricas ------------------------------------------- #
def normaliza(txt: str) -> str:
    return re.sub(r"\s*\(\d{4}\)$", "", str(txt)).strip().lower()

# ----------------- Carga de los tres motores ------------------------------- #
def load_faiss(pkl_path: str, is_overview_engine: bool = False):
    print(f"Cargando motor FAISS desde: {pkl_path}...")
    data = pickle.load(open(pkl_path, "rb"))
    
    if "embeddings" in data:
        emb = data["embeddings"]
    elif "embeddings_for_rebuild" in data:
        emb = data["embeddings_for_rebuild"]
    else:
        raise ValueError(f"No se encontró la clave 'embeddings' o 'embeddings_for_rebuild' en {pkl_path}")

    model = SentenceTransformer(data["model_name"])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    if is_overview_engine:
        if "titles_for_display" in data and "titles_orig" not in data:
            data["titles_orig"] = data["titles_for_display"]
    
    if "titles_orig" not in data:
        raise ValueError(f"Clave 'titles_orig' (o 'titles_for_display' para overview) no encontrada en {pkl_path}")

    if "titles_norm" not in data:
        print(f"  Generando 'titles_norm' para {pkl_path} a partir de 'titles_orig'.")
        data["titles_norm"] = [normaliza(t) for t in data["titles_orig"]]
    
    print(f"  Motor {pkl_path} cargado. Modelo: {data['model_name']}, {len(data['titles_orig'])} títulos.")
    return data, model, index

def load_genre_engine(pkl_path: str) -> Dict[str, Any]:
    print(f"Cargando motor de géneros desde: {pkl_path}...")
    with open(pkl_path, "rb") as f:
        genre_data_raw = pickle.load(f)
    
    processed_data = {
        "titles_orig": list(genre_data_raw.get("titles", [])),
        "genres_orig": list(genre_data_raw.get("genres", [])),
        "all_genres": set(map(str.lower, genre_data_raw.get("catalogo", [])))
    }
    print(f"  Motor de géneros {pkl_path} cargado. {len(processed_data['titles_orig'])} títulos, {len(processed_data['all_genres'])} géneros en catálogo.")
    return processed_data

# --- Carga ---
data_title, model_title, index_title = load_faiss(PKL_TITLE)
data_genre: Dict[str, Any] = load_genre_engine(PKL_GENRE)
all_genres: Set[str] = data_genre["all_genres"]
title_to_genre_map: Dict[str, str] = {
    title: genres_str for title, genres_str in zip(data_genre["titles_orig"], data_genre["genres_orig"])
}
print(f"  Mapa título->género creado con {len(title_to_genre_map)} entradas.")
data_over, model_over, index_over = load_faiss(PKL_OVERVIEW, is_overview_engine=True)

# ---------------------- funciones de búsqueda ------------------------------ #
def buscar_faiss(q:str, model, index, titles_orig: List[str], titles_norm: List[str])->List[str]:
    qn = normaliza(q)
    sug = difflib.get_close_matches(qn, titles_norm, n=1, cutoff=0.7)
    corrected_query_for_embedding = qn
    if sug and sug[0] != qn:
        try:
            original_suggested_title = titles_orig[titles_norm.index(sug[0])]
            print(f"(💡 ¿Quizás quisiste decir: '{original_suggested_title}'? Usando '{sug[0]}' para la búsqueda semántica)")
        except (ValueError, IndexError):
            print(f"(💡 ¿Quizás quisiste decir: '{sug[0]}'? Usando '{sug[0]}' para la búsqueda semántica)")
        corrected_query_for_embedding = sug[0]

    vec = model.encode([corrected_query_for_embedding], convert_to_numpy=True,
                       normalize_embeddings=True).astype("float32")
    _, idx = index.search(vec, TOP_K)
    return [titles_orig[i] for i in idx[0] if i < len(titles_orig)]

def parse_genres_query(user_q:str)->List[str]:
    q = user_q.lower()
    q = re.sub(r"[,&]", " ", q)
    q = re.sub(r"\band\b", " ", q, flags=re.IGNORECASE)
    q = " ".join(q.split())
    found=[]
    tmp=q
    for g in sorted(all_genres, key=len, reverse=True):
        pat=r"\b"+re.escape(g)+r"\b"
        if re.search(pat,tmp):
            found.append(g)
            tmp=re.sub(pat," ",tmp,1)
    return found

def format_genres_output(movie_g_str:str, wanted_genres_norm:List[str])->str:
    wanted_set_norm = {normaliza(x) for x in wanted_genres_norm}
    out=[]
    for g_individual in [s.strip() for s in movie_g_str.split(",") if s.strip()]:
        g_individual_norm = normaliza(g_individual)
        colored_g = f"{GREEN if g_individual_norm in wanted_set_norm else RED}{g_individual}{RESET}"
        out.append(colored_g)
    return ", ".join(out)

def buscar_por_generos(user_q:str)->List[Tuple[str,str]]:
    wanted_norm = set(parse_genres_query(user_q))
    if not wanted_norm: return []
    
    exac, super_, part = [],[],[]
    for title, genres_str in zip(data_genre["titles_orig"], data_genre["genres_orig"]):
        movie_genres_set_norm = {normaliza(x) for x in genres_str.split(",") if x.strip()}
        if not movie_genres_set_norm: continue

        if movie_genres_set_norm == wanted_norm:
            exac.append((title, genres_str))
        elif wanted_norm.issubset(movie_genres_set_norm):
            super_.append((title, genres_str))
        elif movie_genres_set_norm & wanted_norm:
            part.append((title, genres_str))
            
    return (exac+super_+part)[:TOP_K]

# ------------------------ clasificación de la consulta --------------------- #
def classify_query(q:str)->str:
    lq = q.lower()
    words = q.strip().split()
    num_words = len(words)

    if lq.startswith("genero:"):
        return "genre"

    overview_keywords = [
        "película sobre", "pelicula sobre", "peli sobre", "película de un", "peli de un",
        "películas parecidas a", "peliculas parecidas a", "historia de", "trama sobre",
        "busco algo que trate de", "donde sale un", "persona que", "grupo de", "un tío que", "una tía que"
    ]
    if any(kw in lq for kw in overview_keywords) or q.endswith("?"):
        return "overview"

    if re.search(r"\(\d{4}\)$", q.strip()):
        return "title"

    extracted_genres = parse_genres_query(q) 
    if extracted_genres:
        temp_q = q
        for g_norm in extracted_genres: 
            temp_q = re.sub(r'\b' + re.escape(g_norm) + r'\b', '', temp_q, flags=re.IGNORECASE)
        remaining_text = temp_q.replace("genero:", "").strip() 
        remaining_words_list = [w for w in remaining_text.split() if w]
        if not remaining_words_list: 
            return "genre"
        if len(remaining_words_list) <= 2 and all(len(word) <=3 for word in remaining_words_list):
             return "genre"

    qn = normaliza(q)

    if 1 <= num_words <= 3 and not extracted_genres: 
        short_query_title_match = difflib.get_close_matches(qn, data_title["titles_norm"], n=1, cutoff=0.72)
        if short_query_title_match:
            return "title"

    high_confidence_title_match = difflib.get_close_matches(qn, data_title["titles_norm"], n=1, cutoff=0.88)
    if high_confidence_title_match:
        return "title"

    if num_words > 6: 
        return "overview"
        
    if extracted_genres and num_words > len(extracted_genres) + 1 : 
        return "ambiguous"

    return "ambiguous"

def pedir_confirmacion(tipo_sugerido:str)->bool:
    resp=input(f"¿Buscar por {tipo_sugerido}? (s/n): ").strip().lower()
    return resp.startswith("s")

# --------------------------- CLI principal --------------------------------- #
def main():
    print("🤖 Buscador agente listo.\nEscribe tu consulta o 'exit'.\n")
    while True:
        q_user = input("Consulta: ").strip()
        if not q_user: continue
        if q_user.lower() in {"exit","salir","apagar sistema", "quit", "terminar"}:
            print("👋 Hasta luego"); break

        tipo = classify_query(q_user)
        
        print(f"--- Clasificación inicial: {tipo} ---")

        if tipo=="ambiguous":
            print("No tengo claro si es título o descripción/género.")
            if pedir_confirmacion("título"):
                tipo="title"
            elif pedir_confirmacion("descripción (sinopsis)"):
                tipo="overview"
            elif pedir_confirmacion("género(s)"):
                tipo="genre"
            else:
                print("Búsqueda abortada."); continue
        
        start_time = time.perf_counter()
        results_found = False

        if tipo=="title":
            res = buscar_faiss(q_user, model_title, index_title,
                               data_title["titles_orig"], data_title["titles_norm"])
            if not res: print("Sin resultados por TÍTULO.");
            else:
                results_found = True
                print(f"Top {len(res)} por TÍTULO:")
                for r_title in res:
                    genres_for_title = title_to_genre_map.get(r_title, "?")
                    print(f" • {r_title}  (Géneros: {genres_for_title})")

        elif tipo=="genre":
            gq = q_user
            if q_user.lower().startswith("genero:"):
                gq = q_user[7:].strip()
            
            res = buscar_por_generos(gq)
            if not res: print("Sin resultados para esos géneros.");
            else:
                results_found = True
                wanted_genres_list_norm = parse_genres_query(gq)
                print(f"Top {len(res)} por GÉNERO ({', '.join(wanted_genres_list_norm)}):")
                for r_title, r_genres_str in res:
                    print(f" • {r_title}  ({format_genres_output(r_genres_str, wanted_genres_list_norm)})")

        elif tipo=="overview":
            res = buscar_faiss(q_user, model_over, index_over,
                               data_over["titles_orig"], data_over["titles_norm"])
            if not res: print("Sin resultados por DESCRIPCIÓN.");
            else:
                results_found = True
                print(f"Top {len(res)} por DESCRIPCIÓN:")
                for r_title in res:
                    genres_for_title = title_to_genre_map.get(r_title, "?")
                    print(f" • {r_title}  (Géneros: {genres_for_title})")
        
        if results_found:
             print(f"Búsqueda completada en {time.perf_counter() - start_time:.2f}s")
        print("-" * 30)

# --------------------------------------------------------------------------- #
if __name__=="__main__":
    missing_files = []
    for f_p, f_name in [(PKL_TITLE, "buscador por título"), 
                        (PKL_GENRE, "buscador por géneros"), 
                        (PKL_OVERVIEW, "buscador por sinopsis")]:
        if not Path(f_p).exists():
            missing_files.append(f"'{f_p}' (para {f_name})")
    
    if missing_files:
        print(f"❌ Faltan los siguientes archivos PKL:")
        for mf in missing_files:
            print(f"  - {mf}")
        print("Asegúrate de haber generado los tres modelos (.pkl) y que estén en el mismo directorio que este script.")
        exit(1)
        
    main()