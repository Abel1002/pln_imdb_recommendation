Okay, aquí tienes una versión mejorada de tu `README.md` lista para GitHub, combinando tu estructura con la información detallada del proyecto y ajustando los detalles técnicos para que coincidan con los scripts que proporcionaste.

```markdown
# 🎬 IA Movie Search Engine

> Módulo impulsado por IA para la **búsqueda semántica de películas** dentro de una plataforma web/app de _streaming_. Este repositorio cubre únicamente el apartado de **búsqueda avanzada** por títulos, combinaciones de géneros y descripciones/sinopsis. Forma parte de un proyecto mayor que emplea SVD, auto-encoders y redes neuronales para la recomendación.

---

## 🚀 Características Principales

* **Agente Jefe Inteligente (`main.py`):** Analiza la consulta del usuario y la dirige al motor de búsqueda más adecuado.
* **Búsqueda Multimodal:**
    * **Por Título:** Encuentra películas por su título, incluso con errores tipográficos leves, utilizando embeddings semánticos.
    * **Por Género:** Permite buscar por uno o múltiples géneros con lógica de coincidencia flexible.
    * **Por Descripción:** Busca películas basándose en el significado semántico de su sinopsis.
* **Corrección de Typos:** Para búsquedas por título, se integra `difflib` para sugerir correcciones.
* **CLI Interactiva:** Una interfaz de línea de comandos para probar el sistema de búsqueda.

---

## 🛠️ Motores de Búsqueda

| Motor         | Script que lo Genera    | Artefacto (`*.pkl`)         | Ejemplos de Consulta                        |
| :------------ | :---------------------- | :-------------------------- | :------------------------------------------ |
| **Título** | `buscador_por_titulo.py`  | `model_db_title_only.pkl`   | `toy story`, `se7en (1995)`, `toi storip`   |
| **Género** | `buscador_por_genero.py`  | `genre_engine.pkl`          | `Action Comedy Horror`, `genero: drama & romance` |
| **Descripción** | `buscador_por_sinopsis.py`| `st_overview_engine.pkl`    | `película sobre un muñeco que cobra vida`    |
| **Agente Jefe** | `main.py`                 | —                           | (Decide qué motor usar según la query)      |

`main.py` carga los tres modelos (`.pkl`) y ofrece una CLI unificada para interactuar con el sistema de búsqueda:

```
python main.py

Consulta: toy story
--- Clasificación inicial: title ---
(💡 ¿Quizás quisiste decir: 'Toy Story'? Usando 'toy story' para la búsqueda semántica)
Top 5 por TÍTULO:
 • Toy Story  (Géneros: Animation, Adventure, Family, Comedy)
...

Consulta: genero: action comedy
--- Clasificación inicial: genre ---
Top 5 por GÉNERO (action, comedy):
 • Hot Fuzz  (Géneros: Action, Comedy, Crime)
...

Consulta: película sobre un muñeco que cobra vida
--- Clasificación inicial: overview ---
Top 5 por DESCRIPCIÓN:
 • Toy Story  (Géneros: Animation, Adventure, Family, Comedy)
...
```

---

## 🗂 Estructura del Proyecto (Mínima)

```
pln_movie_search/
│
├── main.py                     # Agente jefe y CLI principal
├── buscador_por_titulo.py      # Script para generar el motor de títulos
├── buscador_por_genero.py      # Script para generar el motor de géneros
├── buscador_por_sinopsis.py    # Script para generar el motor de sinopsis
│
├── model_db_title_only.pkl   # (Generado)
├── genre_engine.pkl          # (Generado)
├── st_overview_engine.pkl    # (Generado)
│
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
│
└── data/
    └── tmdb_movies.db          # Base de datos de películas (requiere descarga)
```

**Base de Datos `tmdb_movies.db`:**

* Contiene las tablas `movies`, `movie_genres`, `genres` necesarias para construir los motores.
* Tamaño aproximado: 65 MB.
* **Descarga:** Debes descargar este archivo (por ejemplo, desde un [enlace de Drive - ¡actualiza este enlace!](https://drive.google.com/your-link-here)) y colocarlo dentro del directorio `data/`.

---

## ⚙️ Instalación y Configuración

Se recomienda utilizar un entorno virtual.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Abel1002/pln_imdb_recommendation.git](https://github.com/Abel1002/pln_imdb_recommendation.git) # O la URL de tu repositorio
    cd pln_imdb_recommendation # O el nombre de tu directorio
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    ```
    * Windows:
        ```bash
        .venv\Scripts\activate
        ```
    * macOS / Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
    Un `requirements.txt` básico podría incluir:
    ```txt
    sentence-transformers
    faiss-cpu
    numpy # Generalmente una dependencia de los anteriores
    # sqlite3 es parte de la librería estándar de Python
    ```

---

## 🔄 Flujo de Trabajo

1.  **Generar los Motores de Búsqueda (solo la primera vez o tras actualizar `tmdb_movies.db`):**
    Asegúrate de tener `tmdb_movies.db` en la carpeta `data/`. Luego, ejecuta los siguientes scripts desde la raíz del proyecto:
    ```bash
    python buscador_por_titulo.py     # Genera -> model_db_title_only.pkl y metrics_db_title_only.txt
    python buscador_por_genero.py     # Genera -> genre_engine.pkl y genre_engine_metrics.txt
    python buscador_por_sinopsis.py   # Genera -> st_overview_engine.pkl y st_overview_metrics.txt
    ```
    Cada script también crea un archivo `*_metrics.txt` con información sobre el tiempo de construcción y el tamaño del artefacto.

2.  **Iniciar el Agente Jefe:**
    Una vez que los archivos `.pkl` han sido generados, puedes ejecutar el buscador principal:
    ```bash
    python main.py
    ```

3.  **Actualizaciones:**
    Si actualizas o amplías la base de datos `tmdb_movies.db` con nuevas películas, deberás volver a ejecutar los scripts de generación de motores para que los cambios se reflejen en las búsquedas.

---

## ✨ Tecnologías y Librerías Clave

Este proyecto utiliza varias librerías de Python especializadas en PLN y ciencia de datos:

* **`sentence-transformers`**:
    * Para generar *embeddings* (representaciones vectoriales densas) de alta calidad para los títulos y las sinopsis de las películas. Estos embeddings capturan el significado semántico.
    * Los scripts de construcción de motores utilizan modelos como `paraphrase-multilingual-MiniLM-L12-v2` (o el especificado en `MODEL_NAME` dentro de cada script de construcción).
* **`faiss` (Facebook AI Similarity Search)**:
    * Para realizar búsquedas de similitud eficientes entre los embeddings de las consultas y los embeddings de los datos indexados (títulos y sinopsis).
    * Se utiliza `IndexFlatIP` para búsquedas exactas basadas en el producto interno (similitud coseno para vectores normalizados).
* **`difflib`**:
    * Módulo estándar de Python para comparar secuencias. Se usa para la corrección de errores tipográficos en las consultas de títulos (`get_close_matches`).
* **`re` (Expresiones Regulares)**:
    * Utilizado para la normalización de texto y el análisis de consultas de género.
* **`sqlite3`**:
    * Para interactuar con la base de datos `tmdb_movies.db` durante la fase de construcción de los motores.
* **`pickle`**:
    * Para serializar (guardar) y deserializar (cargar) los datos procesados de los motores (listas, diccionarios, embeddings) en los archivos `.pkl`.

---

## 🛠 Personalización (Aplicable a los scripts actuales)

* **`TOP_K` en `main.py`:** Modifica esta constante para cambiar el número de resultados principales devueltos.
* **Modelos de `sentence-transformers`:** Puedes experimentar con diferentes modelos en los scripts de construcción (`buscador_por_titulo.py`, `buscador_por_sinopsis.py`) cambiando la variable `MODEL_NAME`. **Recuerda regenerar los `.pkl`** si cambias el modelo.
* **Umbrales de `difflib`:**
    * En `main.py`, la función `classify_query` usa `cutoff=0.72` para la clasificación de títulos cortos con typos y `cutoff=0.88` para coincidencias de alta confianza.
    * La función `buscar_faiss` usa `cutoff=0.7` para la sugerencia de corrección de typos.
    Estos valores pueden ajustarse para balancear la sensibilidad a typos y la precisión.

---

## 📄 Licencia

MIT License. (O la licencia que hayas elegido para tu proyecto)

```

**Notas Importantes para Ti:**

1.  **Actualiza el Enlace de Descarga:** Reemplaza `(https://drive.google.com/your-link-here)` con el enlace real a tu archivo `tmdb_movies.db`.
2.  **`requirements.txt`:** Asegúrate de tener un archivo `requirements.txt` preciso en tu repositorio. Puedes generarlo con `pip freeze > requirements.txt` después de instalar todas las dependencias en tu entorno virtual.
3.  **Nombre del Repositorio/Directorio:** Ajusta `Abel1002/pln_imdb_recommendation` y `pln_imdb_recommendation` si tu repositorio o directorio local tienen nombres diferentes.
4.  **Personalización:** La sección "Personalización" ahora refleja lo que se puede modificar en los scripts que *proporcionaste para la documentación*. Si implementas las mejoras más avanzadas que mencionaste (E5-large, persistencia de índices `.faiss`, RRF, umbral de similitud en búsqueda), deberás actualizar esta sección y la de "Tecnologías" para reflejar esos cambios.

Este README debería proporcionar una buena visión general de tu proyecto para cualquier persona que visite tu repositorio de GitHub.
