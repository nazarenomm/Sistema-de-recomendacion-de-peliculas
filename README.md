# Sistema de Recomendación de Películas basado en NLP

Trabajo Final Integrador (TFI) — Procesamiento de Lenguaje Natural  
---

## 📋 Descripción

Sistema de recomendación de contenido que utiliza técnicas de NLP para sugerir películas en español a partir de perfiles de usuario simulados. El sistema implementa y compara tres enfoques basados en similitud semántica sobre sinopsis, géneros y demás atributos cinematográficos.

**Corpus:** [`mathigatti/spanish_imdb_synopsis`](https://huggingface.co/datasets/mathigatti/spanish_imdb_synopsis) (~4.970 películas en español)  
**Perfiles:** 14 usuarios simulados (9 definidos + 5 ambiguos)

---

## Estructura del repositorio

```
├── notebooks/
│   ├── entrega_final.ipynb
│   ├── limpieza.ipynb
│   └── topic_modeling.ipynb
├── data/
│   ├── usuarios.csv                # Perfiles de usuario simulados
│   └── peliculas.csv               # Base de datos de películas con sus atributos disponibles
├── evals/                          # Resultados de evaluación de cada enfoque
│   ├── evaluacion_embeddings.csv
│   ├── evaluacion_embeddings_pond.csv 
│   └── evaluacion_tfidf.csv
├── evals/                          # Recomendaciones generadas por cada enfoque
│   ├── recomendaciones_embeddings.csv
│   ├── recomendaciones_embeddings_pond.csv 
│   └── recomendaciones_tfidf.csv
├── informe/
│   └── informe.pdf
├── requirements.txt
└── README.md
```

---

## Enfoques implementados

### Enfoque 1 — Sentence Transformers
Embeddings de oraciones con el modelo multilingüe `paraphrase-multilingual-MiniLM-L12-v2`. El perfil del usuario se representa como el promedio de los embeddings de las sinopsis de películas valoradas positivamente, y se rankean las películas por similitud coseno.

### Enfoque 2 — ST Embeddings ponderados por LLM
Extiende el Enfoque 1 clasificando la consulta del usuario mediante un LLM (`gemini-2.5-flash-lite` vía `google.colab.ai`) en cuatro categorías de intención: `normal`, `historial_positivo`, `historial_negativo`, `no_evaluable`. La ponderación ajusta dinámicamente el peso del historial de valoraciones negativas.

### Enfoque 3 — TF-IDF ponderado por LLM con keywords
El LLM extrae keywords relevantes de la consulta del usuario, que son traducidas al español y lematizadas con spaCy (`es_core_news_sm`). Se construye una representación TF-IDF del perfil enriquecido con estas keywords para rankear las sinopsis.

### (Adicional) BERTopic
Se exploró modelado de tópicos con BERTopic como cuarto enfoque. Fue descartado por la distribución esparsa de tópicos y el colapso de similitudes coseno (muchos scores = 1.0), lo que genera alta varianza entre usuarios y baja capacidad de discriminación.

---

## Evaluación

La evaluación se realizó sobre los **9 perfiles definidos** utilizando métricas basadas en solapamiento de géneros cinematográficos:

| Métrica | Descripción |
|--------|-------------|
| **Precision** | Proporción de recomendadas con género relevante |
| **Recall** | Proporción de géneros del perfil cubiertos |
| **F1** | Media armónica de precision y recall |
| **Exactitud Humana** | Proporción de recomendaciones "buenas", evaluadas con criterio humano |

Los 5 perfiles ambiguos se reportan por separado en las tablas del informe.
---

## Configuración del LLM (Enfoques 2 y 3)

Los enfoques 2 y 3 requieren acceso a un LLM. El proyecto usa `gemini-2.5-flash-lite` disponible en Google Colab:

```python
from google.colab import ai as colab_ai
# El cliente se inicializa automáticamente en el entorno Colab
```

> No puede ejecutarse fuera de UI de Colab

---

## Resultados principales

Los tres enfoques superan un baseline aleatorio en métricas de género. El Enfoque 2 (ST + LLM) mostró la mejor capacidad para capturar preferencias implícitas del usuario, mientras que el Enfoque 3 (TF-IDF + keywords) resultó más sensible a la calidad de las keywords extraídas.

Para el análisis completo, ver `informe/informe_grupo2.pdf`.

---

## Trabajo futuro

- Enriquecer el corpus con el resumen de reseñas de usuarios que provee IMDb, para capturar atributos hoy no evaluables (calidad percibida, ritmo, estilo).
- Reemplazar la clasificación discreta del LLM por una asignación continua de pesos a query e historial, en lugar de un mapeo fijo por categoría.
- Usar el LLM para reformular la query.
- Incorporar una validación humana a mayor escala, con múltiples evaluadores y un protocolo de consenso.
