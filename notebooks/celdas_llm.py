# %% [markdown]
# ## Enfoque con LLM (Gemini)
# 
# ### Motivación
# 
# El pipeline "Promediando" usa pesos fijos (0.7 query, 0.3 historial) y siempre
# busca las 5 películas **más similares**. Esto no es óptimo para todos los casos:
# 
# - Si la query es "quiero algo distinto a lo de siempre", usar el historial como 
#   señal positiva y buscar similitud máxima **perjudica** la recomendación.
# - Si la query es muy específica ("una de Tarantino"), el historial no debería 
#   pesar tanto.
# - Si la query es vaga ("algo entretenido"), el historial debería ayudar más.
#
# ### Propuesta
#
# Agregar un paso con un LLM (Gemini) que infiera dinámicamente:
# 1. **weights**: `[w_query, w_historial]` para ponderar la contribución de cada fuente
# 2. **dirección**: `"similar"` (top-5 más cercanas) o `"distinto"` (top-5 más lejanas)
# 3. **query reescrita**: transformar la query coloquial en una sinopsis cinematográfica
#    para mejorar la similitud semántica con las descripciones de las películas.

# %%
# Setup de Gemini
import os
import json
from dotenv import load_dotenv
from google import genai

# Cargar API key desde .env (ubicado en el directorio padre del proyecto)
# Intentamos múltiples ubicaciones posibles del .env
for env_path in [
    os.path.join(os.path.dirname(os.getcwd()), '.env'),       # ../
    os.path.join(os.getcwd(), '.env'),                          # ./
    os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), '.env'),  # ../../
    r'C:\Users\recla\OneDrive\Escritorio\INLP\.env',           # ruta absoluta
]:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Cargado .env desde: {env_path}")
        break

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("No se encontro GEMINI_API_KEY en el .env")

client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-2.0-flash"
print("Gemini configurado correctamente")

# %% [markdown]
# ### Funciones LLM

# %%
def analizar_query_con_llm(query: str) -> dict:
    """
    Usa Gemini para analizar la query del usuario e inferir:
    - weights: [peso_query, peso_historial] que suman 1
    - direccion: "similar" o "distinto"
    """
    prompt = f"""Sos un asistente experto en sistemas de recomendación de películas.

Un usuario quiere una recomendación. Su query es:
"{query}"

Tu tarea es analizar la intención del usuario y devolver un JSON con dos campos:

1. "weights": una lista de dos números [peso_query, peso_historial] que indican cuánta 
   importancia darle a la query actual vs. al historial de películas que ya vio.
   - Deben sumar 1.0
   - Si la query es muy específica (ej: "quiero una de Tarantino"), el peso de la query 
     debería ser alto (ej: [0.9, 0.1])
   - Si la query es vaga y el historial puede ayudar (ej: "algo entretenido"), 
     el peso del historial sube (ej: [0.5, 0.5])
   - Si la query rechaza el historial (ej: "algo distinto a lo que siempre veo"), 
     el peso del historial puede ser bajo o incluso 0 (ej: [1.0, 0.0])

2. "direccion": "similar" o "distinto"
   - "similar": recomendar películas parecidas al perfil del usuario (lo habitual)
   - "distinto": recomendar películas con BAJA similitud al perfil del usuario.
     Usar cuando el usuario explícitamente pide algo diferente, nuevo, fuera de su 
     zona de comfort, o dice que quiere cambiar.

Respondé ÚNICAMENTE con el JSON válido, sin explicaciones, sin markdown, sin bloques de código:
{{"weights": [<peso_query>, <peso_historial>], "direccion": "<similar o distinto>"}}"""

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip()
        
        # Limpiar posibles bloques de código markdown
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # quitar primera línea ```json
            text = text.rsplit("```", 1)[0]  # quitar última línea ```
            text = text.strip()
        
        result = json.loads(text)
        
        # Validación
        weights = result.get("weights", [0.7, 0.3])
        direccion = result.get("direccion", "similar")
        
        # Validar que weights sumen ~1
        if len(weights) == 2 and abs(sum(weights) - 1.0) < 0.05:
            weights = [float(w) for w in weights]
        else:
            print(f"  ⚠ Weights inválidos ({weights}), usando default [0.7, 0.3]")
            weights = [0.7, 0.3]
        
        # Validar dirección
        if direccion not in ["similar", "distinto"]:
            print(f"  ⚠ Dirección inválida ({direccion}), usando 'similar'")
            direccion = "similar"
        
        return {"weights": weights, "direccion": direccion}
    
    except Exception as e:
        print(f"  ⚠ Error al llamar a Gemini: {e}")
        print(f"  Usando valores default: weights=[0.7, 0.3], direccion='similar'")
        return {"weights": [0.7, 0.3], "direccion": "similar"}


def reescribir_query_como_sinopsis(query: str) -> str:
    """
    Usa Gemini para reescribir la query coloquial del usuario como una sinopsis
    cinematográfica, facilitando la similitud semántica con las descripciones 
    de películas en el dataset.
    """
    prompt = f"""Sos un crítico de cine experto. 

Un usuario busca una película y describió lo que quiere así:
"{query}"

Tu tarea es reescribir esa descripción como si fuera la sinopsis de una película ideal 
que satisfaga ese pedido. Escribí un párrafo breve (2-3 oraciones) en español, 
en tercera persona, con el estilo de una sinopsis de IMDb.

No menciones que estás reescribiendo. No incluyas título. Solo escribí la sinopsis directamente."""

    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        sinopsis = response.text.strip()
        
        # Limpiar posibles comillas envolventes
        if sinopsis.startswith('"') and sinopsis.endswith('"'):
            sinopsis = sinopsis[1:-1]
        
        return sinopsis
    
    except Exception as e:
        print(f"  ⚠ Error al reescribir query: {e}")
        return query  # fallback: usar la query original

# %% [markdown]
# ### Aplicar LLM a cada usuario
# 
# Para cada usuario:
# 1. Analizar su query con Gemini → obtener weights y dirección
# 2. Reescribir la query como sinopsis → generar embedding más compatible
# 3. Construir el perfil combinando query reescrita + historial con weights dinámicos
# 4. Seleccionar top-5 o bottom-5 según dirección

# %%
import time

resultados_llm = []

print("=" * 80)
print("PIPELINE CON LLM (Gemini)")
print("=" * 80)

for idx, row in usuarios.iterrows():
    print(f"\n{'─' * 70}")
    print(f"[Usuario] {row['nombre']} ({row['tipo_perfil']})")
    print(f"[Query]   {row['query']}")
    
    # 1. Analizar query con LLM
    params = analizar_query_con_llm(row['query'])
    weights = params['weights']
    direccion = params['direccion']
    print(f"[LLM]    weights: {weights}, direccion: {direccion}")
    
    # 2. Reescribir query como sinopsis
    query_reescrita = reescribir_query_como_sinopsis(row['query'])
    print(f"[Sinopsis] {query_reescrita[:100]}...")
    
    # 3. Generar embedding de la query reescrita
    query_emb = model.encode([query_reescrita])[0]
    
    # 4. Obtener embedding del historial del usuario
    historial_emb = historial_embeddings[idx]
    
    # 5. Construir perfil del usuario con weights dinámicos
    user_emb = np.average(
        [query_emb, historial_emb],
        axis=0,
        weights=weights
    )
    
    # 6. Calcular similitud coseno
    scores_user = cosine_similarity([user_emb], pelis_embeddings)[0]
    
    # 7. Seleccionar top-5 según dirección
    if direccion == 'distinto':
        top5 = scores_user.argsort()[:5]  # las 5 MENOS similares
        print(f"  >> Modo DISTINTO: seleccionando las 5 peliculas mas diferentes")
    else:
        top5 = scores_user.argsort()[-5:][::-1]  # las 5 MAS similares
    
    # 8. Mostrar y guardar resultados
    print(f"\nRecomendaciones:")
    recomendaciones = []
    for rank, movie_idx in enumerate(top5, 1):
        pelicula = df_pelis.iloc[movie_idx]
        score = scores_user[movie_idx]
        print(f"  {rank}. {pelicula['name']} ({int(pelicula['year']) if not pd.isna(pelicula['year']) else 'N/A'}) — {score:.4f}")
        recomendaciones.append({
            'rank': rank,
            'nombre': pelicula['name'],
            'year': pelicula['year'],
            'genre': pelicula['genre'],
            'score': score,
            'movie_idx': movie_idx
        })
    
    resultados_llm.append({
        'usuario_idx': idx,
        'nombre': row['nombre'],
        'tipo_perfil': row['tipo_perfil'],
        'query': row['query'],
        'query_reescrita': query_reescrita,
        'weights': weights,
        'direccion': direccion,
        'top5_indices': top5,
        'recomendaciones': recomendaciones
    })
    
    # Breve pausa para no exceder rate limits de la API
    time.sleep(1)

# %% [markdown]
# ### Evaluación del pipeline LLM

# %%
resultados_eval_llm = []

for r in resultados_llm:
    user_idx = r['usuario_idx']
    
    # Solo evaluamos los primeros 9 usuarios (los que tienen etiquetas)
    if user_idx >= len(etiquetas_a_ojo_def):
        continue
    
    row = usuarios.iloc[user_idx]
    generos_esperados = set(etiquetas_a_ojo_def[user_idx])
    
    print(f"\n{'=' * 70}")
    print(f"{row['nombre']} ({row['tipo_perfil']})")
    print(f"Query: {row['query']}")
    print(f"Query reescrita: {r['query_reescrita'][:80]}...")
    print(f"Weights: {r['weights']} | Dirección: {r['direccion']}")
    print(f"Géneros Esperados: {', '.join(generos_esperados)}")
    print(f"{'=' * 70}")
    
    generos_recomendados = Counter()
    peliculas_buenas = 0
    peliculas_malas = []
    
    print("\nTop-5 Recomendaciones:")
    for rec in r['recomendaciones']:
        pelicula = df_pelis.iloc[rec['movie_idx']]
        score = rec['score']
        
        generos_str = pelicula['genre'].strip('[]')
        generos_list = [g.strip() for g in generos_str.split(',')]
        generos_pelicula = set(generos_list)
        generos_recomendados.update(generos_list)
        
        es_buena = bool(generos_pelicula & generos_esperados)
        if es_buena:
            peliculas_buenas += 1
            marker = "✓"
        else:
            peliculas_malas.append(pelicula['name'])
            marker = "✗"
        
        print(f"  {rec['rank']}. [{marker}] {pelicula['name']} ({int(pelicula['year']) if not pd.isna(pelicula['year']) else 'N/A'}) — {score:.4f}")
        print(f"     Géneros: {', '.join(generos_list)}")
    
    # Métricas
    generos_capturados = set(generos_recomendados.keys())
    recall = len(generos_capturados & generos_esperados) / len(generos_esperados)
    precision = peliculas_buenas / 5
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMÉTRICAS:")
    print(f"  Recall (géneros):     {recall:.1%}  ({len(generos_capturados & generos_esperados)}/{len(generos_esperados)})")
    print(f"  Precision (películas): {precision:.1%}  ({peliculas_buenas}/5)")
    print(f"  F1-Score:            {f1:.1%}")
    
    if peliculas_malas:
        print(f"\nPelículas problemáticas (sin géneros esperados):")
        for pelicula in peliculas_malas:
            print(f"    - {pelicula}")
    
    resultados_eval_llm.append({
        'Usuario': row['nombre'],
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'Películas Malas': len(peliculas_malas)
    })

# Resumen
print(f"\n\n{'=' * 70}")
print("RESUMEN DE EVALUACIÓN - PIPELINE LLM")
print(f"{'=' * 70}")
df_eval_llm = pd.DataFrame(resultados_eval_llm)
df_eval_llm.to_csv("evaluacion_transformer_llm.csv", index=False)
print(df_eval_llm.to_string(index=False))
print(f"\nPromedios:")
print(f"  Recall:    {df_eval_llm['Recall'].mean():.1%}")
print(f"  Precision: {df_eval_llm['Precision'].mean():.1%}")
print(f"  F1-Score:  {df_eval_llm['F1'].mean():.1%}")

# %% [markdown]
# ### Comparación entre los 3 pipelines

# %%
print(f"\n{'=' * 80}")
print("COMPARACIÓN DE PIPELINES")
print(f"{'=' * 80}")

comparacion = pd.DataFrame({
    'Pipeline': ['Sin promediar', 'Promediado (0.7/0.3)', 'LLM (Gemini)'],
    'Recall': [
        df_eval['Recall'].mean(),
        df_eval_2['Recall'].mean(),
        df_eval_llm['Recall'].mean()
    ],
    'Precision': [
        df_eval['Precision'].mean(),
        df_eval_2['Precision'].mean(),
        df_eval_llm['Precision'].mean()
    ],
    'F1': [
        df_eval['F1'].mean(),
        df_eval_2['F1'].mean(),
        df_eval_llm['F1'].mean()
    ]
})

print(comparacion.to_string(index=False))

# %% [markdown]
# ### Detalle de parámetros inferidos por el LLM

# %%
# Tabla resumen de lo que decidió el LLM para cada usuario
params_llm = pd.DataFrame([{
    'Usuario': r['nombre'],
    'Tipo Perfil': r['tipo_perfil'],
    'w_query': r['weights'][0],
    'w_historial': r['weights'][1],
    'Dirección': r['direccion'],
    'Query reescrita': r['query_reescrita'][:60] + '...'
} for r in resultados_llm])

print(params_llm.to_string(index=False))
