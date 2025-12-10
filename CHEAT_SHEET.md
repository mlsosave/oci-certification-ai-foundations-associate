# 📋 CHEAT SHEET - OCI 2025 AI Foundations


## 🔑 JERARQUÍA AI → ML → DL → GenAI

```
AI (Artificial Intelligence) - MÁS AMPLIO
  ↓ Máquinas imitan inteligencia humana
  ↓ Ejemplo: Auto autónomo
  │
  ├─ ML (Machine Learning) - SUBCONJUNTO
  │    ↓ Aprende de datos sin programación explícita
  │    ↓ Ejemplo: Filtro spam
  │    │
  │    ├─ DL (Deep Learning) - SUBCONJUNTO
  │    │    ↓ Redes neuronales multicapa
  │    │    ↓ Ejemplo: Reconocimiento facial
  │    │    │
  │    │    └─ GenAI (Generative AI) - SUBCONJUNTO
  │    │         ↓ Crea contenido NUEVO
  │    │         ↓ Ejemplo: ChatGPT, DALL-E
```

**Diferencia clave:**
- **ML:** Clasifica/Predice (Datos + Etiquetas → Etiqueta)
- **GenAI:** Genera/Crea (Datos sin etiquetas → Contenido nuevo)

---

## 📊 MACHINE LEARNING - 3 TIPOS

| Tipo | Datos | Output | Algoritmo | Ejemplo | Aplicación |
|------|-------|--------|-----------|---------|------------|
| **SUPERVISED** | Etiquetados | Predicción | Linear Reg, Logistic Reg | Precio casa, Spam | Enfermedad, Clima, Crédito |
| **UNSUPERVISED** | Sin etiquetas | Clusters | K-Means | Segmentación | Fraude, Marketing, Outliers |
| **REINFORCEMENT** | Recompensas | Decisión | Q-Learning | Auto autónomo | Robots, Juegos |

### SUPERVISED - Subtipos

| | Regression | Classification |
|---|------------|----------------|
| **Output** | Número continuo | Categoría |
| **Ejemplo** | $250,000 | "Spam" |
| **Algoritmo** | Linear Regression | Logistic Regression |
| **Función** | `y = wx + b` | Sigmoid (S-curve) |
| **Loss** | `(Pred - Real)²` | Cross-entropy |

**Regression:** Predice NÚMEROS (precio, temperatura, edad)  
**Classification:** Predice CLASES (spam/no spam, gato/perro)

### SUPERVISED - Proceso ML

```
1. LOAD DATA → iris.csv
2. SPLIT → X (features) + y (labels)
3. TRAIN/TEST SPLIT → 70%/30%
4. STANDARDIZE → Mean=0, SD=1
5. TRAIN → model.fit(X_train, y_train)
6. EVALUATE → accuracy_score
7. PREDICT → model.predict(new_data)
```

**Fórmulas:**
```
Linear Regression:  f(x) = wx + b
Loss:               (Predicción - Real)²
Accuracy:           Correctas / Total
Sigmoid:            σ(x) = 1/(1+e^-x)  [Output: 0-1]
```

### UNSUPERVISED - Clustering

**Pasos:**
1. Preparar datos (normalizar, escalar)
2. Crear matriz similitud (Euclidean, Cosine, Jaccard)
3. Ejecutar algoritmo (K-Means, Hierarchical, DBSCAN)
4. Interpretar clusters

**Métricas similitud:** 0 (diferentes) → 1 (idénticos)

**Aplicaciones:**
- Segmentación mercado
- Detección fraude (outliers)
- Sistemas recomendación

### REINFORCEMENT LEARNING

| Componente | Definición | Ejemplo (Auto autónomo) |
|------------|------------|-------------------------|
| **Agent** | Quien aprende | El auto |
| **Environment** | Mundo de operación | Carretera |
| **State** | Situación actual | Cámara muestra camino |
| **Action** | Decisión | Girar izq/der/recto |
| **Reward** | Retroalimentación | +10 (bien), -100 (mal) |
| **Policy** | Estrategia | Reglas conducción |

**Proceso:** State → Action → Reward → Learning → Optimal Policy

---

## 🧠 DEEP LEARNING

### ANN (Artificial Neural Network)

**Componentes:**
```
Input Layer (obligatoria)
    ↓
Hidden Layers (opcionales, múltiples)
    ↓
Output Layer (obligatoria)
```

| Componente | Función | Ejemplo |
|------------|---------|---------|
| **Neuron** | Suma ponderada + activación | σ(Σ(w·x) + b) |
| **Weight** | Importancia conexión | 0.5, 0.9 |
| **Bias** | Flexibilidad | Desplazar función |
| **Activation** | Decide si activa | ReLU, Sigmoid, Tanh |
| **Backpropagation** | Ajusta pesos | Reduce error |

**Activaciones:**
- **ReLU:** `f(x) = max(0, x)` [Capas ocultas]
- **Sigmoid:** `σ(x) = 1/(1+e^-x)` [Clasificación binaria]
- **Softmax:** Normaliza probabilidades [Multi-clase]

### Arquitecturas DL

| Arquitectura | Datos | Procesa | Memoria | Usa Para |
|--------------|-------|---------|---------|----------|
| **FNN** | Tabulares | Directo | No | Predicción simple |
| **CNN** | Imágenes | Paralelo | No | Clasificar, Detectar objetos |
| **RNN** | Secuencias | Secuencial | Corto | Series tiempo cortas |
| **LSTM** | Secuencias | Secuencial | Largo | Traducción, NLP |
| **Transformer** | Texto | Paralelo | Self-attention | LLMs (GPT, BERT) |
| **GAN** | Imágenes | Generativo | No | Crear imágenes realistas |
| **Autoencoder** | Cualquiera | Compresión | No | Reducción dimensionalidad |

**Cuándo usar:**
- 📸 Imágenes → **CNN**
- 🎵 Música/Texto secuencial → **RNN/LSTM**
- 💬 LLMs/Traducción → **Transformer**
- 🎨 Generar imágenes → **GAN**

### CNN (Convolutional Neural Network)

**Capas:**
```
Input → Convolutional → ReLU → Pooling → Flatten → Dense → Output
```

| Capa | Función |
|------|---------|
| **Convolutional** | Detecta features (filtros/kernels) |
| **ReLU** | Activación no-lineal |
| **Pooling** | Reduce dimensiones (Max/Avg) |
| **Flatten** | 2D → 1D |
| **Dense/FC** | Clasificación final |
| **Dropout** | Previene overfitting |

**Aplicaciones:** Clasificación imagen, Detección objetos, Segmentación, Reconocimiento facial

### RNN (Recurrent Neural Network)

**Tipos de arquitectura:**
```
One-to-One:    x → y           [FNN normal]
One-to-Many:   x → y,y,y       [Generar música]
Many-to-One:   x,x,x → y       [Análisis sentimiento]
Many-to-Many:  x,x,x → y,y,y   [Traducción]
```

**Problema:** Vanishing gradient (olvida secuencias largas)  
**Solución:** LSTM

### LSTM (Long Short-Term Memory)

**3 Gates (Compuertas):**
1. **Forget Gate:** ¿Qué olvidar?
2. **Input Gate:** ¿Qué recordar?
3. **Output Gate:** ¿Qué mostrar?

**Ventaja:** Memoria a LARGO plazo (soluciona vanishing gradient)

---

## ✨ GENERATIVE AI & LLMs

### LLM (Large Language Model)

**Definición:** Modelo probabilístico que predice siguiente palabra

**Proceso:**
```
Input: "They sent me a"
  ↓
LLM calcula probabilidades:
  dog: 0.45
  cat: 0.25
  lion: 0.03
  ↓
Selecciona palabra con P más alta: "dog"
  ↓
Input: "They sent me a dog"
  ↓
LLM: EOS (End of Sentence): 0.85
  ↓
Output final: "They sent me a dog"
```

**"Large" = Número de parámetros** (millones a billones)

**Capacidades:**
- Responder preguntas
- Razonar paso a paso
- Generar artículos
- Traducir idiomas

### TRANSFORMERS

**Arquitectura:**
```
Encoder (procesa input → embeddings)
Decoder (genera output desde embeddings)
```

| Tipo | Componentes | Output | Uso | Ejemplo |
|------|-------------|--------|-----|---------|
| **Encoder-only** | Solo Encoder | Embeddings | Búsqueda, clasificación | BERT |
| **Decoder-only** | Solo Decoder | Texto nuevo | Generación | GPT-4 |
| **Encoder-Decoder** | Ambos | Seq→Seq | Traducción | T5 |

**Self-Attention:** Ve TODA la oración simultáneamente (vs RNN secuencial)

**Ventajas vs RNN:**
- ✅ Procesa en paralelo (más rápido)
- ✅ Captura dependencias largo plazo
- ✅ Entiende contexto completo

### TOKENS & EMBEDDINGS

**Token:** Unidad básica de texto
- Palabra simple: 1 token ("hello")
- Palabra compleja: 2+ tokens ("friendship" = "friend"+"ship")

**Regla:** Texto simple ~1 token/palabra, Complejo ~2-3 tokens/palabra

**Embedding:** Texto → Vector de números
```
"perro" → [0.2, 0.7, 0.1, 0.5, ..., 0.3]
          (Vector de 384-1024 dimensiones)
```

**Uso:** Búsqueda semántica (por significado, no keywords)

### PROMPT ENGINEERING

| Técnica | Qué es | Cuándo |
|---------|--------|--------|
| **0-shot** | Sin ejemplos | Tarea simple |
| **Few-shot** | k ejemplos en prompt | Mejorar precisión |
| **Chain-of-thought** | Pedir razonamiento paso a paso | Matemáticas, lógica |

**In-context Learning:** Dar ejemplos sin cambiar modelo

### PERSONALIZACIÓN LLMs

| Método | Setup | Costo | Cuándo | Ventaja | Desventaja |
|--------|-------|-------|--------|---------|------------|
| **Prompting** | Minutos | $0 | Inicio rápido | Fácil, gratis | Limitado |
| **RAG** | Días | $$ | Datos cambian | Actualizado | Complejo |
| **Fine-tuning** | Semanas | $$$ | Dominio específico | Mejor desempeño | Requiere datos |

**RAG (Retrieval-Augmented Generation):**
```
Query → Busca docs relevantes → LLM(query + docs) → Respuesta fundamentada
```

**Fine-tuning:**
```
Modelo Pre-entrenado + Datos custom → Modelo Fine-tuned
```

**Hallucination:** Texto no fundamentado/incorrecto (NO hay solución 100%)

**RLHF:** Reinforcement Learning from Human Feedback (alinea modelo con preferencias humanas)

---

## 🌐 OCI LANGUAGE SERVICE (7 capacidades)

| # | Capacidad | Qué hace |
|---|-----------|----------|
| 1 | **Language Detection** | Detecta idioma (75 idiomas) |
| 2 | **Sentiment Analysis** | 3 niveles: Document, Aspect, Sentence |
| 3 | **NER** | Extrae entidades (14 tipos) |
| 4 | **Key Phrases** | Ideas importantes |
| 5 | **Text Classification** | 600+ categorías |
| 6 | **PII Detection** | Info personal sensible |
| 7 | **Translation** | Traduce entre idiomas |

**14 Tipos NER:**
```
PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY,
QUANTITY, PERCENT, EMAIL, URL, PHONE, PRODUCT,
EVENT, IP_ADDRESS
```

**3 Niveles Sentimiento:**
1. **Document-level:** General
2. **Aspect-based:** Por aspecto ("comida":+, "servicio":-)
3. **Sentence-level:** Por oración

**Modelos custom:** NER + Text Classification

---

## 🖼️ OCI VISION SERVICE

| Capacidad | Output | Uso |
|-----------|--------|-----|
| **Image Classification** | Tags + scores | "perro" (0.95) |
| **Object Detection** | Bounding boxes + labels | Car (x:100,y:50,w:80,h:60) |
| **OCR** | Texto extraído | Placas, letreros |

**Classification vs Detection:**
- **Classification:** Qué hay en imagen completa
- **Detection:** DÓNDE está cada objeto (coordenadas)

**Custom models:** Objetos personalizados, Clasificación personalizada

---

## 📄 OCI DOCUMENT UNDERSTANDING SERVICE

| Feature | Qué hace | Mejor para |
|---------|----------|------------|
| **OCR** | Extrae TODO el texto | Digitalizar docs |
| **Key-Value** | Pares clave-valor (13 campos) | Recibos, facturas |
| **Table** | Extrae tablas | Facturas con items |
| **Classification** | Clasifica en 10 tipos | Enrutar docs |

**10 Tipos docs:**
```
Invoice, Receipt, Resume, Tax Form, Bank Statement,
Legal, Insurance, Medical Record, Passport, Other
```

**13 Campos K-V:** merchant_name, transaction_date, total, tax, subtotal, etc.

**Formatos:** PDF, JPEG, PNG, TIFF

---

## 🎙️ OCI SPEECH SERVICE

| Feature | Descripción |
|---------|-------------|
| **Idiomas** | Inglés, Español, Portugués (3) |
| **Velocidad** | 2h audio → 5 min |
| **Normalización** | "calle uno dos tres" → "Calle 123" |
| **SRT** | Subtítulos para videos |
| **Confidence** | Por palabra + total |
| **Profanity** | 3 modos |
| **Batch** | Múltiples archivos |

**Profanity Filtering:**
- **Removing:** `****`
- **Masking:** `f***`
- **Tagging:** `<profanity>word</profanity>`

**Formato:** JSON, SRT

---

## 💻 OCI DATA SCIENCE

| Componente | Función |
|------------|---------|
| **Projects** | Contenedor organizacional |
| **Notebook Sessions** | JupyterLab IDE |
| **Conda Environments** | Gestión paquetes Python |
| **ADS SDK** | Librería Oracle (AutoML, explicación) |
| **Model Catalog** | Repositorio modelos (metadata, Git info) |
| **Model Deployments** | HTTP endpoints |
| **Jobs** | Tareas repetibles |

**Workflow:**
```
Project → Notebook → Entrenar → Model Catalog → Deploy → HTTP Endpoint
```

**3 Principios:** Accelerated, Collaborative, Enterprise Grade

---

## 🤖 OCI GENERATIVE AI SERVICE

### 3 Características

**1. Modelos Pre-entrenados:**

**Chat Models:**
| Modelo | Tokens Max | Costo | Uso |
|--------|------------|-------|-----|
| **Command-R-Plus** | 128K | 💰💰💰 | Docs largos, avanzado |
| **Command-R** | 16K | 💰 | General, económico |
| **Llama 3 70B** | 8K | 💰💰 | Open source |

**Embedding Models:**
- **Embed English:** Solo inglés
- **Embed Multilingual:** 100+ idiomas (cross-language search)

**2. Fine-tuning (T-Few):**
- Inserta nuevas capas
- Actualiza solo fracción de pesos
- 10-100x más rápido y barato

**3. Dedicated AI Clusters:**
- GPUs aisladas
- RDMA networking
- Para fine-tuning e inference

**Playground:** UI para probar modelos sin código → View Code → Copiar

---

## 🔍 ORACLE AI VECTOR SEARCH (DB 23ai)

**Tipo de dato:**
```sql
CREATE TABLE docs (
    id NUMBER,
    texto VARCHAR2(4000),
    embedding VECTOR(1024)  -- o solo VECTOR
);
```

**Funciones:**

| Función | Uso |
|---------|-----|
| `VECTOR_EMBEDDING()` | Generar embeddings |
| `VECTOR_DISTANCE(v1, v2, COSINE)` | Calcular similitud |

**Métricas distancia:** COSINE (default), EUCLIDEAN, DOT

**Índice:**
```sql
CREATE VECTOR INDEX idx ON table(col)
ORGANIZATION INMEMORY NEIGHBOR GRAPH  -- o NEIGHBOR PARTITIONS
DISTANCE COSINE
WITH TARGET ACCURACY 95;
```

**Búsqueda:**
```sql
SELECT * FROM docs
ORDER BY VECTOR_DISTANCE(embedding, :query_vec)
FETCH APPROXIMATE FIRST 5 ROWS ONLY
WITH TARGET ACCURACY 90;
```

**TARGET ACCURACY:** 80-99 (precisión vs velocidad)

---

## 💬 SELECT AI

**Sintaxis:**
```sql
-- Query en lenguaje natural
SELECT AI ¿Cuáles son las 10 películas más vistas?;

-- Ver SQL generado
SELECT AI SHOWSQL ¿Cuáles son las 10 películas más vistas?;
```

**AI Profile:**
```sql
DBMS_CLOUD_AI.CREATE_PROFILE(
  profile_name => 'mi_perfil',
  attributes => JSON_OBJECT(
    'provider' VALUE 'oci',
    'model' VALUE 'cohere.command-r'
  )
);
```

**Ventaja:** SQL sin saber SQL, datos seguros en tenancy

---

## 🖥️ GPUs & SUPERCLUSTERS

### GPUs NVIDIA

| GPU | Año | Arquitectura | Mejor para | Status |
|-----|-----|--------------|------------|--------|
| **A100** | 2020 | Ampere | Pequeño/mediano | ✅ Disponible |
| **H100** | 2022 | Hopper | LLMs gran escala | ✅ Disponible |
| **H200** | 2024 | Hopper+ | Modelos muy grandes | 📅 2025 |
| **B200** | 2025 | Blackwell | 4x performance H100 | 📅 2025 |
| **GB200** | 2025 | Grace Blackwell | Máximo (2 GPU + 2 CPU) | 📅 2025 |

**GPU vs CPU:**
- CPU: 8-64 cores, secuencial
- GPU: Miles de cores, paralelo masivo (10-100x más rápido ML)

### Superclusters

**Specs:**
- Hasta 100,000+ GPUs
- RDMA networking
- Latencia: ~6.5 µs (dentro bloque), ~20 µs (entre bloques)
- Red lossless (sin pérdida paquetes)
- 3-tier Clos fabric

**RDMA (Remote Direct Memory Access):**
- GPU → GPU directo (sin CPU)
- Ultra baja latencia
- Alto throughput

**Placement:** Workloads pequeños → 1 bloque (menor latencia)

---

## ⚖️ IA RESPONSABLE

**3 Principios = LEGAL + ÉTICA + ROBUSTA**

| Principio | Significa |
|-----------|-----------|
| **LEGAL** | Cumple leyes (GDPR, CCPA) |
| **ÉTICA** | Dignidad, libertad, igualdad, sin sesgos |
| **ROBUSTA** | Segura, transparente, explicable |

**Proceso:**
1. Gobernanza
2. Políticas y procedimientos
3. Implementación
4. Monitoreo continuo

**Roles:** Developers, Deployers, End Users

---

## 🔢 NÚMEROS CLAVE PARA MEMORIZAR

### OCI Language
- ✓ **75** idiomas
- ✓ **14** tipos entidades (NER)
- ✓ **3** niveles sentimiento
- ✓ **600+** categorías clasificación

### OCI Document Understanding
- ✓ **10** tipos documentos
- ✓ **13** campos key-value

### OCI Speech
- ✓ **3** idiomas (inglés, español, portugués)
- ✓ **3** modos filtrado obscenidades

### OCI Generative AI
- ✓ **128K** tokens (Command-R-Plus)
- ✓ **16K** tokens (Command-R)
- ✓ **8K** tokens (Llama 3)
- ✓ **100+** idiomas (Embed Multilingual)

### Oracle Database
- ✓ **23ai** tiene VECTOR datatype
- ✓ **ONNX** modelos cargables

---

## ⚡ DECISIÓN RÁPIDA: ¿QUÉ SERVICIO?

```
TENGO...

├─ TEXTO
│  └→ OCI Language (sentimiento, NER, clasificación, traducción)
│
├─ IMAGEN
│  └→ OCI Vision (clasificación, detección, OCR)
│
├─ DOCUMENTO PDF/Escaneado
│  └→ OCI Document Understanding (OCR, K-V, tablas, clasificación)
│
├─ AUDIO/VIDEO
│  └→ OCI Speech (transcripción, normalización, SRT)
│
├─ QUIERO ENTRENAR MODELO ML
│  └→ OCI Data Science (notebooks, catalog, deploy)
│
└─ QUIERO USAR LLM
   └→ OCI Generative AI (chat, embeddings, fine-tuning)
```

---

## 🎯 PREGUNTAS TIPO EXAMEN

### Tipo 1: ¿Qué servicio OCI para...?

| Tarea | Servicio |
|-------|----------|
| Analizar sentimiento reviews | **OCI Language** |
| Detectar objetos en foto | **OCI Vision** |
| Extraer datos de factura | **OCI Document Understanding** |
| Transcribir podcast | **OCI Speech** |
| Entrenar modelo custom | **OCI Data Science** |
| Chatbot | **OCI Generative AI** |

### Tipo 2: ¿Qué tipo ML para...?

| Tarea | Tipo |
|-------|------|
| Predecir precio casa | Supervised - **Regression** |
| Spam/No spam | Supervised - **Classification** |
| Agrupar clientes | Unsupervised - **Clustering** |
| Auto autónomo | **Reinforcement Learning** |

### Tipo 3: ¿Qué arquitectura DL para...?

| Tarea | Arquitectura |
|-------|--------------|
| Clasificar imágenes | **CNN** |
| Generar música | **RNN** |
| Traducir idiomas | **LSTM** o **Transformer** |
| Reconocer rostros | **CNN** |
| Chatbot/LLM | **Transformer** |

### Tipo 4: ¿Qué GPU para...?

| Workload | GPU |
|----------|-----|
| Pequeño/mediano | **A100** |
| LLMs gran escala | **H100** |
| Máximo rendimiento | **GB200** |

---

## 📝 COMPARATIVAS CRÍTICAS

### ML: Supervised vs Unsupervised

| | Supervised | Unsupervised |
|---|------------|--------------|
| **Datos** | Con etiquetas | Sin etiquetas |
| **Objetivo** | Predecir | Descubrir patrones |
| **Validación** | Accuracy conocido | Interpretación |
| **Ejemplo** | Spam detection | Customer clustering |

### DL: CNN vs RNN vs LSTM

| | CNN | RNN | LSTM |
|---|-----|-----|------|
| **Datos** | Imágenes | Secuencias | Secuencias |
| **Procesa** | Paralelo | Secuencial | Secuencial |
| **Memoria** | No | Corto plazo | Largo plazo |
| **Problema** | - | Vanishing gradient | - |
| **Usa para** | Clasificar imagen | Series tiempo | Traducción |

### LLM: Encoder vs Decoder

| | Encoder | Decoder |
|---|---------|---------|
| **Input** | Texto | Embeddings |
| **Output** | Embeddings | Texto nuevo |
| **Usa para** | Búsqueda, clasificación | Generación |
| **Ejemplo** | BERT | GPT-4 |

### GenAI: Prompt vs RAG vs Fine-tuning

| | Prompt | RAG | Fine-tuning |
|---|--------|-----|-------------|
| **Tiempo** | Minutos | Días | Semanas |
| **Costo** | $0 | $$ | $$$ |
| **Datos** | No requiere | Vector DB | Labeled dataset |
| **Actualiza** | Inmediato | Auto | Manual |
| **Mejor para** | Pruebas | Datos cambiantes | Dominio específico |

---

## 🎓 OCI DATA SCIENCE vs AI SERVICES

| OCI Data Science | OCI AI Services |
|------------------|-----------------|
| **Entrenar** modelos custom | **Usar** modelos pre-entrenados |
| Requiere expertise DS | Sin expertise necesaria |
| JupyterLab, Python | API calls |
| Flexible, personalizable | Rápido, fácil |
| Notebooks, Catalog, Deploy | Language, Vision, Speech, Document |

---

## 🔤 PALABRAS CLAVE POR SERVICIO

**OCI Language:**
```Sentiment • NER • PII • Classification • Translation • Key Phrases • 75 idiomas • 14 tipos • 3 niveles • 600+ categorías```

**OCI Vision:**
```Bounding Box • Tags • Confidence Score • OCR • Classification • Detection • Custom Models```

**OCI Document Understanding:**
```OCR • Key-Value • Tables • Classification • 10 tipos • 13 campos • Receipt • Invoice • Resume```

**OCI Speech:**
```Transcription • Normalization • SRT • Profanity • Confidence • 3 idiomas • Batch • JSON```

**OCI Data Science:**
```Projects • Notebooks • Conda • ADS SDK • Model Catalog • Deployments • Jobs • HTTP API```

**OCI Generative AI:**
```Chat Models • Embeddings • T-Few • Dedicated Clusters • Playground • Fine-tuning • RAG • Command-R • Llama 3```

**Oracle DB 23ai:**
```VECTOR datatype • VECTOR_DISTANCE() • VECTOR_EMBEDDING() • SELECT AI • ONNX • Target Accuracy```

---

## ✅ TÉRMINOS ESENCIALES (5 PALABRAS)

| Término | Definición |
|---------|------------|
| **AI** | Máquinas imitan inteligencia humana |
| **ML** | Aprende de datos sin programar |
| **DL** | Redes neuronales profundas multicapa |
| **GenAI** | Crea contenido completamente nuevo |
| **Supervised** | Aprende con datos etiquetados |
| **Unsupervised** | Agrupa datos sin etiquetas |
| **Reinforcement** | Aprende por recompensas y castigos |
| **Regression** | Predice número continuo exacto |
| **Classification** | Predice categoría o clase |
| **Clustering** | Agrupa datos similares juntos |
| **Overfitting** | Memoriza training, falla test |
| **Backpropagation** | Ajusta pesos reduciendo error |
| **CNN** | Red especializada en imágenes |
| **RNN** | Red secuencial memoria corta |
| **LSTM** | RNN con memoria largo plazo |
| **Transformer** | Arquitectura con self-attention paralelo |
| **Token** | Unidad básica texto LLM |
| **Embedding** | Texto convertido vector números |
| **Prompt** | Texto entrada al LLM |
| **Fine-tuning** | Entrenar modelo pre-entrenado dominio |
| **RAG** | Fundamentar respuestas en documentos |
| **Hallucination** | Texto no fundamentado incorrecto |
| **Inference** | Predecir con modelo entrenado |
| **RDMA** | GPU comunicación directa CPU-less |
| **NER** | Reconocimiento entidades nombradas texto |
| **OCR** | Reconocimiento óptico caracteres imagen |
| **Sigmoid** | Función S-curve output 0-1 |

---

## 🧮 FÓRMULAS

```
Linear Regression:       y = wx + b
Loss (MSE):             Σ(Predicción - Real)²
Accuracy:               Correctas / Total
Sigmoid:                σ(x) = 1/(1+e^-x)
ReLU:                   f(x) = max(0, x)
Standardization:        (x - μ) / σ
```

---

## 📚 COMPARATIVA COMPLETA OCI SERVICES

| Servicio | Input | Output | Pre-trained | Custom | API Access |
|----------|-------|--------|-------------|--------|------------|
| **OCI Language** | Texto | Análisis | ✅ 7 caps | ✅ NER, Class | ✅ |
| **OCI Vision** | Imagen | Tags/Boxes | ✅ | ✅ | ✅ |
| **OCI Document** | PDF/Img | K-V/Tablas | ✅ | ❌ | ✅ |
| **OCI Speech** | Audio | Texto | ✅ | ❌ | ✅ |
| **OCI Gen AI** | Prompt | Texto/Embed | ✅ Cohere/Llama | ✅ T-Few | ✅ |
| **OCI Data Science** | Datos | Modelo | ❌ | ✅ Custom | ✅ |

---

## 🎯 ESTRATEGIA EXAMEN

### ANTES
- ✓ Memoriza números (75, 14, 3, 10, 600)
- ✓ Repasa tablas comparativas
- ✓ Duerme bien

### DURANTE
- ✓ Lee pregunta 2 veces
- ✓ Elimina 2 incorrectas primero
- ✓ Entre 2 finales, elige MÁS ESPECÍFICA
- ✓ Marca difíciles, revisa al final
- ✓ 90 seg/pregunta

### PATTERNS COMUNES

**"¿Cuál servicio OCI...?"**
→ Language/Vision/Document/Speech/Data Science/Gen AI

**"¿Qué tipo ML...?"**
→ Supervisado/No supervisado/Refuerzo

**"¿Qué arquitectura DL...?"**
→ CNN/RNN/LSTM/Transformer

**"¿Predecir número...?"**
→ Regression

**"¿Predecir categoría...?"**
→ Classification

**"¿Sin etiquetas...?"**
→ Unsupervised

**"¿Recompensas...?"**
→ Reinforcement

---

## 💡 REGLAS MNEMOTÉCNICAS

### Servicios OCI: **"LVSD-G"**
```
L → Language    (Texto)
V → Vision      (Imagen)
S → Speech      (Audio)
D → Document    (PDF)
G → Gen AI      (LLM)
```

### Tipos ML: **"SUuR"**
```
S → Supervised      (CON etiquetas)
U → Unsupervised    (SIN etiquetas)
R → Reinforcement   (Recompensas)
```

### Arquitecturas DL: **"CRL-T"**
```
C → CNN         (Imágenes)
R → RNN         (Secuencias cortas)
L → LSTM        (Secuencias largas)
T → Transformer (LLMs)
```

### IA Responsable: **"LER"**
```
L → Legal
E → Ética
R → Robusta
```

---

## 🎯 CHECKLIST PRE-EXAMEN (30 SEG)

- [ ] AI > ML > DL > GenAI (jerarquía)
- [ ] 3 tipos ML (Supervisado, No supervisado, Refuerzo)
- [ ] Regression (números) vs Classification (clases)
- [ ] CNN (imagen), RNN (seq corta), LSTM (seq larga), Transformer (LLM)
- [ ] OCI Language: 7 caps, 14 NER, 3 sentimiento, 75 idiomas
- [ ] OCI Vision: Classification vs Detection vs OCR
- [ ] OCI Document: 4 features, 10 tipos, 13 campos K-V
- [ ] OCI Speech: 3 idiomas, normalización, SRT
- [ ] OCI Gen AI: Command-R (16K), R-Plus (128K), Llama (8K)
- [ ] OCI Data Science: 7 componentes
- [ ] GPUs: A100 (pequeño), H100 (grande), GB200 (máximo)
- [ ] Vector Search: VECTOR type, VECTOR_DISTANCE()
- [ ] Select AI: lenguaje natural → SQL
- [ ] IA Responsable: Legal + Ética + Robusta
- [ ] T-Few: rápido, económico, fracción pesos

**Si ✓ todo → ¡LISTO! 🚀**

---

## 📊 RESUMEN ULTRA-CONDENSADO (60 SEG)

**JERARQUÍA:** AI > ML > DL > GenAI

**3 TIPOS ML:**
- Supervised (etiquetas) → Predice
- Unsupervised (sin etiquetas) → Agrupa
- Reinforcement (recompensas) → Decide

**2 SUBTIPOS SUPERVISED:**
- Regression → NÚMEROS
- Classification → CLASES

**4 ARQUITECTURAS DL:**
- CNN → Imágenes
- RNN → Secuencias cortas
- LSTM → Secuencias largas
- Transformer → LLMs

**SERVICIOS OCI:**
1. **Language** → Texto (7 caps: sentimiento, NER, clasificación, PII, traducción, frases, idioma)
2. **Vision** → Imagen (classification, detection, OCR)
3. **Document** → PDF (OCR, K-V, tablas, clasificación 10 tipos)
4. **Speech** → Audio (3 idiomas, normalización, SRT)
5. **Data Science** → ML custom (notebooks, catalog, deploy)
6. **Gen AI** → LLMs (R-Plus:128K, R:16K, Llama:8K, T-Few fine-tuning)

**ORACLE DB 23ai:**
- VECTOR datatype
- VECTOR_DISTANCE()
- Select AI (natural → SQL)

**GPUs:** A100 (pequeño) < H100 (grande) < GB200 (máximo)

**IA RESPONSABLE:** Legal + Ética + Robusta

**NÚMEROS:**
75 idiomas | 14 NER | 3 sentimiento | 10 tipos docs | 600 categorías | 128K tokens

---

## 🏆 RESPUESTAS DIRECTAS

**¿GenAI task?** → Crear contenido nuevo (poema, imagen, música)

**¿Extraer trends?** → Unsupervised ML

**¿Speech task?** → Speech-to-text

**¿NOT vision task?** → Reparar imágenes

**¿Auto autónomo?** → Reinforcement Learning

**¿Sentiment + traducción?** → NLP (OCI Language)

**¿GenAI vs Supervised?** → GenAI crea contenido, Supervised predice

**¿IA confiable?** → Legal + Ética + Robusta

**¿Spam detection?** → ML (Supervised Classification)

**¿Precio casa?** → Regression

**¿Predecir valores continuos?** → Linear Regression

**¿Aprende de outcomes?** → Reinforcement Learning

**¿NO requiere ML?** → Password validation

**¿Logistic Regression función?** → Sigmoidal (S-curve)

**¿Model training?** → Relación input-output

**¿Target variable?** → Salida/etiquetas deseadas

**¿Inference?** → Predecir nuevos datos

**¿NOT unsupervised app?** → Spam detection (es supervised)

**¿Recomendar shows?** → Supervised ML

**¿Loss function?** → Cuantifica error predicción

**¿Predecir precio casa?** → Regression

**¿Non-parametric?** → KNN

**¿Stock prices data type?** → Time series

**¿Memoria largo plazo?** → LSTM

**¿Feedback loop secuencial?** → RNN

**¿Suma ponderada + activación?** → Neuron

**¿Hidden layers ayudan?** → Aprenden features complejas

**¿Traducción arquitectura?** → Many-to-Many

**¿CNN propósito?** → Detectar patrones imágenes

**¿Hidden layer propósito?** → Procesar con weights y activations

**¿DL tipo data?** → Compleja no-interpretable (imágenes, audio)

**¿Generar música?** → RNN

**¿Reconocer rostros?** → CNN

**¿Clasificar objetos?** → DL

**¿Completar poema?** → RNN

**¿Auto detectar peatones?** → DL (AI aplicada)

**¿RNN limitación?** → Vanishing gradient (largo plazo)

**¿GenAI descripción?** → Crea nuevo contenido

**¿Fine-tuning innecesario?** → Sin adaptación tarea específica

**¿LLM impacto?** → Tamaño y parámetros

**¿NOT secuencia?** → Clasificación imágenes (usa CNN)

**¿In-context learning?** → Ejemplos en prompt

**¿Few-shot?** → Ejemplos explícitos en prompt

**¿Tokens rol?** → Unidades división texto

**¿Pre-training GenAI?** → Patrones datos no estructurados sin etiquetas

**¿LLM vs ML diferencia?** → LLM pre-entrenado corpus grande

**¿Fine-tuning objetivo?** → Ajustar parámetros dataset específico

**¿NOT OCI AI Service?** → Translator (está en Language)

**¿Model Deployments?** → Endpoints HTTP

**¿Superclusters ventaja?** → Performance y escalabilidad complejos

**¿Jobs OCI DS?** → Tareas ML repetibles

**¿Vector datatype?** → Oracle DB 23ai comparar docs

**¿GPU masivo scale?** → GB200

**¿Deploy modelo?** → OCI Data Science

**¿NOT AI Infrastructure?** → OCI Vault

**¿ONNX models?** → Cargar directo en DB

**¿Model Catalog propósito?** → Repositorio almacenar/rastrear modelos

**¿GPU pequeño/mediano?** → A100

**¿VECTOR store embeddings?** → VECTOR datatype

**¿Chat vs Embedding models?** → Chat genera texto, Embedding vectoriza

**¿Playground propósito?** → Probar modelos sin código

**¿Fine-tune LLM recurso?** → Dedicated AI Cluster

**¿Model endpoints propósito?** → Host fine-tuned para inference

**¿Select AI genera SQL?** → Conecta LLM, infiere intent, formula SQL

**¿T-Few reduce costo?** → Actualiza solo fracción pesos

**¿Select AI mejora?** → Natural language vs SQL

**¿Extraer tablas docs?** → OCI Vision (Document Understanding)

**¿Bounding box?** → Object Detection

**¿OCI Language capacidad?** → Text Sentiment Analysis

**¿NOT Speech idioma?** → Mandarín (solo inglés, español, portugués)

**¿Normalización mejora?** → Formatea números, fechas, URLs

**¿Retener + marcar obsceno?** → Tagging

**¿Extraer merchant, date?** → Key-value extraction

**¿Clasificar doc tipo?** → Document classification

**¿Detectar vehículos + placas?** → Object detection

**¿Categorizar artículos tópicos?** → Text classification

**¿Evaluar confianza palabra?** → Confidence scoring

**¿Subtítulos videos?** → SRT file support

---

## 🎯 CASO DE USO → SERVICIO

| Caso | Servicio OCI |
|------|--------------|
| Reviews de productos | Language (sentiment) |
| Moderar comentarios | Language (sentiment + PII) |
| Traducir docs | Language (translation) |
| Clasificar tickets | Language (classification) |
| Fotos de productos | Vision (classification) |
| Detector objetos almacén | Vision (detection custom) |
| Leer placas autos | Vision (OCR) |
| Facturas automáticas | Document Understanding (K-V) |
| Digitalizar archivos | Document Understanding (OCR) |
| Extraer formularios | Document Understanding (tables) |
| Llamadas soporte | Speech (transcription) |
| Subtítulos videos | Speech (SRT) |
| Chatbot cliente | Gen AI (chat models) |
| Búsqueda semántica | Gen AI (embeddings) |
| LLM dominio legal | Gen AI (T-Few fine-tuning) |
| Entrenar modelo custom | Data Science |
| Deploy modelo producción | Data Science (deployments) |

---

## ⚡ ÚLTIMO MINUTO (2 MIN ANTES EXAMEN)

**REPASA:**
1. AI > ML > DL > GenAI (jerarquía)
2. Supervised (etiquetas), Unsupervised (sin), Reinforcement (recompensas)
3. Regression (números), Classification (clases)
4. CNN (imagen), RNN (seq), LSTM (seq larga), Transformer (LLM)
5. OCI Language: 7, 14, 3, 75, 600
6. OCI Vision: Classification, Detection, OCR
7. OCI Document: OCR, K-V (13), Tables, Classification (10)
8. OCI Speech: 3 idiomas, Normalización, SRT
9. Command-R-Plus (128K), Command-R (16K), Llama 3 (8K)
10. A100 (pequeño), H100 (grande), GB200 (máximo)
11. VECTOR datatype, VECTOR_DISTANCE(), Select AI
12. Legal + Ética + Robusta

**RESPIRAR → LEER 2 VECES → ELIMINAR 2 → ELEGIR MEJOR → ¡ÉXITO! 🚀**

---

**Creado con 💙 | OCI 2025 AI Foundations Associate**  
**Última actualización:** Dic 2024
