# Predicción de Churn de Clientes de Telecomunicaciones

**Autora:** Maria Fernanda Alvarez  
**Curso:** Introduction to Artificial Intelligence

---

## Descripción del proyecto

Este proyecto desarrolla un sistema de **Machine Learning** para predecir si un cliente de telecomunicaciones cancelará su servicio (**churn**).

A partir de variables como:

- tipo de contrato,
- cargos mensuales,
- tiempo de permanencia (tenure),
- servicios contratados,
- método de pago,

se entrenan varios modelos de clasificación para estimar la probabilidad de fuga de cada cliente.

Además, se incluye un **agente basado en reglas** que interpreta la predicción y genera recomendaciones de retención personalizadas para apoyar la toma de decisiones del negocio.

---

## Objetivos

1. Analizar y entender el comportamiento de churn en el dataset.
2. Entrenar y comparar distintos modelos de clasificación.
3. Seleccionar el mejor modelo según métricas de desempeño.
4. Explicar predicciones individuales de forma comprensible para negocio.

---

## ¿Qué hace el proyecto?

- Realiza análisis exploratorio de datos (EDA).
- Entrena 3 modelos:
  - Regresión Logística
  - Random Forest
  - Gradient Boosting
- Evalúa cada modelo con métricas clave:
  - Accuracy
  - F1-score
  - ROC-AUC
  - Classification report
- Guarda el mejor modelo entrenado.
- Ejecuta un agente de IA que:
  - calcula la probabilidad de churn de un cliente,
  - clasifica su nivel de riesgo,
  - identifica factores de riesgo,
  - propone acciones de retención.

---

## Tecnologías utilizadas

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib

---

## Instalación (una sola vez)

```bash
# 1) Clonar repositorio
git clone <repo-url>
cd customer-churn-prediction

# 2) Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows (PowerShell)

# 3) Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del proyecto

Todos los scripts están en la carpeta `src/`.

```bash
cd src
```

### 1) Análisis exploratorio (EDA)

```bash
python eda.py
```

- Descarga el dataset automáticamente.
- Genera visualizaciones en `reports/figures/`.

### 2) Entrenamiento de modelos

```bash
python train.py
```

- Entrena los 3 modelos.
- Selecciona el mejor desempeño.
- Guarda el modelo en `models/best_model.pkl`.

### 3) Evaluación

```bash
python evaluate.py
```

- Imprime métricas de todos los modelos.
- Permite comparar rendimiento de forma objetiva.

### 4) Agente de IA (explicación de predicción)

```bash
python agent.py
python agent.py --customer-id 42
```

- Muestra probabilidad de churn.
- Indica nivel de riesgo (bajo/medio/alto).
- Resume factores clave del caso.
- Sugiere recomendaciones personalizadas.

---

## Estructura del repositorio

```text
customer-churn-prediction/
├── data/               # Dataset (descarga automática)
├── models/             # Modelos guardados
├── reports/
│   ├── figures/        # Gráficas de EDA y entrenamiento
│   └── metrics.json    # Métricas de los modelos
├── src/
│   ├── data_loader.py  # Descarga y preprocesamiento
│   ├── eda.py          # Análisis exploratorio
│   ├── train.py        # Entrenamiento de modelos
│   ├── evaluate.py     # Evaluación
│   └── agent.py        # Agente explicativo + recomendaciones
├── requirements.txt
└── README.md
```

---

## Posibles mejoras futuras

- Ajuste de hiperparámetros (GridSearch/RandomSearch).
- Validación cruzada más robusta.
- Interpretabilidad con SHAP o LIME.
- Despliegue como API o dashboard interactivo.

---

## Uso de IA

Se utilizó **Claude (Anthropic)** como apoyo para generación de código, depuración y documentación.  
Todo el trabajo fue revisado y validado por la autora.
