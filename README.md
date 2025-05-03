# 🧠 CC65 - Regresiones Concurrentes en Go

Este repositorio contiene la implementación desde cero de dos modelos fundamentales de aprendizaje automático:

- **Regresión Lineal**
- **Regresión Logística**

Cada uno en su versión **secuencial** y **concurrente**, usando `goroutines` de Go para mejorar el rendimiento en tareas de entrenamiento pesado.

---

## 📌 Objetivo

Explorar la diferencia de rendimiento entre modelos de aprendizaje automático implementados de forma secuencial y concurrente, como parte de la práctica PC2 del curso **CC65 - Programación Concurrente y Distribuida**.

---

## 🗂 Estructura del Proyecto

TB2_concurrente/
├── concurrent_emb/ # Entorno Python para preprocesamiento (opcional)
├── dataset/
│ ├── california_housing_train.csv
│ ├── california_housing_test.csv
│ ├── tabular-benchmark_train.csv
│ ├── tabular-benchmark_test.csv
│ └── get_dataset.ipynb # Notebook de preprocesamiento en Python
├── models/
│ └── regression/
│ ├── lineal_regression.go
│ ├── lineal_concurrent_regression.go
│ ├── logistic_regression.go
│ └── logistic_concurrent_regression.go
├── main.go # Ejecuta modelos y muestra resultados
└── go.mod


---

## 📊 Modelos Implementados

### 🔷 Regresión Lineal

- `LinearRegression`  
  → Modelo clásico entrenado de forma secuencial.

- `LinearRegressionConcurrent`  
  → Entrenamiento paralelo usando goroutines para dividir el cálculo de gradientes.

### 🔷 Regresión Logística

- `LogisticRegression`  
  → Modelo de clasificación binaria con descenso por gradiente.

- `LogisticRegressionConcurrent`  
  → Versión concurrente del entrenamiento para tareas distribuidas.

---

## 📁 Datasets

| Dataset                         | Tipo de tarea      | Fuente                                                                                 |
|---------------------------------|--------------------|----------------------------------------------------------------------------------------|
| California Housing              | Regresión          | [huggingface.co/gvlassis/california_housing](https://huggingface.co/datasets/gvlassis/california_housing) |
| Tabular Benchmark (Higgs)      | Clasificación      | [huggingface.co/polinaeterna/tabular-benchmark](https://huggingface.co/datasets/polinaeterna/tabular-benchmark) |

---

## ⚙️ Cómo Ejecutar

1. **Clonar el repositorio**

```bash
git clone https://github.com/tu_usuario/TB2_concurrente.git
cd TB2_concurrente
