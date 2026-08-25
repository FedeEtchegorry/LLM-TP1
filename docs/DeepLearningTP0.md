# Transformers
## Trabajo Práctico 1
### 73.69 Large Language Models - 2026

## 1. Objetivo

Se desea desarrollar una solución para un problema de predicción concreto mediante el uso de modelos de Inteligencia Artificial. Parte de dicha solución implica implementar un modelo de Transformers, con el fin de afianzar la comprensión de la arquitectura del modelo.

## 2. Sistema de Recomendación

Se cuenta con un conjunto de datos que contiene información histórica de eventos de búsqueda de usuarios en un e-commerce. Este conjunto posee información de compras, vistas de productos, interacciones con los filtros de la página, entre otras variables.

Se desea predecir el **Buy Through Rate (BTR)** en un e-commerce de supermercado con el fin de identificar los mejores productos y promocionarlos en otras áreas del e-commerce.

El BTR es una métrica de negocio que se define como la cantidad de productos comprados sobre la cantidad de productos impresos en la página de resultados de búsqueda. En otras palabras, el BTR indica la probabilidad de que un producto sea comprado por el usuario.

El conjunto de datos puede encontrarse en el archivo `supermarket_products.csv`. A continuación se definen los campos:

| Campo | Descripción |
|---|---|
| `title` | Nombre del producto en el catálogo. |
| `description` | Descripción del producto. |
| `price` | Precio del producto en dólares. |
| `category` | Categoría principal a la que pertenece el producto (ej. Dairy). |
| `timestamp` | Fecha y hora (UTC) en que ocurrió el evento de usuario. |
| `query_id` | Identificador de la búsqueda. |
| `filter_category` | Filtro de categoría durante la búsqueda del usuario. |
| `filter_price_min` | Precio mínimo en el filtro de búsqueda. |
| `filter_price_max` | Precio máximo en el filtro de búsqueda. |
| `filter_storage_type` | Filtro de tipo de almacenamiento durante la búsqueda (ej. Pantry). |
| `cart` | Indica si el producto fue añadido al carrito. |
| `bought` | Indica si el producto fue comprado. |
| `brand` | Marca del producto. |
| `package_size` | Tamaño del envase expresado con su unidad (ej. 10 oz). |
| `unit_of_measure` | Unidad de medida utilizada para el peso del envase. |
| `net_weight_oz` | Peso neto del producto en onzas. |
| `dimensions_in` | Dimensiones físicas del envase en pulgadas (largo × ancho × alto). |
| `storage_type` | Tipo de almacenamiento requerido por el producto (ej. Pantry). |
| `ingredients` | Lista de ingredientes del producto. |
| `allergens` | Alérgenos declarados en el producto. |
| `nutrition_score` | Puntuación nutricional del producto (valor numérico). |
| `country_of_origin` | País de origen del producto. |

### Ejercicio 1: Formulación del problema y EDA

Antes de entrenar el modelo, se deberá hacer un Exploratory Data Analysis para justificar con precisión:

1. Qué se predice: definir la variable objetivo.
2. Las características de la información provista: valores posibles, distribución, calidad de los datos, etc.
3. Qué features se utilizarán.
4. Qué preprocesamiento tendrá cada feature para ser tomada como input del modelo. *Sugerencia: investigar técnicas de codificación de variables, como por ejemplo one-hot encoding.*

### Ejercicio 2: Desarrollo del sistema

Deberán diseñar e implementar un sistema basado en modelos de IA que pueda resolver dicha tarea. El desarrollo deberá contar con al menos un modelo basado en la arquitectura Transformer. Deberán decidir en qué parte de la solución es pertinente y por qué.

Es importante tener en consideración los aspectos fundamentales de diseño:

1. **¿Cómo particiono mi conjunto de datos?** *Sugerencia: recordar train/valid/test split.*
2. **¿Qué experimentos realizo para definir la configuración del modelo que predice BTR?** (y cualquier otro modelo adicional que decida usar). *Sugerencia: definir una arquitectura base pequeña para iniciar la experimentación (Ej: d_model < 100) y asegurarse de que el costo computacional no sea limitante. Luego, ir aumentando la complejidad de la arquitectura dentro de las posibilidades de cómputo.*
3. **¿Cómo evalúo la performance del sistema?** *Sugerencia: Simplificar la evaluación del BTR con PR-AUC, ROC-AUC y métricas propias de modelos teniendo en cuenta overfitting y underfitting. No es necesario complejizar el análisis con una definición de umbral (threshold) para la predicción de BTR.*

> **Aclaración:** El foco del trabajo es la comprensión de la arquitectura Transformer y su aplicación a un problema concreto. Por lo que se evaluará la correcta justificación de las decisiones de diseño tomadas, la comparación de alternativas de los distintos módulos que podría tener la arquitectura y la realización del estudio de ablación correspondiente.

### Ejercicio 3: Personalización

Ejercicio teórico en el cual deberán explicar brevemente (una diapositiva, menos de cinco minutos) cómo modificarían la solución para incluir el factor de personalización de usuario al momento de definir el BTR.

## 3. Presentación

Presentación de aproximadamente 25-30 minutos que deberá contener:

- Explicación del problema a resolver.
- Decisiones de implementación correspondientes.
- Resultados obtenidos.
- Desafíos encontrados.
- Conclusiones.

## 4. Entrega

Adjuntar repositorio con README.md, hash del commit y presentación por Campus.
