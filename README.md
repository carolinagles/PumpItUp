# Pump It Up: Data Mining the Water Table

Competencia pública de clasificación multiclase en [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

**Score en leaderboard oficial → `0.8132` (LightGBM)**

---

## Objetivo

Predecir el **estado operativo de puntos de agua (pozos)** en Tanzania a partir de variables geográficas, técnicas y administrativas.

**Clases objetivo:**

| Clase | Descripción |
|---|---|
| `functional` | El pozo opera correctamente |
| `functional needs repair` | Opera, pero requiere reparación |
| `non functional` | El pozo no opera |



## Pipeline

### 1. Exploración y limpieza
- Dataset combinado: ~59,400 registros de entrenamiento, 40 variables
- Eliminación de columnas redundantes (variables en distintos niveles de agregación) y sin valor predictivo
- Imputación de nulos categóricos con categoría `'unknown'`; variables booleanas imputadas con la moda

### 2. Feature Engineering

**Variables temporales**
- `pump_age`: edad de la bomba (año de registro − año de construcción), tratando `0` como dato faltante
- `pump_age_bin`: discretización en rangos (0–5, 6–10, 11–20, 21–30, 30+)
- `year_recorded`, `month_recorded`: extraídos de `date_recorded`

**Variables geoespaciales**
- `dist_to_center`: distancia euclidiana al centro geográfico de Tanzania (Dodoma)
- `lat_lon_product`: interacción latitud × longitud para capturar patrones espaciales no lineales
- `region_pump_density`: densidad de bombas por región como proxy de infraestructura
- `gps_height_sq`: transformación cuadrática de la altitud para modelar efectos no lineales

**Variables binarias y transformaciones**
- `amount_tsh_bin`, `has_construction_year`, `has_population`: indicadores de presencia de datos relevantes
- `amount_tsh_log`: transformación logarítmica para reducir el sesgo de outliers extremos

### 3. Encoding
- **Target Encoding con smoothing** aplicado a variables de alta cardinalidad (`installer`, `funder`, `ward`, `lga`, etc.)
- Train y test separados **antes del encoding** para prevenir data leakage
- El encoder se ajusta exclusivamente sobre train y se aplica (`.transform`) sobre test

### 4. Modelos y optimización

| Modelo | Optimización | Accuracy CV | Score DrivenData |
|---|---|---|---|
| LightGBM | Optuna (50 trials) | 0.817 ± 0.002 | **0.8132** |
| CatBoost | Optuna (30 trials) | ~0.815 | 0.8106 |

- Validación cruzada estratificada (5-fold) para estimación robusta del rendimiento
- Early stopping en ambos modelos para controlar overfitting
- Subsampling de features y muestras como regularización adicional

### 5. Evaluación en validación (LightGBM)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| functional | — | 0.95 | — |
| functional needs repair | — | 0.55 | — |
| non functional | — | — | — |
| **Accuracy global** | | | **0.89** |

> La clase `functional needs repair` es la más difícil de detectar por ser la minoría con mayor solapamiento entre clases.

---

## Skills

```
Python · Pandas · NumPy · Scikit-learn
LightGBM · CatBoost · XGBoost
Optuna · category_encoders (TargetEncoder)
```

### Mejor resultado 

**LightGBM: 0.8132** | CatBoost: 0.8106
