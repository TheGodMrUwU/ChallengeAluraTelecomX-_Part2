# Telecom X · Parte 2 — Churn Modeling (Colab)

**Objetivo.** Tomar el dataset tratado de la Parte 1, **mapear** columnas al esquema analítico, **crear el target** `Cliente_cancelado`, entrenar y comparar modelos (LogReg / LogReg+SMOTE / RandomForest), **evaluar** con métricas robustas y **generar entregables** (CSV único + métricas + tablas + figuras).

---

## 📦 Datos de entrada

Archivo de la Parte 1: `telco_entregable_unico.csv` (o la ruta que uses).

**Columnas que reconoce el flujo** (se mapean automáticamente si existen):

| En tu CSV (Parte 1)                                               | Usado como (Parte 2)                                 |
| ----------------------------------------------------------------- | ---------------------------------------------------- |
| `Activo`                                                          | (si falta target) → `Cliente_cancelado = 1 - Activo` |
| `Ciudadano_65+`                                                   | `Mayor_de_65_años`                                   |
| `Dependientes`                                                    | `Tiene_Dependientes`                                 |
| `Meses_Contrato`                                                  | `Meses_Contratados`                                  |
| `Contrato`                                                        | `Tipo_de_Contrato`                                   |
| `Método_Pago`                                                     | `Metodo_Pago`                                        |
| `Cargos_Mensuales`                                                | `Cobro_Mensual`                                      |
| `Cargo_Total`                                                     | `Gasto_Total`                                        |
| *(opcionales)* `Servicio_Internet`, `Género`, `Servicio_Teléfono` | Se usan si están                                     |

> El pipeline también normaliza `Sí/No` → `1/0` (tolerante a acentos/espacios) y tipa numéricos.

---

## 🧱 Estructura (Colab por celdas)

1. **Instalación** (solo si falta): `imbalanced-learn` para SMOTE.
2. **Imports & Carga**: lee `telco_entregable_unico.csv`, crea `outputs/`.
3. **Mapeo & Selección**: renombra columnas, crea `Cliente_cancelado` si falta, castea tipos y elige columnas base (+ extras si hay).
4. **Encoding & Split**: mapea `Sí/No`, define `X/y`, **train/test** estratificado (80/20).
5. **Preprocesamiento & Modelos**: `ColumnTransformer` con imputer+scaler+OHE y tres pipelines:

   * `logreg_weighted`: Regresión Logística con `class_weight='balanced'`.
   * `logreg_smote`: LogReg + **SMOTE** (oversampling sintético) dentro del pipeline.
   * `rf_weighted`: RandomForest con `class_weight='balanced_subsample'`.
6. **Entrenamiento & Métricas**: `accuracy`, `precision`, `recall`, `f1`, **ROC AUC**, **PR AUC**. Selección del **mejor por ROC AUC** (empate → F1).
7. **Curvas & Matriz**: ROC, Precision–Recall y Matriz de Confusión del mejor.
8. **Exportables**:

   * `outputs/telco_parte2_unico.csv` (dataset analítico mapeado).
   * `outputs/metrics_parte2.json` (métricas del mejor).
9. **(Opcional) Importancias**: Top variables (RF) o coeficientes (LogReg).
10. **Celda final (reporting)**: guarda **tablas** y **figuras** clave en:

    * `outputs/tables/` (CSV): report de clasificación, comparativa de modelos, tasas por segmento, resumen numérico, matriz de correlación, scores de probas.
    * `outputs/figs/` (PNG): matriz de confusión, ROC, PR, tasas por segmento, correlación, importancias/coeficientes.

---

## ▶️ Cómo ejecutar

1. Abre el notebook en **Google Colab**.
2. Sube `telco_entregable_unico.csv` o monta Drive y ajusta la ruta en la **Celda 2**.
3. Corre **todas las celdas en orden**.
4. Revisa los entregables en `outputs/`.

---

## 📊 Métricas y criterios

* **Primarias:** ROC AUC (ranking de riesgo), PR AUC (robusta a desbalance), **F1** y **Recall** de la clase 1 (canceló).
* **Selección del mejor:** mayor **ROC AUC** (si empata, mejor **F1**).
* **Curvas:** ROC y Precision–Recall para inspección visual.
* **Matriz de Confusión:** errores por clase (útil para coste de falsos negativos).

---

## 🛠️ Personalización rápida

* **Umbral de decisión** (subir recall ↓ precisión):

  ```python
  y_prob = best_pipe.predict_proba(X_test)[:,1]
  umbral = 0.35  # ejemplo
  y_pred = (y_prob >= umbral).astype(int)
  ```
* **SMOTE**:

  * Cambia el balance interno: `SMOTE(random_state=42, sampling_strategy=0.7)`
* **RandomForest**:

  * Hiperparámetros clave: `n_estimators`, `max_depth`, `min_samples_leaf`.

---

## 🧩 Salidas esperadas

* `outputs/telco_parte2_unico.csv` — dataset final (target + features mapeadas).
* `outputs/metrics_parte2.json` — resumen del mejor modelo.
* `outputs/tables/` — informes en CSV (clasif., comparativas, tasas, correlación, resumen por clase, probas).
* `outputs/figs/` — PNG (confusión, ROC/PR, tasas por segmento, correlación, importancias).

---

## 🧯 Troubleshooting

* **`Input y contains NaN`**: el target tenía vacíos. La celda de normalización crea `Cliente_cancelado` desde `Activo` y **filtra filas sin target**. Recorre la Celda 3–4.
* **`KeyError` en columnas**: tu CSV no trae algún nombre esperado. Revisa la **tabla de mapeo** de arriba o ajusta el dict `alias`.
* **`imbalanced-learn` no encontrado**: corre la **Celda 1** (instalación).
* **One-Hot con categorías nuevas**: `handle_unknown='ignore'` ya lo cubre; asegúrate de no re-entrenar el OHE con test.

---

## 📐 Reproducibilidad

* `random_state=42` en splits/modelos.
* Preprocesamiento, SMOTE y modelo viven **en el mismo pipeline** (evita fugas de datos).
* Split **estratificado** para mantener proporciones.

---

## 📜 Licencia

Uso educativo/demostrativo. Ajusta a la política de tu organización si lo despliegas en producción.
