# ARCEME Results Workflow

Dieser Workflow reduziert die Thesis-Auswertung auf drei wissenschaftlich
zentrale Schritte und hält Modellselektion, unabhängigen Test und explorative
Diagnostik strikt getrennt.

## Welche Daten wofür?

| Daten | Zulässige Verwendung |
|---|---|
| OOF-Predictions aller vier CV-Konfigurationen | Architektur-/Inputvergleich und Auswahl der Gewinnerkonfiguration |
| Predictions des Final-Refits auf dessen 146 Trainings-Cubes | **Keine Evaluation**; diese Predictions sind in-sample |
| Final-Refit-Predictions auf 27 Holdout-Cubes | Unabhängige, bestätigende Hauptperformance |
| OOF-Predictions der ausgewählten Konfiguration | Explorative Klima-, Land-Cover- und Ereignisanalyse mit höherem n |

Der „beste Fold“ wird niemals als finales Modell verwendet. Aus der CV wird
eine **Konfiguration** ausgewählt. Diese Konfiguration wird anschließend auf
allen 146 Development-Cubes für eine vorab aus der CV festgelegte Epochenzahl
neu trainiert. Erst danach wird der Holdout genau einmal ausgewertet.

## 0. Prediction-Tensoren exportieren

Die normale CV speichert die rekonstruierten Tensoren mit der aktuellen
Konfiguration nicht dauerhaft. Nach Abschluss eines CV-Runs werden deshalb die
besten Fold-Checkpoints jeweils einmal auf ihrem eigenen Validation-Fold
ausgeführt:

```bash
python -m evaluation.export_predictions oof \
  --run-dir /path/to/CV_SGConvLSTM_RGBN
```

Wiederhole dies für alle vier Konfigurationen. Es entstehen getrennte
Fold-Verzeichnisse und je Run ein:

```text
results/oof_predictions/
├── prediction_manifest.csv
├── export_metadata.json
├── fold_0/
│   ├── tensors/<cube_id>.zarr
│   └── metrics/
├── fold_1/
└── fold_2/
```

Jedes Prediction-Zarr enthält `pred`, `true`, `mask` und `base` mit Dimensionen
`[time, y, x]`. `base` ist die statische Last-Observation-/Persistence-Baseline.

## 1. CV-Modell- und Inputauswahl

```bash
python -m evaluation.results.cli cv \
  --manifest /path/to/CV_SG_RGBN/results/oof_predictions/prediction_manifest.csv \
  --manifest /path/to/CV_SG_INDICES/results/oof_predictions/prediction_manifest.csv \
  --manifest /path/to/CV_SGED_RGBN/results/oof_predictions/prediction_manifest.csv \
  --manifest /path/to/CV_SGED_INDICES/results/oof_predictions/prediction_manifest.csv \
  --output-dir /path/to/thesis_results/01_cv_selection
```

Die formale Auswahlregel ist dieselbe wie beim Training:

> höchste gleichgewichtete mittlere Fold-Macro-NNSE

Zusätzlich werden Cube-gepaarte Architektur-, Input- und Interaktionseffekte
mit nach Fold stratifiziertem Cube-Bootstrap berechnet. Der Code prüft vorab,
dass alle Konfigurationen dieselben OOF-Cubes, Folds, Targets, Masken und
Baselines verwenden.

Zentrale Outputs:

```text
01_cv_selection/
├── selection_recommendation.json
├── final_refit_config.yaml
├── tables/
│   ├── cv_configuration_summary.csv
│   ├── cv_fold_metrics.csv
│   ├── cv_pairwise_contrasts.csv
│   └── cv_factor_contrasts.csv
└── figures/
    ├── cv_model_selection.png
    └── cv_factor_contrasts.png
```

`final_refit_config.yaml` ist eine vollständige Kopie der Gewinnerkonfiguration,
nicht nur ein YAML-Snippet. `enabled`, `epochs`, `source_cv_run` und die
Tensor-/Metrik-Speicherung werden automatisch gesetzt.

## 2. Final Refit und unabhängige Holdout-Analyse

Final Refit mit der automatisch erzeugten Konfiguration:

```bash
python model/train.py \
  --config /path/to/thesis_results/01_cv_selection/final_refit_config.yaml
```

Danach den finalen Checkpoint auf den Holdout-Cubes exportieren:

```bash
python -m evaluation.export_predictions holdout \
  --run-dir /path/to/FinalRefit_Run
```

Und ausschließlich diese Predictions für die bestätigende Performance nutzen:

```bash
python -m evaluation.results.cli holdout \
  --manifest /path/to/FinalRefit_Run/results/holdout_predictions/prediction_manifest.csv \
  --output-dir /path/to/thesis_results/02_holdout
```

Analysiert werden:

1. Added Value gegenüber Persistence;
2. Fehlerentwicklung von Tag 5 bis Tag 30;
3. beobachtete versus vorhergesagte mittlere kNDVI-Reaktion.

Primärgrößen sind Macro-NNSE, MAE, Bias, MSE-Skill gegenüber Persistence,
Win-Rate und Cube-Bootstrap-Intervalle. Der Schritt-Skill wird als

```text
1 - mean_cube(MSE_model) / mean_cube(MSE_persistence)
```

berechnet. Das vermeidet instabile Mittelwerte von Cube-Ratios, wenn die
Persistence bei einzelnen Cubes nahezu fehlerfrei ist.

`(pred-base) - (true-base)` ist mathematisch identisch zu `pred-true`. Deshalb
werden `mse_delta` und `bias_delta` nicht als zusätzliche Dynamikmetriken
verkauft. Stattdessen werden die mittlere beobachtete/vorhergesagte Reaktion,
Day-30-Richtung und räumliche Response-Korrelation ausgewertet.

## 3. Bedingte Vorhersagbarkeit

Diese Analyse verwendet die OOF-Predictions der **ausgewählten**
CV-Konfiguration. Sie ist cross-validiert, aber explorativ und kein zweiter
unabhängiger Test.

```bash
python -m evaluation.results.cli conditional \
  --manifest /path/to/SELECTED_CV_RUN/results/oof_predictions/prediction_manifest.csv \
  --metadata-csv data_processing/data/train_test_split.csv \
  --event-feature tp_rollingmax \
  --output-dir /path/to/thesis_results/03_conditional
```

Der Code lädt `ESA_LC` aus den originalen Postprocessed-Cubes und aggregiert
zuerst pro Cube × Land-Cover-Gruppe. Pixel werden niemals als unabhängige
Replikate behandelt. Standardgruppen sind Forest, Shrub/Grass, Cropland und
Other Vegetation. Im Hauptplot erscheinen standardmäßig nur Gruppen mit
mindestens zehn Cubes.

Zwei Outcomes werden strikt getrennt:

- `observed_r30`: tatsächliche mittlere Vegetationsreaktion an Tag 30;
- `mae_gain = MAE_persistence - MAE_model`: Added Value des Modells.

Für Ereignisfeatures werden Tertile nur zur Visualisierung verwendet. Zusätzlich
wird ein explorativer standardisierter Bootstrap-Koeffizient ausgegeben. Beim
Forecast-Skill wird für absolute Response-Stärke, Zielvarianz und Datenabdeckung
kontrolliert.

Mit der aktuellen Metadatentabelle ist `tp_rollingmax` direkt nutzbar. Für
echte Dürredauer, Dürreintensität, Niederschlagsdauer und Transition Timing muss
die Tabelle später um meteorologisch berechnete Spalten ergänzt werden. Dann
können sie einfach wiederholt angegeben werden:

```bash
  --event-feature drought_severity \
  --event-feature drought_duration \
  --event-feature precipitation_intensity \
  --event-feature transition_gap_days
```

`start_drought_days` ist derzeit konstant und deshalb keine analysierbare
Dürredauer. `start_date` bis `end_date` beschreibt einen Disaster-Berichtszeitraum
und darf nicht als meteorologische Dauer interpretiert werden.

## Methodische Defaults

- 5.000 Cube-Bootstrap-Wiederholungen
- NNSE-Mindestcoverage: 15 % der möglichen vegetierten Target-Pixelzeiten
- mindestens 1.000 valide Target-Pixelzeiten
- minimale Zielvarianz: `1e-6`
- Greening/Browning-Schwelle für mittlere Response: `|ΔkNDVI| > 0.005`

Alle Werte sind CLI-Optionen. Änderungen sollten vor Sichtung der Ergebnisse
begründet und dokumentiert werden.
