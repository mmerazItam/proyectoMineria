# Preprocesamiento de Datos - Fase 1

Este directorio contiene los scripts para el preprocesamiento de datos del proyecto de minería de datos sobre Alzheimer.

## Estructura

- `images_preprocessing.py`: Preprocesamiento de imágenes MRI
- `tabular_preprocessing.py`: Preprocesamiento de datos tabulares clínicos
- `run_preprocessing.py`: Script principal que ejecuta todo el pipeline

## Requisitos

Las siguientes librerías deben estar instaladas:

```bash
pip install torch torchvision
pip install scikit-learn
pip install pandas numpy
pip install pillow
pip install tqdm
```

O instalar todas a la vez:

```bash
pip install torch torchvision scikit-learn pandas numpy pillow tqdm
```

## Uso

### Opción 1: Ejecutar el pipeline completo

```bash
python src/preprocessing/run_preprocessing.py
```

Este script ejecuta:
1. Preprocesamiento de imágenes (normalización, redimensionamiento, data augmentation, división train/val/test)
2. Preprocesamiento de datos tabulares (limpieza, encoding, escalamiento, feature engineering)

### Opción 2: Ejecutar módulos individuales

#### Preprocesamiento de imágenes

```python
from src.preprocessing.images_preprocessing import preprocess_images
from pathlib import Path

train_ds, val_ds, test_ds = preprocess_images(
    raw_images_dir=Path("data/raw/images"),
    output_dir=Path("data/processed/images_resized"),
    image_size=224,  # o 128
    use_augmented=True,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### Preprocesamiento de datos tabulares

```python
from src.preprocessing.tabular_preprocessing import preprocess_tabular_data
from pathlib import Path

df_processed = preprocess_tabular_data(
    input_path=Path("data/raw/clinical/alzheimers_disease_data.csv"),
    output_path=Path("data/processed/tabular_clean/clinical_data_cleaned.csv"),
    target_col='Diagnosis',
    imputation_method='knn',
    apply_feature_selection=True,
    top_k_features=50
)
```

## Entregables

Después de ejecutar el preprocesamiento, se generan los siguientes archivos:

### Imágenes procesadas (`data/processed/images_resized/`)

- `dataset_config.pt`: Configuración de los datasets PyTorch
- `splits/train_split.csv`: Lista de imágenes de entrenamiento
- `splits/val_split.csv`: Lista de imágenes de validación
- `splits/test_split.csv`: Lista de imágenes de test
- `splits/split_summary.json`: Resumen de la división

### Datos tabulares (`data/processed/tabular_clean/`)

- `clinical_data_cleaned.csv`: Datos clínicos preprocesados
- `preprocessing_info.txt`: Información sobre el preprocesamiento aplicado

## Características implementadas

### Preprocesamiento de imágenes

✅ Normalización de intensidad (0-1)  
✅ Redimensionamiento (224×224 o 128×128)  
✅ Data augmentation:
   - Rotación ±10°
   - Variación de brillo
   - Zoom ligero (0.9-1.1x)
✅ División estratificada en train/val/test  
✅ Conversión a tensores PyTorch  
✅ Normalización ImageNet

### Preprocesamiento de datos tabulares

✅ Limpieza de valores faltantes (KNN imputer o mediana)  
✅ Encoding:
   - One-hot para variables categóricas
✅ Escalamiento:
   - StandardScaler para características numéricas
✅ Ingeniería de características:
   - Edad² (edad al cuadrado)
   - Categorías de BMI
   - MAP (Mean Arterial Pressure)
   - Ratio Colesterol Total/HDL
   - Conteo de comorbilidades
   - Conteo de síntomas cognitivos
   - Interacción Edad × MMSE
   - Riesgo cardiovascular combinado
✅ Feature selection:
   - Eliminación de características altamente correlacionadas (>0.95)
   - Selección por Mutual Information
   - Top K características más relevantes

## Notas

- El preprocesamiento de imágenes usa el dataset aumentado por defecto (`AugmentedAlzheimerDataset`)
- Las imágenes se cargan on-the-fly durante el entrenamiento (no se guardan versiones procesadas en disco)
- El feature selection se puede desactivar si se desea usar todas las características
- Los datos tabulares se guardan en formato CSV para fácil inspección

