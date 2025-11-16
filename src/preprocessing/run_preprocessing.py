"""
Script principal para ejecutar el pipeline completo de preprocesamiento.

Este script ejecuta:
1. Preprocesamiento de imágenes (MRI)
2. Preprocesamiento de datos tabulares
"""

from pathlib import Path
import sys

# Agregar el directorio raíz al path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.images_preprocessing import preprocess_images
from src.preprocessing.tabular_preprocessing import preprocess_tabular_data


def main():
    """Ejecuta el pipeline completo de preprocesamiento."""
    
    print("=" * 80)
    print("PIPELINE DE PREPROCESAMIENTO - FASE 1")
    print("=" * 80)
    print()
    
    # Configuración de rutas
    BASE_DIR = Path(__file__).resolve().parents[2]
    
    # ==================== PREPROCESAMIENTO DE IMÁGENES ====================
    print("\n" + "=" * 80)
    print("1. PREPROCESAMIENTO DE IMÁGENES")
    print("=" * 80)
    
    RAW_IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"
    PROCESSED_IMAGES_DIR = BASE_DIR / "data" / "processed" / "images_resized"
    
    try:
        train_ds, val_ds, test_ds = preprocess_images(
            raw_images_dir=RAW_IMAGES_DIR,
            output_dir=PROCESSED_IMAGES_DIR,
            image_size=224,  # Puede cambiarse a 128
            use_augmented=True,  # Usar dataset aumentado
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        print(f"\n✅ Preprocesamiento de imágenes completado exitosamente")
        print(f"   - Train: {len(train_ds)} imágenes")
        print(f"   - Val: {len(val_ds)} imágenes")
        print(f"   - Test: {len(test_ds)} imágenes")
        
    except Exception as e:
        print(f"\n❌ Error en preprocesamiento de imágenes: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== PREPROCESAMIENTO DE DATOS TABULARES ====================
    print("\n" + "=" * 80)
    print("2. PREPROCESAMIENTO DE DATOS TABULARES")
    print("=" * 80)
    
    INPUT_CSV = BASE_DIR / "data" / "raw" / "clinical" / "alzheimers_disease_data.csv"
    OUTPUT_CSV = BASE_DIR / "data" / "processed" / "tabular_clean" / "clinical_data_cleaned.csv"
    
    try:
        df_processed = preprocess_tabular_data(
            input_path=INPUT_CSV,
            output_path=OUTPUT_CSV,
            target_col='Diagnosis',
            imputation_method='knn',
            apply_feature_selection=True,
            top_k_features=50,
            task_type='classification'
        )
        
        print(f"\n✅ Preprocesamiento de datos tabulares completado exitosamente")
        print(f"   - Filas: {df_processed.shape[0]}")
        print(f"   - Columnas: {df_processed.shape[1]}")
        
    except Exception as e:
        print(f"\n❌ Error en preprocesamiento de datos tabulares: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== RESUMEN FINAL ====================
    print("\n" + "=" * 80)
    print("RESUMEN FINAL - FASE 1 COMPLETADA")
    print("=" * 80)
    print()
    print("✅ Entregables generados:")
    print(f"   1. Imágenes procesadas: {PROCESSED_IMAGES_DIR}")
    print(f"      - Datasets PyTorch (train/val/test)")
    print(f"      - Configuración guardada en dataset_config.pt")
    print(f"      - Información de división en splits/")
    print()
    print(f"   2. Datos tabulares limpios: {OUTPUT_CSV}")
    print(f"      - CSV con características procesadas")
    print(f"      - Información de preprocesamiento en preprocessing_info.txt")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

