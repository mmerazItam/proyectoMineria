"""
Script para descargar e integrar imágenes adicionales del dataset de Kaggle
"Augmented Alzheimer MRI Dataset" al directorio de procesamiento.

Este script debe ejecutarse antes de los notebooks que procesan imágenes.
"""

import kagglehub
import shutil
from pathlib import Path
import os

def integrate_kaggle_images():
    """Descarga e integra imágenes de Kaggle al dataset."""
    
    # Download latest version
    print("📥 Descargando dataset adicional de Kaggle...")
    try:
        kaggle_path = kagglehub.dataset_download("uraninjo/augmented-alzheimer-mri-dataset")
        print(f"✓ Dataset descargado en: {kaggle_path}")
        
        # Configurar rutas
        BASE_DIR = Path(__file__).resolve().parent
        IMG_DIR = BASE_DIR / "data" / "processed" / "OASIS_2D"
        KAGGLE_DIR = Path(kaggle_path)
        
        # Crear directorios si no existen
        for cat in ["CN", "MCI", "AD"]:
            (IMG_DIR / cat).mkdir(parents=True, exist_ok=True)
        
        # Mapeo de clases de Kaggle a nuestras clases
        class_mapping = {
            'NonDemented': 'CN',
            'VeryMildDemented': 'MCI',
            'MildDemented': 'MCI',
            'ModerateDemented': 'AD'
        }
        
        copied_count = 0
        
        # Buscar imágenes en el directorio de Kaggle
        for kaggle_class, our_class in class_mapping.items():
            source_dir = KAGGLE_DIR / kaggle_class
            target_class_dir = IMG_DIR / our_class
            
            if not source_dir.exists():
                # Intentar buscar en subdirectorios comunes
                for subdir in KAGGLE_DIR.rglob(kaggle_class):
                    if subdir.is_dir():
                        source_dir = subdir
                        break
            
            if source_dir.exists() and source_dir.is_dir():
                # Buscar imágenes PNG, JPG, JPEG
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
                images = []
                for ext in image_extensions:
                    images.extend(list(source_dir.rglob(ext)))
                
                print(f"  📁 {kaggle_class} → {our_class}: {len(images)} imágenes encontradas")
                
                # Copiar imágenes
                for img_path in images:
                    # Generar nombre único
                    img_name = f"kaggle_{img_path.stem}{img_path.suffix}"
                    target_path = target_class_dir / img_name
                    
                    # Solo copiar si no existe
                    if not target_path.exists():
                        shutil.copy2(img_path, target_path)
                        copied_count += 1
                
                print(f"    ✓ {copied_count} imágenes nuevas copiadas a {our_class}")
            else:
                print(f"  ⚠️  No se encontró directorio para {kaggle_class}")
        
        print(f"\n✅ Integración completada: {copied_count} imágenes nuevas agregadas")
        return copied_count
        
    except Exception as e:
        print(f"⚠️  Error al descargar/integrar imágenes de Kaggle: {e}")
        print("   Continuando con las imágenes existentes...")
        return 0

if __name__ == "__main__":
    integrate_kaggle_images()

