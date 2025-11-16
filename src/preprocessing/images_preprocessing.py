"""
Preprocesamiento de imágenes MRI para clasificación de Alzheimer.

Este módulo incluye:
- Normalización de intensidad
- Redimensionamiento
- Data augmentation
- División en train/val/test
- Conversión a tensores PyTorch
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd


class AlzheimerImageDataset(Dataset):
    """Dataset personalizado para imágenes de Alzheimer."""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        """
        Args:
            image_paths: Lista de rutas a las imágenes
            labels: Lista de etiquetas (0: NonDemented, 1: VeryMildDemented, 2: MildDemented, 3: ModerateDemented)
            transform: Transformaciones a aplicar
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Normaliza la intensidad de la imagen a rango [0, 1].
    
    Args:
        image: Imagen como array numpy
        
    Returns:
        Imagen normalizada
    """
    # Convertir a float
    image = image.astype(np.float32)
    
    # Normalizar a [0, 1]
    if image.max() > 0:
        image = image / 255.0
    
    return image


def get_augmentation_transforms(image_size: int = 224, is_training: bool = True):
    """
    Obtiene las transformaciones de data augmentation.
    
    Args:
        image_size: Tamaño objetivo de la imagen (224x224 o 128x128)
        is_training: Si es True, aplica data augmentation; si es False, solo normalización
        
    Returns:
        Objeto transforms de torchvision
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),  # Rotación ±10°
            transforms.ColorJitter(brightness=0.2),  # Variación de brillo
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # Zoom ligero
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])  # Normalización ImageNet
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])


def load_images_from_directory(data_dir: Path) -> Tuple[list, list]:
    """
    Carga todas las imágenes y sus etiquetas desde el directorio.
    
    Args:
        data_dir: Directorio raíz con subcarpetas por clase
        
    Returns:
        Tupla (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Mapeo de clases
    class_mapping = {
        'NonDemented': 0,
        'VeryMildDemented': 1,
        'MildDemented': 2,
        'ModerateDemented': 3
    }
    
    # Recorrer subcarpetas
    for class_name, label in class_mapping.items():
        class_dir = data_dir / class_name
        if class_dir.exists():
            for img_file in class_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                labels.append(label)
    
    return image_paths, labels


def split_dataset(image_paths: list, labels: list, 
                 train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, 
                 test_ratio: float = 0.15,
                 random_state: int = 42) -> Tuple[list, list, list, list, list, list]:
    """
    Divide el dataset en train/val/test con estratificación.
    
    Args:
        image_paths: Lista de rutas a imágenes
        labels: Lista de etiquetas
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Primero dividir en train y temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # Luego dividir temp en val y test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_info(X_train: list, X_val: list, X_test: list,
                    y_train: list, y_val: list, y_test: list,
                    output_dir: Path):
    """
    Guarda información sobre la división del dataset.
    
    Args:
        X_train, X_val, X_test: Listas de rutas de imágenes
        y_train, y_val, y_test: Listas de etiquetas
        output_dir: Directorio de salida
    """
    # Crear DataFrame con información
    train_df = pd.DataFrame({
        'image_path': X_train,
        'label': y_train
    })
    val_df = pd.DataFrame({
        'image_path': X_val,
        'label': y_val
    })
    test_df = pd.DataFrame({
        'image_path': X_test,
        'label': y_test
    })
    
    # Guardar CSVs
    train_df.to_csv(output_dir / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'test_split.csv', index=False)
    
    # Guardar resumen
    summary = {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test),
        'total': len(X_train) + len(X_val) + len(X_test)
    }
    
    import json
    with open(output_dir / 'split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Información de división guardada en {output_dir}")
    print(f"  Train: {summary['train']}, Val: {summary['val']}, Test: {summary['test']}")


def preprocess_images(
    raw_images_dir: Path,
    output_dir: Path,
    image_size: int = 224,
    use_augmented: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
):
    """
    Pipeline completo de preprocesamiento de imágenes.
    
    Args:
        raw_images_dir: Directorio con imágenes raw
        output_dir: Directorio de salida para imágenes procesadas
        image_size: Tamaño objetivo (224 o 128)
        use_augmented: Si True, usa AugmentedAlzheimerDataset; si False, OriginalDataset
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        random_state: Semilla para reproducibilidad
    """
    print("=" * 60)
    print("PREPROCESAMIENTO DE IMÁGENES MRI")
    print("=" * 60)
    
    # Seleccionar dataset
    if use_augmented:
        dataset_name = "AugmentedAlzheimerDataset"
    else:
        dataset_name = "OriginalDataset"
    
    data_dir = raw_images_dir / dataset_name
    print(f"\n📁 Cargando imágenes desde: {data_dir}")
    
    if not data_dir.exists():
        raise ValueError(f"Directorio no encontrado: {data_dir}")
    
    # Cargar imágenes
    print("\n🔄 Cargando imágenes...")
    image_paths, labels = load_images_from_directory(data_dir)
    print(f"✓ Total de imágenes cargadas: {len(image_paths)}")
    
    # Mostrar distribución de clases
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    print("\n📊 Distribución de clases:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} imágenes ({count/len(labels)*100:.1f}%)")
    
    # Dividir dataset
    print("\n🔄 Dividiendo dataset en train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        image_paths, labels, train_ratio, val_ratio, test_ratio, random_state
    )
    
    # Crear directorios de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # Guardar información de división
    save_split_info(X_train, X_val, X_test, y_train, y_val, y_test, splits_dir)
    
    # Crear datasets PyTorch
    print("\n🔄 Creando datasets PyTorch...")
    train_dataset = AlzheimerImageDataset(
        X_train, y_train, 
        transform=get_augmentation_transforms(image_size, is_training=True)
    )
    val_dataset = AlzheimerImageDataset(
        X_val, y_val,
        transform=get_augmentation_transforms(image_size, is_training=False)
    )
    test_dataset = AlzheimerImageDataset(
        X_test, y_test,
        transform=get_augmentation_transforms(image_size, is_training=False)
    )
    
    # Guardar datasets (solo metadatos, las imágenes se cargan on-the-fly)
    print("\n💾 Guardando configuración de datasets...")
    torch.save({
        'train_paths': X_train,
        'val_paths': X_val,
        'test_paths': X_test,
        'train_labels': y_train,
        'val_labels': y_val,
        'test_labels': y_test,
        'image_size': image_size,
        'class_names': class_names
    }, output_dir / 'dataset_config.pt')
    
    print(f"\n✅ Preprocesamiento completado!")
    print(f"   Datasets guardados en: {output_dir}")
    print(f"   Configuración guardada en: {output_dir / 'dataset_config.pt'}")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Configuración
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"
    OUTPUT_DIR = BASE_DIR / "data" / "processed" / "images_resized"
    
    # Ejecutar preprocesamiento
    train_ds, val_ds, test_ds = preprocess_images(
        raw_images_dir=RAW_IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        image_size=224,  # Puede cambiarse a 128
        use_augmented=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Train dataset: {len(train_ds)} imágenes")
    print(f"Val dataset: {len(val_ds)} imágenes")
    print(f"Test dataset: {len(test_ds)} imágenes")

