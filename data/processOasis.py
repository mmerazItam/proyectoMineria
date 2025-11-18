import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd

# -----------------------------
# CONFIGURACIÓN DEL USUARIO
# -----------------------------
RAW_DIR = r"C:\Users\mmera\OneDrive\Escritorio\ProyectoMineria\proyectoMineria\data\raw\OASIS\OAS2_RAW_PART1\OAS2_RAW_PART1"
CLINICAL_CSV = r"C:\Users\mmera\OneDrive\Escritorio\ProyectoMineria\proyectoMineria\data\raw\OASIS\oasis_longitudinal.csv"

OUTPUT_DIR = r"C:\Users\mmera\OneDrive\Escritorio\ProyectoMineria\proyectoMineria\data\processed\OASIS_2D"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Crear carpetas finales
for d in ["CN", "MCI", "AD"]:
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# -----------------------------
# CARGAR DATOS CLÍNICOS
# -----------------------------
df = pd.read_csv(CLINICAL_CSV)

def obtener_cdr(subject_id):
    """Regresa el CDR del sujeto."""
    # El CSV tiene "Subject ID" y "MRI ID", buscar por "Subject ID" o "MRI ID"
    row = df[(df["Subject ID"] == subject_id) | (df["MRI ID"] == subject_id)]
    if row.empty:
        return None
    # Si hay múltiples filas, tomar la primera
    return row["CDR"].values[0]

def cdr_to_label(cdr):
    """Convierte CDR a categoría."""
    if cdr == 0:
        return "CN"
    elif cdr == 0.5:
        return "MCI"
    else:
        return "AD"

# -----------------------------
# PROCESAR IMAGEN
# -----------------------------
def procesar_mri(vol_path, subject_id):
    try:
        img = nib.load(vol_path).get_fdata()

        # Normalización 0–255
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)

        # Slice axial central (asegurarse de que sea 2D)
        if len(img.shape) == 3:
            axial = img[:, :, img.shape[2] // 2]
        elif len(img.shape) == 2:
            axial = img
        else:
            # Si tiene más dimensiones, tomar el slice medio
            axial = img[:, :, img.shape[2] // 2] if img.shape[2] > 1 else img[:, :, 0]
        
        # Asegurarse de que sea 2D
        if len(axial.shape) > 2:
            axial = axial[:, :, 0] if axial.shape[2] == 1 else axial.squeeze()

        # Obtener CDR
        cdr = obtener_cdr(subject_id)
        if cdr is None:
            print(f"[WARN] No se encontró CDR para {subject_id}")
            return

        label = cdr_to_label(cdr)

        # Guardar como PNG usando PIL (mejor para imágenes en escala de grises)
        # Asegurarse de que axial sea 2D y uint8
        if len(axial.shape) != 2:
            axial = axial.squeeze()
        if axial.dtype != np.uint8:
            axial = axial.astype(np.uint8)
        
        # Crear nombre único basado en el archivo original para evitar sobrescrituras
        file_name = os.path.basename(vol_path).replace('.hdr', '').replace('.nii', '').replace('.gz', '')
        out_path = os.path.join(OUTPUT_DIR, label, f"{subject_id}_{file_name}.png")
        
        img_pil = Image.fromarray(axial)  # PIL detecta automáticamente el modo para uint8
        img_pil.save(out_path)
        print(f"[OK] Guardado {out_path}")

    except Exception as e:
        print(f"[ERROR] procesando {vol_path}: {e}")

# -----------------------------
# RECORRER TODAS LAS CARPETAS
# -----------------------------
for folder in os.listdir(RAW_DIR):
    folder_path = os.path.join(RAW_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # ID del sujeto (ejemplo: OAS2_0001_MR1 -> OAS2_0001_MR1 o OAS2_0001)
    # El folder ya tiene el formato correcto (OAS2_0001_MR1)
    subject_id = folder

    # Buscar archivos .nii, .nii.gz, .hdr (Analyze)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz") or f.endswith(".hdr"):
                vol_path = os.path.join(root, f)

                # Para Analyze (.hdr/.img) nibabel carga desde .hdr automáticamente
                procesar_mri(vol_path, subject_id)

print("\n[OK] Procesamiento terminado.")
