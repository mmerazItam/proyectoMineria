import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd

# -----------------------------
# CONFIGURACIÓN DEL USUARIO
# -----------------------------
RAW_DIR = r"C:\Users\mmera\OneDrive\Escritorio\ProyectoMineria\proyectoMineria\data\raw\OASIS\OAS2_RAW_PART2\OAS2_RAW_PART2"
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
    """
    Busca el CDR usando 'Subject ID' o 'MRI ID'.
    subject_id aquí es algo como 'OAS2_0001_MR1', así que tomaremos solo 'OAS2_0001'.
    """
    subj = subject_id.split("_")[0] + "_" + subject_id.split("_")[1]  # Ejemplo: OAS2_0001
    
    row = df[(df["Subject ID"] == subj) | (df["MRI ID"] == subj)]
    if row.empty:
        return None
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
# PROCESAR IMAGEN EN CORONAL
# -----------------------------
def procesar_mri(vol_path, subject_id):
    try:
        img = nib.load(vol_path).get_fdata()

        # Normalización 0–255
        img = (img - img.min()) / (np.ptp(img))  # ptp = max-min
        img = (img * 255).astype(np.uint8)

        # -----------------------------
        # CORONAL = eje Y medio
        # img[:, Y, :]
        # -----------------------------
        if len(img.shape) == 3:
            coronal = img[:, img.shape[1] // 2, :]
        else:
            # Si no es 3D, intentamos tomar la 3ª dimensión
            coronal = img[:, img.shape[1] // 2, :]

        # Asegurar que sea 2D
        if len(coronal.shape) != 2:
            coronal = np.squeeze(coronal)

        # Obtener CDR
        cdr = obtener_cdr(subject_id)
        if cdr is None:
            print(f"[WARN] No se encontró CDR para {subject_id}")
            return

        label = cdr_to_label(cdr)

        # Nombre del archivo limpio
        file_name = os.path.basename(vol_path).replace('.hdr', '').replace('.nii', '').replace('.gz', '')
        out_path = os.path.join(OUTPUT_DIR, label, f"{subject_id}_{file_name}_coronal.png")

        # Guardar PNG
        img_pil = Image.fromarray(coronal)
        img_pil.save(out_path)

        print(f"[OK] CORONAL guardado: {out_path}")

    except Exception as e:
        print(f"[ERROR] procesando {vol_path}: {e}")

# -----------------------------
# RECORRER TODAS LAS CARPETAS
# -----------------------------
for folder in os.listdir(RAW_DIR):
    folder_path = os.path.join(RAW_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    subject_id = folder   # Ejemplo: OAS2_0001_MR1

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz") or f.endswith(".hdr"):
                vol_path = os.path.join(root, f)
                procesar_mri(vol_path, subject_id)

print("\n[OK] Procesamiento terminado (CORONAL).")
