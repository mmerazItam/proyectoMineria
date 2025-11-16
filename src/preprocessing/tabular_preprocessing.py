"""
Preprocesamiento de datos tabulares clínicos.

Este módulo incluye:
- Limpieza de valores faltantes (KNN imputer o mediana)
- Encoding (One-hot para categóricas)
- Escalamiento (StandardScaler)
- Ingeniería de características (IMC, edad², combinaciones)
- Feature selection (correlaciones, VIF, mutual information)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_clinical_data(data_path: Path) -> pd.DataFrame:
    """
    Carga los datos clínicos desde CSV.
    
    Args:
        data_path: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos
    """
    print(f"📁 Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Verifica valores faltantes en el dataset.
    
    Args:
        df: DataFrame a verificar
        
    Returns:
        Series con conteo de valores faltantes por columna
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    result = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    result = result[result['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(result) > 0:
        print("\n⚠️  Valores faltantes encontrados:")
        print(result)
    else:
        print("\n✓ No se encontraron valores faltantes")
    
    return missing


def impute_missing_values(df: pd.DataFrame, 
                          method: str = 'knn',
                          n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputa valores faltantes usando KNN o mediana.
    
    Args:
        df: DataFrame con valores faltantes
        method: 'knn' o 'median'
        n_neighbors: Número de vecinos para KNN
        
    Returns:
        DataFrame con valores imputados
    """
    df_imputed = df.copy()
    missing = df.isnull().sum()
    
    if missing.sum() == 0:
        print("✓ No hay valores faltantes para imputar")
        return df_imputed
    
    # Separar numéricas y categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Imputar numéricas
    numeric_missing = [col for col in numeric_cols if missing[col] > 0]
    if numeric_missing:
        print(f"\n🔄 Imputando {len(numeric_missing)} columnas numéricas usando {method}...")
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        else:  # median
            imputer = SimpleImputer(strategy='median')
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print("✓ Imputación numérica completada")
    
    # Imputar categóricas (moda)
    categorical_missing = [col for col in categorical_cols if missing[col] > 0]
    if categorical_missing:
        print(f"\n🔄 Imputando {len(categorical_missing)} columnas categóricas usando moda...")
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
        print("✓ Imputación categórica completada")
    
    return df_imputed


def encode_categorical_variables(df: pd.DataFrame, 
                                 categorical_cols: Optional[List[str]] = None,
                                 drop_first: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Aplica One-Hot Encoding a variables categóricas.
    
    Args:
        df: DataFrame con variables categóricas
        categorical_cols: Lista de columnas categóricas (si None, detecta automáticamente)
        drop_first: Si True, elimina la primera categoría (evita multicolinealidad)
        
    Returns:
        Tupla (DataFrame codificado, diccionario con mapeos)
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    if categorical_cols is None:
        # Detectar categóricas (object o int con pocos valores únicos)
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'int32']:
                unique_vals = df[col].nunique()
                if unique_vals <= 10:  # Considerar categórica si tiene <= 10 valores únicos
                    categorical_cols.append(col)
    
    # Filtrar solo las que realmente son categóricas (no binarias que ya están codificadas)
    # Variables binarias (0/1) no necesitan encoding
    true_categorical = []
    for col in categorical_cols:
        if df[col].dtype == 'object' or (df[col].nunique() > 2 and df[col].nunique() <= 10):
            true_categorical.append(col)
    
    if not true_categorical:
        print("✓ No hay variables categóricas que requieran encoding")
        return df_encoded, encoding_info
    
    print(f"\n🔄 Aplicando One-Hot Encoding a {len(true_categorical)} variables categóricas...")
    
    for col in true_categorical:
        if df[col].dtype == 'object':
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            encoding_info[col] = {
                'method': 'one_hot',
                'new_columns': dummies.columns.tolist()
            }
            print(f"  ✓ {col}: {len(dummies.columns)} nuevas columnas")
    
    print(f"✓ Encoding completado. Nuevas dimensiones: {df_encoded.shape}")
    return df_encoded, encoding_info


def create_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Crea nuevas características mediante ingeniería de características.
    
    Características creadas:
    - BMI ya existe, pero se puede crear categorías
    - edad² (edad al cuadrado)
    - Combinaciones relevantes según literatura biomédica
    
    Args:
        df: DataFrame original
        
    Returns:
        Tupla (DataFrame con nuevas características, lista de nombres de características creadas)
    """
    df_fe = df.copy()
    new_features = []
    
    print("\n🔧 Creando características de ingeniería...")
    
    # 1. Edad al cuadrado (relación no lineal con demencia)
    if 'Age' in df.columns:
        df_fe['Age_squared'] = df['Age'] ** 2
        new_features.append('Age_squared')
        print("  ✓ Age_squared creada")
    
    # 2. Categorías de BMI
    if 'BMI' in df.columns:
        df_fe['BMI_category'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 25, 30, np.inf],
            labels=[0, 1, 2, 3]  # Bajo peso, Normal, Sobrepeso, Obesidad
        ).astype(int)
        new_features.append('BMI_category')
        print("  ✓ BMI_category creada")
    
    # 3. Presión arterial media (MAP - Mean Arterial Pressure)
    if 'SystolicBP' in df.columns and 'DiastolicBP' in df.columns:
        df_fe['MAP'] = (2 * df['DiastolicBP'] + df['SystolicBP']) / 3
        new_features.append('MAP')
        print("  ✓ MAP (Mean Arterial Pressure) creada")
    
    # 4. Ratio Colesterol Total/HDL (indicador de riesgo cardiovascular)
    if 'CholesterolTotal' in df.columns and 'CholesterolHDL' in df.columns:
        df_fe['Cholesterol_ratio'] = df['CholesterolTotal'] / (df['CholesterolHDL'] + 1e-6)
        new_features.append('Cholesterol_ratio')
        print("  ✓ Cholesterol_ratio creada")
    
    # 5. Comorbilidades combinadas (índice de carga de enfermedad)
    comorbidity_cols = [
        'CardiovascularDisease', 'Diabetes', 'Depression', 
        'Hypertension', 'HeadInjury'
    ]
    existing_comorb = [col for col in comorbidity_cols if col in df.columns]
    if existing_comorb:
        df_fe['Comorbidity_count'] = df[existing_comorb].sum(axis=1)
        new_features.append('Comorbidity_count')
        print("  ✓ Comorbidity_count creada")
    
    # 6. Síntomas cognitivos combinados
    cognitive_symptoms = [
        'MemoryComplaints', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]
    existing_symptoms = [col for col in cognitive_symptoms if col in df.columns]
    if existing_symptoms:
        df_fe['Cognitive_symptoms_count'] = df[existing_symptoms].sum(axis=1)
        new_features.append('Cognitive_symptoms_count')
        print("  ✓ Cognitive_symptoms_count creada")
    
    # 7. Interacción: Edad × MMSE (deterioro cognitivo ajustado por edad)
    if 'Age' in df.columns and 'MMSE' in df.columns:
        df_fe['Age_MMSE_interaction'] = df['Age'] * df['MMSE']
        new_features.append('Age_MMSE_interaction')
        print("  ✓ Age_MMSE_interaction creada")
    
    # 8. Riesgo cardiovascular combinado
    if 'Hypertension' in df.columns and 'CardiovascularDisease' in df.columns:
        df_fe['Cardiovascular_risk'] = (
            df['Hypertension'].astype(int) + 
            df['CardiovascularDisease'].astype(int)
        )
        new_features.append('Cardiovascular_risk')
        print("  ✓ Cardiovascular_risk creada")
    
    print(f"✓ Ingeniería de características completada. {len(new_features)} nuevas características creadas")
    
    return df_fe, new_features


def scale_features(df: pd.DataFrame, 
                   target_col: Optional[str] = None,
                   exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Aplica StandardScaler a las características numéricas.
    
    Args:
        df: DataFrame con características
        target_col: Nombre de la columna objetivo (se excluye del escalamiento)
        exclude_cols: Lista de columnas a excluir del escalamiento
        
    Returns:
        Tupla (DataFrame escalado, objeto StandardScaler ajustado)
    """
    df_scaled = df.copy()
    
    # Columnas a excluir
    exclude = []
    if target_col:
        exclude.append(target_col)
    if exclude_cols:
        exclude.extend(exclude_cols)
    
    # Identificar columnas numéricas a escalar
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude]
    
    if not cols_to_scale:
        print("⚠️  No hay columnas numéricas para escalar")
        return df_scaled, None
    
    print(f"\n🔄 Escalando {len(cols_to_scale)} características numéricas...")
    
    scaler = StandardScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    print("✓ Escalamiento completado")
    
    return df_scaled, scaler


def feature_selection(df: pd.DataFrame,
                     target_col: str,
                     task_type: str = 'classification',
                     top_k: int = 50,
                     correlation_threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Realiza selección de características usando:
    - Correlaciones (elimina altas correlaciones)
    - Mutual Information
    - VIF (opcional, para regresión)
    
    Args:
        df: DataFrame con características
        target_col: Nombre de la columna objetivo
        task_type: 'classification' o 'regression'
        top_k: Número de características top a seleccionar
        correlation_threshold: Umbral para eliminar características altamente correlacionadas
        
    Returns:
        Tupla (DataFrame con características seleccionadas, lista de características seleccionadas)
    """
    df_selected = df.copy()
    
    # Separar características y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Solo numéricas para análisis
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    print(f"\n🔍 Realizando selección de características...")
    print(f"   Características iniciales: {X_numeric.shape[1]}")
    
    # 1. Eliminar características altamente correlacionadas
    print("\n   Paso 1: Eliminando características altamente correlacionadas...")
    corr_matrix = X_numeric.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_cols = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > correlation_threshold)
    ]
    
    if high_corr_cols:
        print(f"     Eliminando {len(high_corr_cols)} características con correlación > {correlation_threshold}")
        X_numeric = X_numeric.drop(columns=high_corr_cols)
    
    # 2. Mutual Information
    print("\n   Paso 2: Calculando Mutual Information...")
    if task_type == 'classification':
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
    
    mi_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Seleccionar top k
    top_features = mi_df.head(min(top_k, len(mi_df)))['feature'].tolist()
    
    print(f"   ✓ Top {len(top_features)} características seleccionadas por Mutual Information")
    
    # Combinar con características no numéricas (si las hay)
    non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
    selected_cols = top_features + non_numeric_cols + [target_col]
    
    df_selected = df_selected[selected_cols]
    
    print(f"   Características finales: {df_selected.shape[1] - 1} (sin contar objetivo)")
    
    return df_selected, top_features


def preprocess_tabular_data(
    input_path: Path,
    output_path: Path,
    target_col: str = 'Diagnosis',
    imputation_method: str = 'knn',
    apply_feature_selection: bool = True,
    top_k_features: int = 50,
    task_type: str = 'classification'
) -> pd.DataFrame:
    """
    Pipeline completo de preprocesamiento de datos tabulares.
    
    Args:
        input_path: Ruta al archivo CSV de entrada
        output_path: Ruta donde guardar el CSV procesado
        target_col: Nombre de la columna objetivo
        imputation_method: 'knn' o 'median'
        apply_feature_selection: Si True, aplica selección de características
        top_k_features: Número de características top a seleccionar
        task_type: 'classification' o 'regression'
        
    Returns:
        DataFrame procesado
    """
    print("=" * 60)
    print("PREPROCESAMIENTO DE DATOS TABULARES")
    print("=" * 60)
    
    # 1. Cargar datos
    df = load_clinical_data(input_path)
    
    # 2. Verificar valores faltantes
    missing = check_missing_values(df)
    
    # 3. Imputar valores faltantes
    if missing.sum() > 0:
        df = impute_missing_values(df, method=imputation_method)
    else:
        print("\n✓ No se requiere imputación")
    
    # 4. Encoding de variables categóricas
    df, encoding_info = encode_categorical_variables(df)
    
    # 5. Ingeniería de características
    df, new_features = create_feature_engineering(df)
    
    # 6. Escalamiento (excluyendo ID y objetivo)
    exclude_from_scaling = ['PatientID', target_col, 'DoctorInCharge']
    exclude_from_scaling = [col for col in exclude_from_scaling if col in df.columns]
    df, scaler = scale_features(df, target_col=target_col, exclude_cols=exclude_from_scaling)
    
    # 7. Feature selection
    if apply_feature_selection:
        df, selected_features = feature_selection(
            df, target_col, task_type=task_type, top_k=top_k_features
        )
    else:
        selected_features = None
    
    # 8. Guardar datos procesados
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Preprocesamiento completado!")
    print(f"   Datos guardados en: {output_path}")
    print(f"   Dimensiones finales: {df.shape}")
    
    # Guardar información del preprocesamiento
    info_path = output_path.parent / 'preprocessing_info.txt'
    with open(info_path, 'w') as f:
        f.write("INFORMACIÓN DE PREPROCESAMIENTO\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Método de imputación: {imputation_method}\n")
        f.write(f"Características creadas: {len(new_features)}\n")
        if new_features:
            f.write(f"  - {', '.join(new_features)}\n")
        if selected_features:
            f.write(f"\nCaracterísticas seleccionadas: {len(selected_features)}\n")
        f.write(f"\nDimensiones finales: {df.shape}\n")
    
    print(f"   Información guardada en: {info_path}")
    
    return df


if __name__ == "__main__":
    # Configuración
    BASE_DIR = Path(__file__).resolve().parents[2]
    INPUT_PATH = BASE_DIR / "data" / "raw" / "clinical" / "alzheimers_disease_data.csv"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "tabular_clean" / "clinical_data_cleaned.csv"
    
    # Ejecutar preprocesamiento
    df_processed = preprocess_tabular_data(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        target_col='Diagnosis',
        imputation_method='knn',
        apply_feature_selection=True,
        top_k_features=50,
        task_type='classification'
    )
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Filas: {df_processed.shape[0]}")
    print(f"Columnas: {df_processed.shape[1]}")
    print(f"\nPrimeras columnas: {list(df_processed.columns[:10])}")

