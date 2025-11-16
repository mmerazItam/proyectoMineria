# Descripción de Variables - Datos Raw

Este documento describe todas las variables disponibles en los datos raw del proyecto de minería de datos sobre la enfermedad de Alzheimer.

## Estructura General de los Datos

El dataset raw está organizado en dos tipos principales de datos:
1. **Datos Clínicos**: Archivo CSV con información demográfica, clínica y de diagnóstico
2. **Datos de Imágenes**: Conjunto de imágenes de resonancia magnética (MRI) organizadas por nivel de demencia

---

## Datos Clínicos (`alzheimers_disease_data.csv`)

**Total de registros**: 2,149 pacientes  
**Total de variables**: 35  
**Valores faltantes**: Ninguno (dataset completo)

### Variables de Identificación

#### `PatientID`
- **Tipo**: Entero (int64)
- **Descripción**: Identificador único del paciente
- **Rango**: 4,751 - 6,899
- **Valores únicos**: 2,149

#### `DoctorInCharge`
- **Tipo**: Texto (object)
- **Descripción**: Identificador del médico responsable del caso
- **Valores**: Códigos de identificación (ej: "XXXConfid")

---

### Variables Demográficas

#### `Age`
- **Tipo**: Entero (int64)
- **Descripción**: Edad del paciente en años
- **Rango**: 60 - 90 años
- **Media**: 74.91 años
- **Mediana**: 75 años
- **Desviación estándar**: 8.99 años

#### `Gender`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Género del paciente
- **Valores**:
  - `0`: Femenino
  - `1`: Masculino

#### `Ethnicity`
- **Tipo**: Entero (int64) - Categórico
- **Descripción**: Grupo étnico del paciente
- **Valores**:
  - `0`: Grupo étnico 1
  - `1`: Grupo étnico 2
  - `2`: Grupo étnico 3
  - `3`: Grupo étnico 4

#### `EducationLevel`
- **Tipo**: Entero (int64) - Categórico
- **Descripción**: Nivel educativo del paciente
- **Valores**:
  - `0`: Sin educación formal / Primaria incompleta
  - `1`: Primaria completa / Secundaria incompleta
  - `2`: Secundaria completa / Universitaria incompleta
  - `3`: Universitaria completa / Postgrado

---

### Variables de Salud Física

#### `BMI`
- **Tipo**: Flotante (float64)
- **Descripción**: Índice de Masa Corporal (Body Mass Index)
- **Rango**: ~15.46 - ~39.46 kg/m²
- **Unidad**: kg/m²

#### `Smoking`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Historial de tabaquismo
- **Valores**:
  - `0`: No fumador
  - `1`: Fumador / Ex-fumador

#### `AlcoholConsumption`
- **Tipo**: Flotante (float64)
- **Descripción**: Consumo de alcohol (probablemente en unidades por semana o similar)
- **Rango**: ~0.65 - ~19.97
- **Unidad**: Unidades de alcohol (a verificar)

#### `PhysicalActivity`
- **Tipo**: Flotante (float64)
- **Descripción**: Nivel de actividad física
- **Rango**: ~0.21 - ~9.99
- **Unidad**: Escala de actividad física (a verificar)

#### `DietQuality`
- **Tipo**: Flotante (float64)
- **Descripción**: Calidad de la dieta
- **Rango**: ~0.04 - ~9.77
- **Unidad**: Escala de calidad dietética (a verificar)

#### `SleepQuality`
- **Tipo**: Flotante (float64)
- **Descripción**: Calidad del sueño
- **Rango**: ~4.21 - ~9.99
- **Unidad**: Escala de calidad del sueño (a verificar)

---

### Variables de Historial Médico y Comorbilidades

#### `FamilyHistoryAlzheimers`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Historial familiar de enfermedad de Alzheimer
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `CardiovascularDisease`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de enfermedad cardiovascular
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Diabetes`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de diabetes
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Depression`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de depresión
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `HeadInjury`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Historial de lesiones en la cabeza
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Hypertension`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de hipertensión
- **Valores**:
  - `0`: No
  - `1`: Sí

---

### Variables Cardiovasculares

#### `SystolicBP`
- **Tipo**: Entero (int64)
- **Descripción**: Presión arterial sistólica
- **Rango**: 90 - 178 mmHg
- **Unidad**: mmHg (milímetros de mercurio)

#### `DiastolicBP`
- **Tipo**: Entero (int64)
- **Descripción**: Presión arterial diastólica
- **Rango**: 60 - 119 mmHg
- **Unidad**: mmHg (milímetros de mercurio)

#### `CholesterolTotal`
- **Tipo**: Flotante (float64)
- **Descripción**: Colesterol total en sangre
- **Rango**: ~151.38 - ~299.87 mg/dL
- **Unidad**: mg/dL (miligramos por decilitro)

#### `CholesterolLDL`
- **Tipo**: Flotante (float64)
- **Descripción**: Colesterol LDL (lipoproteína de baja densidad) - "colesterol malo"
- **Rango**: ~52.47 - ~198.45 mg/dL
- **Unidad**: mg/dL

#### `CholesterolHDL`
- **Tipo**: Flotante (float64)
- **Descripción**: Colesterol HDL (lipoproteína de alta densidad) - "colesterol bueno"
- **Rango**: ~20.89 - ~99.29 mg/dL
- **Unidad**: mg/dL

#### `CholesterolTriglycerides`
- **Tipo**: Flotante (float64)
- **Descripción**: Triglicéridos en sangre
- **Rango**: ~52.79 - ~379.48 mg/dL
- **Unidad**: mg/dL

---

### Variables Cognitivas y Funcionales

#### `MMSE`
- **Tipo**: Flotante (float64)
- **Descripción**: Mini-Mental State Examination - Escala de evaluación cognitiva
- **Rango**: ~0.41 - ~28.72
- **Escala**: 0-30 (donde 30 es normal y valores más bajos indican deterioro cognitivo)
- **Nota**: Valores por debajo de 24 sugieren deterioro cognitivo

#### `FunctionalAssessment`
- **Tipo**: Flotante (float64)
- **Descripción**: Evaluación funcional de las actividades de la vida diaria
- **Rango**: ~0.00 - ~9.99
- **Unidad**: Escala de evaluación funcional (a verificar)

#### `ADL`
- **Tipo**: Flotante (float64)
- **Descripción**: Activities of Daily Living - Actividades de la vida diaria
- **Rango**: ~0.01 - ~9.93
- **Unidad**: Escala de independencia funcional (valores más altos pueden indicar mayor dependencia)

---

### Variables de Síntomas y Comportamiento

#### `MemoryComplaints`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Quejas de memoria reportadas
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `BehavioralProblems`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de problemas de comportamiento
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Confusion`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de confusión
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Disorientation`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de desorientación
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `PersonalityChanges`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de cambios de personalidad
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `DifficultyCompletingTasks`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Dificultad para completar tareas
- **Valores**:
  - `0`: No
  - `1`: Sí

#### `Forgetfulness`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Presencia de olvidos
- **Valores**:
  - `0`: No
  - `1`: Sí
- **Prevalencia**: ~30.15% de los casos presentan este síntoma

---

### Variable de Diagnóstico (Variable Objetivo)

#### `Diagnosis`
- **Tipo**: Entero (int64) - Binario
- **Descripción**: Diagnóstico de enfermedad de Alzheimer
- **Valores**:
  - `0`: Sin diagnóstico de Alzheimer / Control
  - `1`: Con diagnóstico de Alzheimer
- **Prevalencia**: ~35.37% de los casos tienen diagnóstico positivo
- **Nota**: Esta es la variable objetivo principal para modelos de clasificación

---

## Datos de Imágenes

### Estructura de Carpetas

Los datos de imágenes están organizados en dos conjuntos principales:

#### 1. Dataset Original (`OriginalDataset/`)
- **Formato**: Imágenes JPG
- **Estructura**: Organizadas por nivel de demencia
- **Clases**:
  - `NonDemented/`: 3,200 imágenes
  - `VeryMildDemented/`: 2,240 imágenes
  - `MildDemented/`: 896 imágenes
  - `ModerateDemented/`: 64 imágenes
- **Total**: 6,400 imágenes

#### 2. Dataset Aumentado (`AugmentedAlzheimerDataset/`)
- **Formato**: Imágenes JPG
- **Descripción**: Dataset con técnicas de data augmentation aplicadas
- **Estructura**: Organizadas por nivel de demencia
- **Clases**:
  - `NonDemented/`: 9,600 imágenes
  - `VeryMildDemented/`: 8,960 imágenes
  - `MildDemented/`: 8,960 imágenes
  - `ModerateDemented/`: 6,464 imágenes
- **Total**: 33,984 imágenes

### Clasificación de Niveles de Demencia

Las imágenes están clasificadas en 4 niveles de severidad:

1. **NonDemented** (Sin demencia)
   - Pacientes sin signos de demencia
   - Clase de control

2. **VeryMildDemented** (Demencia muy leve)
   - Primeros signos de deterioro cognitivo
   - Cambios mínimos detectables

3. **MildDemented** (Demencia leve)
   - Deterioro cognitivo moderado
   - Síntomas más evidentes

4. **ModerateDemented** (Demencia moderada)
   - Deterioro cognitivo significativo
   - Síntomas claramente visibles en las imágenes

### Características de las Imágenes

- **Tipo**: Imágenes de resonancia magnética (MRI) del cerebro
- **Formato**: JPG
- **Uso**: Clasificación de imágenes para diagnóstico asistido por computadora
- **Aplicación**: Modelos de deep learning (CNN) para clasificación multiclase

---

## Notas Importantes

1. **Codificación de Variables Binarias**: Todas las variables binarias usan 0/1, donde típicamente 0 = No/Ausente y 1 = Sí/Presente.

2. **Variables Categóricas**: Las variables `Ethnicity` y `EducationLevel` son categóricas ordinales, pero los valores específicos de cada categoría deben verificarse con la documentación original del dataset.

3. **Escalas de Variables Continuas**: Algunas variables como `PhysicalActivity`, `DietQuality`, y `SleepQuality` usan escalas numéricas que requieren verificación de la documentación original para interpretación precisa.

4. **Relación entre Datos**: Los datos clínicos y las imágenes pueden estar relacionados a través del `PatientID`, aunque esto debe verificarse en el proceso de integración multimodal.

5. **Balance de Clases**: 
   - En datos clínicos: ~65% controles vs ~35% casos de Alzheimer
   - En imágenes: Desbalance significativo entre clases (especialmente en el dataset original)

6. **Uso en Modelos**:
   - **Clasificación**: `Diagnosis` como variable objetivo
   - **Regresión**: Variables como `MMSE`, `ADL`, `FunctionalAssessment` pueden ser objetivos de regresión
   - **Multimodal**: Combinación de datos clínicos y de imágenes para modelos híbridos

---

## Referencias y Fuentes

- Dataset clínico: `data/raw/clinical/alzheimers_disease_data.csv`
- Dataset de imágenes original: `data/raw/images/OriginalDataset/`
- Dataset de imágenes aumentado: `data/raw/images/AugmentedAlzheimerDataset/`

---

*Última actualización: Generado automáticamente a partir del análisis del dataset raw*

