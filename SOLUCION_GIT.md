# Solución para el Problema de Archivos Grandes en Git

## Problema
El directorio `env/` con archivos grandes (torch_cpu.dll de 250MB) está en el historial de Git, causando que el push falle.

## Soluciones

### Opción 1: Usar BFG Repo-Cleaner (RECOMENDADO)

1. **Descargar BFG Repo-Cleaner:**
   - Ve a: https://rtyley.github.io/bfg-repo-cleaner/
   - Descarga el JAR file

2. **Ejecutar BFG:**
   ```bash
   # Crear un clon limpio
   git clone --mirror https://github.com/mmerazItam/proyectoMineria.git proyectoMineria-clean.git
   
   # Ejecutar BFG para eliminar env/
   java -jar bfg.jar --delete-folders env proyectoMineria-clean.git
   
   # Limpiar y hacer push
   cd proyectoMineria-clean.git
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push
   ```

### Opción 2: Crear Nuevo Repositorio Limpio (MÁS SIMPLE)

Si no necesitas el historial completo, puedes crear un nuevo repositorio:

1. **Hacer backup del estado actual:**
   ```bash
   # Copiar todos los archivos (excepto .git) a un directorio temporal
   ```

2. **Eliminar .git y crear nuevo repositorio:**
   ```bash
   Remove-Item -Recurse -Force .git
   git init
   git add .
   git commit -m "Initial commit - proyecto limpio"
   git remote add origin https://github.com/mmerazItam/proyectoMineria.git
   git push -u origin main --force
   ```

### Opción 3: Usar git filter-repo (MODERNO)

1. **Instalar git-filter-repo:**
   ```bash
   pip install git-filter-repo
   ```

2. **Ejecutar:**
   ```bash
   git filter-repo --path env --invert-paths
   git push origin --force --all
   ```

## Nota Importante

⚠️ **ADVERTENCIA**: Las opciones 2 y 3 requieren `--force` push, lo que sobrescribirá el historial remoto. 
Asegúrate de que nadie más esté trabajando en el repositorio o coordina con tu equipo.

## Prevención Futura

El archivo `.gitignore` ya está configurado para evitar que `env/` se agregue de nuevo.
Siempre verifica con `git status` antes de hacer commit.

