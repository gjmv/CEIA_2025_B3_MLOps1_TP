# CEIA_2025_B3_MLOps1_TP

Curso de Especialización en Inteligencia Artificial  
Año 2025  
Bimestre 3  

## Materia: Operaciones de Aprendizaje Automático 1  

## Docente:
* Facundo Lucianna

## Integrantes:
* Mealla Pablo
* Mendoza Dante
* Viñas Gustavo


## Iniciar servicios

```bash
docker compose --profile all up
```

   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

## Apagar los servicios

Detener los servicios:

```bash
docker compose --profile all down
```

Detener los servicios y eliminar toda la infraestructura (liberando espacio en disco):

```bash
docker compose down --rmi all --volumes
```
Nota: Si haces esto, perderás todo en los buckets y bases de datos.
