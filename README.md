# ML Model Parallelization

José María Manzano Ortega

## Resumen

Este Trabajo de Fin de Máster investiga las técnicas de paralelización aplicadas a modelos de Machine Learning para optimizar su rendimiento y escalabilidad. El estudio llevado a cabo se centra en dos arquitecturas fundamentales: XGBoost y modelos LLaMA-2 basados en la arquitectura Transformer.

### Objetivos Principales
- Analizar las estrategias de paralelización: paralelismo de datos y tensor
- Evaluar el rendimiento en diferentes arquitecturas de hardware (CPU y GPU)
- Comparar métricas de speed-up, size-up y scale-up
- Investigar las limitaciones de comunicación en entrenamientos distribuidos

### Metodología
El trabajo implementa y evalúa dos paradigmas de paralelización:
1. **Paralelismo de Datos**: Distribución de lotes de datos entre múltiples dispositivos
2. **Paralelismo Tensor**: Partición de operaciones matriciales entre dispositivos

### Resultados Clave
- **XGBoost**: Muestra mejor rendimiento en GPU comparado con CPU, pero limitaciones significativas en escalabilidad multi-GPU debido al overhead de comunicación
- **Modelos LLaMA-2**: El paralelismo de tensor supera al paralelismo de datos en arquitecturas Transformer en cuanto a escalabilidad del tiempo de entrenamiento
- **FSDP (Fully Sharded Data Parallel)**: Demuestra eficiencia en la gestión de memoria para modelos grandes

### Contenido

**src/llama/** - Scripts de experimentos con modelos LLaMA-2
- `llama_master.py` - Script maestro para coordinar experimentos LLaMA
- `llama_dp.py` - Implementación de paralelismo de datos para LLaMA
- `llama_tp.py` - Implementación de paralelismo tensor para LLaMA
- `llama_1gpu.py` - Implementación de referencia en una sola GPU

**src/xgboost/** - Scripts de experimentos con XGBoost
- `cpu_xgboost.py` - Implementación de XGBoost en CPU
- `gpu_xgboost.py` - Implementación de XGBoost en GPU

**data/llama/** - Resultados experimentales de modelos LLaMA-2
- `pytorch.csv`, `pytorch2.csv`, `pytorch3.csv` - Métricas de rendimiento LLaMA

**data/xgboost/** - Resultados experimentales de XGBoost
- `cpu/` - Resultados de experimentos en CPU
- `gpu/` - Resultados de experimentos en GPU

**requirements_pytorch.txt** - Dependencias para experimentos con LLaMA-2

**requirements_xgboost.txt** - Dependencias para experimentos con XGBoost

**Material_Adicional.pdf** - Gráficas de las métricas por experimento

**TFM_Manzano_Ortega_José María.pdf** - Documento completo del Trabajo de Fin de Máster

---

## Summary

This Master's Thesis investigates parallelization techniques applied to Machine Learning models to optimize their performance and scalability. The study focuses on two fundamental architectures: XGBoost (Gradient Boosting Trees) and LLaMA-2 models based on the Transformer architecture.

### Main Objectives
- Analyze parallelization strategies: data and tensor parallelism
- Evaluate performance across different hardware architectures (CPU and GPU)
- Compare speed-up, size-up, and scale-up metrics
- Investigate communication limitations in distributed training

### Methodology
The work implements and evaluates two parallelization paradigms:
1. **Data Parallelism**: Distribution of data batches across multiple devices
2. **Tensor Parallelism**: Partitioning of matrix operations across devices

### Key Results
- **XGBoost**: Shows better performance on GPU compared to CPU, but significant limitations in multi-GPU scalability due to communication overhead
- **LLaMA-2 Models**: Tensor parallelism outperforms data parallelism for Transformer architectures in terms of training speed
- **FSDP (Fully Sharded Data Parallel)**: Demonstrates efficiency in memory management for large models

### Content

**src/llama/** - LLaMA-2 model experiment scripts
- `llama_master.py` - Master script to coordinate LLaMA experiments
- `llama_dp.py` - Data parallelism implementation for LLaMA
- `llama_tp.py` - Tensor parallelism implementation for LLaMA
- `llama_1gpu.py` - Single-GPU reference implementation

**src/xgboost/** - XGBoost experiment scripts
- `cpu_xgboost.py` - XGBoost CPU implementation
- `gpu_xgboost.py` - XGBoost GPU implementation

**data/llama/** - LLaMA-2 model experimental results
- `pytorch.csv`, `pytorch2.csv`, `pytorch3.csv` - LLaMA performance metrics

**data/xgboost/** - XGBoost experimental results
- `cpu/` - CPU experiment results
- `gpu/` - GPU experiment results

**requirements_pytorch.txt** - Dependencies for LLaMA-2 experiments 

**requirements_xgboost.txt** - Dependencies for XGBoost experiments

**Material_Adicional.pdf** - Experimental metrics charts

**TFM_Manzano_Ortega_José María.pdf** - Complete Master's Thesis document