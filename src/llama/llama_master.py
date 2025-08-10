import subprocess
import time
import pandas as pd
import os
import itertools
import socket
import re

def encontrar_puerto_libre():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        puerto = s.getsockname()[1]
    return puerto

def iniciar_logging_gpu(archivo_log="gpu_stats.log"):
    comando = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv",
        "--loop=1",
        f"--filename={archivo_log}"
    ]
    try:
        proceso = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proceso
    except Exception as e:
        return None

def detener_logging_gpu(proceso):
    if proceso is None:
        return
    try:
        proceso.terminate()
        proceso.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proceso.kill()
    except Exception as e:
        pass

def parsear_log_gpu(archivo_log="gpu_stats.log", num_gpus=1):
    if not os.path.exists(archivo_log):
        return {}
    
    try:
        df = pd.read_csv(archivo_log)
        df.columns = [c.strip() for c in df.columns]
        
        columna_util = None
        columna_mem = None
        for col in df.columns:
            if 'utilization' in col.lower():
                columna_util = col
            elif 'memory.used' in col.lower():
                columna_mem = col
        
        if columna_util is None or columna_mem is None:
            return {}
            
        df[columna_util] = df[columna_util].astype(str).str.replace(' %', '').str.replace('%', '')
        df[columna_mem] = df[columna_mem].astype(str).str.replace(' MiB', '').str.replace('MiB', '')
        
        df[columna_util] = pd.to_numeric(df[columna_util], errors='coerce').fillna(0)
        df[columna_mem] = pd.to_numeric(df[columna_mem], errors='coerce').fillna(0)

        resultados = {}
        for i in range(num_gpus):
            gpu_df = df[df['index'] == i]
            if not gpu_df.empty:
                resultados[f'gpu_{i}_max_util_%'] = gpu_df[columna_util].max()
                resultados[f'gpu_{i}_max_mem_mib'] = gpu_df[columna_mem].max()
            else:
                resultados[f'gpu_{i}_max_util_%'] = 0
                resultados[f'gpu_{i}_max_mem_mib'] = 0
                
    except Exception as e:
        return {}
    
    try:
        if os.path.exists(archivo_log):
            os.remove(archivo_log)
    except:
        pass
        
    return resultados

def extraer_tiempo_entrenamiento_de_salida(texto_stdout):
    "Extrae el tiempo de entrenamiento a partir del log de cada script de experimento"
    lineas = texto_stdout.split('\n')
    
    for linea in lineas:    
        coincidencia_tiempo = re.search(r'Tiempo.*?(\d+\.?\d*)\s*segundos?', linea.lower())
        if coincidencia_tiempo:
            try:
                return float(coincidencia_tiempo.group(1))
            except ValueError:
                continue
    
    return None

def ejecutar_experimento_unico(info_dataset, num_gpus, estrategia_paralela, parametros_entrenamiento, configuracion_modelo):
    argumentos_comando = [
        '--dataset_name', info_dataset['name'],
        '--dataset_config', info_dataset.get('config', 'default'),
        '--dataset_fraction', str(parametros_entrenamiento['dataset_fraction']),
        '--batch_size', str(parametros_entrenamiento['batch_size']),
        '--num_iterations', str(parametros_entrenamiento['num_iterations']),
        '--dim', str(configuracion_modelo['dim']),
        '--n_layers', str(configuracion_modelo['n_layers']),
        '--n_heads', str(configuracion_modelo['n_heads']),
    ]
    
    entorno = os.environ.copy()
    
    if estrategia_paralela == 'SingleGPU':
        script_a_ejecutar = 'llama_1gpu.py'
        comando = ['python', script_a_ejecutar] + argumentos_comando
        
    elif estrategia_paralela in ['DP', 'TP']:
        script_a_ejecutar = 'llama_dp_2.py' if estrategia_paralela == 'DP' else 'llama_tp_2.py'
        puerto_libre = encontrar_puerto_libre()
        
        comando = [
            'torchrun',
            '--standalone',
            '--nproc_per_node', str(num_gpus),
            '--master_addr', 'localhost',
            '--master_port', str(puerto_libre),
            script_a_ejecutar
        ] + argumentos_comando
        
        for clave in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
            entorno.pop(clave, None)
            
    else:
        raise ValueError(f"Estrategia paralela inválida: {estrategia_paralela}")

    tiempo_inicio_total = time.time()
    try:
        resultado = subprocess.run(
            comando, 
            capture_output=True, 
            text=True, 
            env=entorno,
            timeout=3600
        )
        tiempo_fin_total = time.time()
        
        if resultado.returncode != 0:
            return {
                'total_time_s': -1, 
                'training_time_s': -1, 
                'success': False, 
                'error_msg': resultado.stderr[:500]
            }

        tiempo_total = tiempo_fin_total - tiempo_inicio_total
        
        tiempo_entrenamiento = extraer_tiempo_entrenamiento_de_salida(resultado.stdout)
        
        if tiempo_entrenamiento is None:
            tiempo_entrenamiento = tiempo_total
        
        return {
            'total_time_s': tiempo_total,
            'training_time_s': tiempo_entrenamiento, 
            'success': True, 
            'error_msg': ''
        }
        
    except subprocess.TimeoutExpired:
        return {
            'total_time_s': -1, 
            'training_time_s': -1, 
            'success': False, 
            'error_msg': 'Timeout después de 1 hora'
        }
    except Exception as e:
        return {
            'total_time_s': -1, 
            'training_time_s': -1, 
            'success': False, 
            'error_msg': str(e)
        }

if __name__ == "__main__":
    
    DATASETS = {
        'bookcorpus': {'name': 'bookcorpus', 'config': 'plain_text'},
        'pg19': {'name': 'pg19', 'config': 'default'},
        'wiki40b': {'name': 'wiki40b', 'config': 'en'},
        'tinystories': {'name': 'roneneldan/TinyStories', 'config': 'default'},
        'multi_news': {'name': 'multi_news', 'config': 'default'}
    }
    
    CONFIGS_GPU = [1, 2]
    ESTRATEGIAS = ['DP', 'TP']
    
    DATASET_FRACTIONS = {
        'bookcorpus': [0.05],
        'pg19': [0.02],
        'wiki40b': [0.02],
        'tinystories': [0.3],
        'multi_news': [0.7]
    }

    BATCH = [16, 32, 64]
    
    ITERACIONES = [128, 512, 1024]
    
    CONFIGURACIONES_MODELO = {
        'small': {'dim': 256, 'n_layers': 2, 'n_heads': 16},
        'base': {'dim': 512, 'n_layers': 4, 'n_heads': 32},
        'large': {'dim': 1024, 'n_layers': 8, 'n_heads': 64}
    }
    
    todos_resultados = []
    archivo_resultados = "archivo.csv"
    
    for dataset_name, dataset_info in DATASETS.items():
        fractions = DATASET_FRACTIONS.get(dataset_name, [1.0])
        combinaciones_parametros_entrenamiento = [
            {'fraccion datos': frac, 'batch': bs, 'iteraciones': ni}
            for frac, bs, ni in itertools.product(fractions, BATCH, ITERACIONES)
        ]

        contador_experimento = 0

        for nombre_dataset, info_dataset in DATASETS.items():
            for num_gpus in CONFIGS_GPU:
                for nombre_modelo, configuracion_modelo in CONFIGURACIONES_MODELO.items():
                    for parametros_entrenamiento in combinaciones_parametros_entrenamiento:
                        
                        estrategias_a_ejecutar = ['SingleGPU'] if num_gpus == 1 else ESTRATEGIAS

                        for estrategia in estrategias_a_ejecutar:
                            contador_experimento += 1
                            
                            archivo_log = f"gpu_stats_{contador_experimento}.log"
                            proceso_log = iniciar_logging_gpu(archivo_log)
                            
                            try:
                                resultados_exp = ejecutar_experimento_unico(
                                    info_dataset, num_gpus, estrategia, parametros_entrenamiento, configuracion_modelo
                                )
                                
                                info_ejecucion = {
                                    'experiment_id': contador_experimento,
                                    'dataset': nombre_dataset,
                                    'num_gpus': num_gpus,
                                    'strategy': estrategia,
                                    'model_name': nombre_modelo,
                                    **parametros_entrenamiento,
                                    **configuracion_modelo
                                }
                                
                                detener_logging_gpu(proceso_log)
                                metricas_gpu = parsear_log_gpu(archivo_log, num_gpus)
                                
                                resultado_final = {**info_ejecucion, **resultados_exp, **metricas_gpu}
                                todos_resultados.append(resultado_final)
                                        
                            except Exception as e:
                                detener_logging_gpu(proceso_log)
                                
                                resultado_error = {
                                    'experiment_id': contador_experimento,
                                    'dataset': nombre_dataset,
                                    'num_gpus': num_gpus,
                                    'strategy': estrategia,
                                    'model_name': nombre_modelo,
                                    **parametros_entrenamiento,
                                    **configuracion_modelo,
                                    'total_time_s': -1,
                                    'training_time_s': -1,
                                    'success': False,
                                    'error_msg': f'Error: {str(e)}'
                                }
                                todos_resultados.append(resultado_error)
                            
                            pd.DataFrame(todos_resultados).to_csv(archivo_resultados, index=False)