import dask
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

from xgboost import dask as dxgb
from xgboost.dask import DaskDMatrix
import subprocess
from sklearn.datasets import load_svmlight_file
import time
import pandas as pd
import os
import itertools
import math
import numpy as np
import socket
import psutil

def obtener_mejor_interfaz_red():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_local = s.getsockname()[0]
        s.close()
        
        for interfaz, direcciones in psutil.net_if_addrs().items():
            for direccion in direcciones:
                if direccion.address == ip_local and direccion.family == socket.AF_INET:
                    return interfaz
    except:
        pass
    
    interfaces = psutil.net_if_addrs()
    prioridades = ['enp', 'eth', 'ib', 'em']
    
    for prioridad in prioridades:
        for interfaz in interfaces:
            if interfaz.startswith(prioridad) and interfaz != 'lo':
                estadisticas = psutil.net_if_stats().get(interfaz)
                if estadisticas and estadisticas.isup:
                    return interfaz
    
    for interfaz in interfaces:
        if interfaz not in ['lo', 'docker0'] and not interfaz.startswith('veth'):
            estadisticas = psutil.net_if_stats().get(interfaz)
            if estadisticas and estadisticas.isup:
                return interfaz
    
    return None

def iniciar_logging_gpu(archivo_log="gpu_stats.log"):
    comando = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv",
        "--loop=1",
        f"--filename={archivo_log}"
    ]
    proceso = subprocess.Popen(comando)
    return proceso

def detener_logging_gpu(proceso):
    proceso.terminate()
    try:
        proceso.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proceso.kill()

def parsear_log_gpu(archivo_log="gpu_stats.log", num_gpus=1):
    if not os.path.exists(archivo_log):
        return {}
    
    try:
        df = pd.read_csv(archivo_log)
        
        df.columns = [c.strip() for c in df.columns]
        df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace(' %', '').astype(int)
        df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace(' MiB', '').astype(int)

        resultados = {}
        for i in range(num_gpus):
            gpu_df = df[df['index'] == i]
            if not gpu_df.empty:
                resultados[f'gpu_{i}_max_util_%'] = gpu_df['utilization.gpu [%]'].max()
                resultados[f'gpu_{i}_max_mem_mib'] = gpu_df['memory.used [MiB]'].max()
            else:
                resultados[f'gpu_{i}_max_util_%'] = 0
                resultados[f'gpu_{i}_max_mem_mib'] = 0
    except Exception as e:
        return {}
            
    if os.path.exists(archivo_log):
        os.remove(archivo_log)
    return resultados

def preprocesar_datos_taxis(ddf):
    ddf['tpep_pickup_datetime'] = dd.to_datetime(ddf['tpep_pickup_datetime'])
    ddf['tpep_dropoff_datetime'] = dd.to_datetime(ddf['tpep_dropoff_datetime'])
    
    ddf['pickup_hour'] = ddf['tpep_pickup_datetime'].dt.hour
    ddf['pickup_dayofweek'] = ddf['tpep_pickup_datetime'].dt.dayofweek
    
    ddf['trip_duration'] = (ddf['tpep_dropoff_datetime'] - ddf['tpep_pickup_datetime']).dt.total_seconds()

    ddf['store_and_fwd_flag'] = (ddf['store_and_fwd_flag'] == 'Y').astype(int)
    
    ddf = ddf.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    return ddf

def muestrear_dataframe_dask(ddf, fraccion, estado_aleatorio=42):
    return ddf.sample(frac=fraccion, random_state=estado_aleatorio)

def muestrear_array_dask(arr, fraccion, estado_aleatorio=42):
    filas_totales = arr.shape[0]
    filas_muestra = int(filas_totales * fraccion)
    
    return arr[:filas_muestra]

def cargar_y_preparar_datos(cliente, info_dataset, num_particiones, fraccion_datos=1.0):
    ruta_datos = info_dataset['path']
    formato_datos = info_dataset['format']
    
    if formato_datos == 'csv':
        opcion_header = 0 if info_dataset.get('has_header') else None
        separador = info_dataset.get('separator', ',')
        ddf = dd.read_csv(ruta_datos, header=opcion_header, sep=separador, blocksize="128MB", assume_missing=True)

        if fraccion_datos < 1.0:
            ddf = muestrear_dataframe_dask(ddf, fraccion_datos)

        if info_dataset.get('needs_preprocessing'):
            ddf = preprocesar_datos_taxis(ddf)
        
        columna_etiqueta = info_dataset.get('label_column')
        if not columna_etiqueta:
            indice_columna_etiqueta = info_dataset.get('label_column_index', -1)
            if isinstance(ddf.columns[indice_columna_etiqueta], int):
                 columna_etiqueta = indice_columna_etiqueta
            else:
                 columna_etiqueta = ddf.columns[indice_columna_etiqueta]

        y_df = ddf[columna_etiqueta]
        X_df = ddf.drop(columns=[columna_etiqueta])

        X_df = X_df.select_dtypes(include=np.number)
        
        X_df = X_df.repartition(npartitions=num_particiones)
        y_df = y_df.repartition(npartitions=num_particiones)

    elif formato_datos == 'svmlight':
        X_sparse, y_numpy = load_svmlight_file(ruta_datos, n_features=info_dataset.get('n_features'))
        
        X_da = da.from_array(X_sparse, chunks='auto')
        y_da = da.from_array(y_numpy, chunks='auto')
        
        if fraccion_datos < 1.0:
            X_da = muestrear_array_dask(X_da, fraccion_datos)
            y_da = muestrear_array_dask(y_da, fraccion_datos)
        
        tamaño_chunk_fila = math.ceil(X_da.shape[0] / num_particiones)
        X_df = X_da.rechunk(chunks={0: tamaño_chunk_fila})
        y_df = y_da.rechunk(chunks=X_df.chunksize[0])

    if isinstance(X_df, dd.DataFrame):
        X = X_df.to_dask_array(lengths=True)
        y = y_df.to_dask_array(lengths=True)
    else:
        X, y = X_df, y_df

    if info_dataset.get('remap_labels', False):
        y = y - 1
        
    X, y = cliente.persist([X, y])
    wait([X, y])
    
    ubicaciones_x = cliente.who_has(X)
    ubicaciones_y = cliente.who_has(y)
    
    trabajadores_con_x = set()
    trabajadores_con_y = set()
    
    for trabajadores_particion in ubicaciones_x.values():
        trabajadores_con_x.update(trabajadores_particion)
    for trabajadores_particion in ubicaciones_y.values():
        trabajadores_con_y.update(trabajadores_particion)
    
    try:
        forma_x = X.shape
        forma_y = y.shape
        
        if forma_x[0] != forma_y[0]:
            raise ValueError(f"Mismatch en número de muestras: X tiene {forma_x[0]}, Y tiene {forma_y[0]}")
            
    except Exception as e:
        raise
    
    return X, y

def ejecutar_experimento_unico(cliente, info_dataset, num_gpus, parametros_xgb, fraccion_datos=1.0):
    if num_gpus == 1:
        num_particiones = 8
    else:
        num_particiones = 8
    
    X, y = cargar_y_preparar_datos(cliente, info_dataset, num_particiones, fraccion_datos)
    
    dtrain = DaskDMatrix(cliente, X, y)
    
    for direccion_trabajador, info_trabajador in cliente.scheduler_info()['workers'].items():
        pass

    parametros_iterativos = parametros_xgb.copy()
    num_rondas = parametros_iterativos.pop('num_boost_round')

    parametros = {
        "tree_method": "hist",
        "device": "cuda",
        **parametros_iterativos
    }
    
    if info_dataset['task_type'] == 'classification':
        if info_dataset['num_class'] == 2:
            parametros['objective'] = 'binary:logistic'
            parametros['eval_metric'] = 'auc'
        else:
            parametros['objective'] = 'multi:softprob'
            parametros['eval_metric'] = 'mlogloss'
            parametros['num_class'] = info_dataset['num_class']
    else:
        parametros['objective'] = 'reg:squarederror'
        parametros['eval_metric'] = 'rmse'

    tiempo_inicio = time.time()
    dxgb.train(cliente, parametros, dtrain, num_boost_round=num_rondas)
    tiempo_fin = time.time()
    
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio
    
    return {
        'training_time_s': tiempo_entrenamiento,
        'data_fraction': fraccion_datos,
        'actual_samples': X.shape[0]
    }

if __name__ == "__main__":
    DIRECTORIO_DATOS = "/una/ruta/cualquiera/"
    
    DATASETS = {
        'covtype': {'path': os.path.join(DIRECTORIO_DATOS, 'covtype.csv'), 'format': 'csv', 'task_type': 'classification', 'num_class': 7, 'remap_labels': True},
        'microsoft': {'path': os.path.join(DIRECTORIO_DATOS, 'Microsoft'), 'format': 'svmlight', 'task_type': 'classification', 'num_class': 5, 'n_features': 136},
        'susy': {'path': os.path.join(DIRECTORIO_DATOS, 'SUSY.csv'), 'format': 'csv', 'task_type': 'classification', 'num_class': 2, 'label_column_index': 0},
        'taxis_usa': {'path': os.path.join(DIRECTORIO_DATOS, 'taxis_usa.csv'), 'format': 'csv', 'task_type': 'regression', 'has_header': True, 'label_column': 'total_amount', 'needs_preprocessing': True},
        'yearprediction': {'path': os.path.join(DIRECTORIO_DATOS, 'YearPredictionMSD.csv'), 'format': 'csv', 'task_type': 'regression', 'label_column_index': 0},
        'epsilon_t': {'path': os.path.join(DIRECTORIO_DATOS, 'epsilon_normalized_t'), 'format': 'svmlight', 'task_type': 'classification', 'num_class': 2, 'n_features': 2000, 'remap_labels': True}
    }
    
    CONFIGS_GPU = [1, 2]
    FRACCIONES_DATOS = [0.25, 0.5, 1.0]
    
    PARAMETROS = {
        'num_boost_round': [500, 1000],
        'max_depth': [6, 10]
    }
    
    LIMITE_MEMORIA_TRABAJADOR = '8GiB'
    
    todos_resultados = []
    archivo_resultados = "archivo.csv"
    
    combinaciones_parametros = [dict(zip(PARAMETROS.keys(), v)) for v in itertools.product(*PARAMETROS.values())]

    interfaz_red = obtener_mejor_interfaz_red()

    for nombre_dataset, info_dataset in DATASETS.items():
        for fraccion_datos in FRACCIONES_DATOS:
            for num_gpus in CONFIGS_GPU:
                for parametros in combinaciones_parametros:
                    
                    archivo_log = f"gpu_stats_{nombre_dataset}_frac{fraccion_datos}_{num_gpus}gpus_{parametros['num_boost_round']}r_{parametros['max_depth']}d.log"
                    
                    dispositivos_gpu = ','.join(map(str, range(num_gpus)))
                    
                    proceso_log = iniciar_logging_gpu(archivo_log)
                    try:
                        kwargs_cluster = {
                            'CUDA_VISIBLE_DEVICES': dispositivos_gpu,
                            'threads_per_worker': 8,
                            'memory_limit': LIMITE_MEMORIA_TRABAJADOR,
                            'n_workers': num_gpus,
                            'processes': True,
                        }
                        
                        if interfaz_red:
                            kwargs_cluster['interface'] = interfaz_red
                        
                        with LocalCUDACluster(**kwargs_cluster) as cluster:
                            with Client(cluster) as cliente:
                                cliente.wait_for_workers(num_gpus, timeout=30)
                                
                                resultados_exp = ejecutar_experimento_unico(cliente, info_dataset, num_gpus, parametros, fraccion_datos)
                                
                                info_ejecucion = {
                                    'dataset': nombre_dataset,
                                    'num_gpus': num_gpus,
                                    **parametros
                                }
                                
                                detener_logging_gpu(proceso_log)
                                metricas_gpu = parsear_log_gpu(archivo_log, num_gpus)
                                
                                todos_resultados.append({**info_ejecucion, **resultados_exp, **metricas_gpu})
                                
                    except Exception as e:
                        if 'proceso_log' in locals() and proceso_log.poll() is None:
                            detener_logging_gpu(proceso_log)
                    
                    pd.DataFrame(todos_resultados).to_csv(archivo_resultados, index=False)

    print(f"\nBenchmark terminado. Resultados finales en '{archivo_resultados}'")