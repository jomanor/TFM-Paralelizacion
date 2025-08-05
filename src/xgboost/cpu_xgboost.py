import xgboost as xgb
import pandas as pd
from sklearn.datasets import load_svmlight_file
import time
import os
import itertools
import numpy as np

def preprocesar_datos_taxis(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    
    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    return df

def cargar_y_preparar_datos(info_dataset):
    ruta_datos = info_dataset['path']
    formato_datos = info_dataset['format']
    
    X, y = None, None
    
    if formato_datos == 'csv':
        opcion_header = 'infer' if info_dataset.get('has_header') else None
        separador = info_dataset.get('separator', ',')
        
        df = pd.read_csv(ruta_datos, header=opcion_header, sep=separador, low_memory=False)

        if info_dataset.get('needs_preprocessing'):
            df = preprocesar_datos_taxis(df)
        
        columna_etiqueta = info_dataset.get('label_column')
        if not columna_etiqueta:
            indice_columna_etiqueta = info_dataset.get('label_column_index', -1)
            columna_etiqueta = df.columns[indice_columna_etiqueta]

        y = df[columna_etiqueta]
        X = df.drop(columns=[columna_etiqueta])

        X = X.select_dtypes(include=np.number)

    elif formato_datos == 'svmlight':
        X, y = load_svmlight_file(ruta_datos, n_features=info_dataset.get('n_features'))

    if info_dataset.get('remap_labels', False):
        y = y - 1
        
    return X, y

def ejecutar_experimento_unico(info_dataset, num_nucleos, parametros_xgb):
    X, y = cargar_y_preparar_datos(info_dataset)

    parametros = {
        "n_jobs": num_nucleos,
        "tree_method": "hist",
        **parametros_xgb
    }
    
    modelo = None
    if info_dataset['task_type'] == 'classification':
        if info_dataset['num_class'] == 2:
            parametros['objective'] = 'binary:logistic'
            parametros['eval_metric'] = 'auc'
        else:
            parametros['objective'] = 'multi:softprob'
            parametros['eval_metric'] = 'mlogloss'
            parametros['num_class'] = info_dataset['num_class']
        modelo = xgb.XGBClassifier(**parametros)
    else:
        parametros['objective'] = 'reg:squarederror'
        parametros['eval_metric'] = 'rmse'
        modelo = xgb.XGBRegressor(**parametros)

    tiempo_inicio = time.time()
    modelo.fit(X, y)
    tiempo_fin = time.time()
    
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio
    
    return {'training_time_s': tiempo_entrenamiento}

if __name__ == "__main__":
    DIRECTORIO_DATOS = "/una/ruta/cualquiera/"
    
    DATASETS = {
        'covtype': {'path': os.path.join(DIRECTORIO_DATOS, 'covtype.csv'), 'format': 'csv', 'task_type': 'classification', 'num_class': 7, 'remap_labels': True},
        'microsoft': {'path': os.path.join(DIRECTORIO_DATOS, 'Microsoft'), 'format': 'svmlight', 'task_type': 'classification', 'num_class': 5, 'n_features': 136},
        'susy': {'path': os.path.join(DIRECTORIO_DATOS, 'SUSY.csv'), 'format': 'csv', 'task_type': 'classification', 'num_class': 2, 'label_column_index': 0},
        'taxis_usa': {'path': os.path.join(DIRECTORIO_DATOS, 'taxis_usa.csv'), 'format': 'csv', 'task_type': 'regression', 'has_header': True, 'label_column': 'total_amount', 'needs_preprocessing': True},
        'yearprediction': {'path': os.path.join(DIRECTORIO_DATOS, 'YearPredictionMSD.csv'), 'format': 'csv', 'task_type': 'regression', 'label_column_index': 0},
        'epsilon_t': {'path': os.path.join(DIRECTORIO_DATOS, 'epsilon_normalized.t'), 'format': 'svmlight', 'task_type': 'classification', 'num_class': 2, 'n_features': 2000, 'remap_labels': True}
    }
    
    CONFIGS_NUCLEOS_CPU = [1, 5, 10, 15, 20]
    FRACCIONES_DATOS = [0.25, 0.5, 1.0]
    PARAMETROS = {
        'n_estimators': [100, 500],
        'max_depth': [6, 10]
    }
    
    todos_resultados = []
    archivo_resultados = "archivo.csv"
    
    combinaciones_parametros = [dict(zip(PARAMETROS.keys(), v)) for v in itertools.product(*PARAMETROS.values())]

    for nombre_dataset, info_dataset in DATASETS.items():
        for data_fraction in FRACCIONES_DATOS:
            for num_nucleos in CONFIGS_NUCLEOS_CPU:
                for parametros in combinaciones_parametros:
                
                    try:
                        resultados_exp = ejecutar_experimento_unico(info_dataset, num_nucleos, parametros)
                        
                        info_ejecucion = {
                            'dataset': nombre_dataset,
                            'num_cores': num_nucleos,
                            **parametros
                        }
                        
                        todos_resultados.append({**info_ejecucion, **resultados_exp})
                                
                    except Exception as e:
                        pass
                    
                    pd.DataFrame(todos_resultados).to_csv(archivo_resultados, index=False)