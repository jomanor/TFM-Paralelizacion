import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import argparse
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from llama2_model import Transformer, ModelArgs
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, PrepareModuleInput, SequenceParallel

def main():

    analizador = argparse.ArgumentParser(description="Benchmark PyTorch TP")
    analizador.add_argument('--dataset_name', type=str, required=True)
    analizador.add_argument('--dataset_config', type=str, default='default')
    analizador.add_argument('--dataset_fraction', type=float, default=1.0)
    analizador.add_argument('--batch_size', type=int, default=16)
    analizador.add_argument('--num_iterations', type=int, default=500)
    analizador.add_argument('--grad_accumulation_steps', type=int, default=1)
    analizador.add_argument('--dim', type=int, default=256)
    analizador.add_argument('--n_layers', type=int, default=2)
    analizador.add_argument('--n_heads', type=int, default=16)
    argumentos = analizador.parse_args()

    try:
        dist.init_process_group("nccl")
        rango = int(os.environ["RANK"])
        rango_local = int(os.environ["LOCAL_RANK"])
        tamaño_mundo = int(os.environ["WORLD_SIZE"])
        
        torch.cuda.set_device(rango_local)
        dispositivo = torch.device(f"cuda:{rango_local}")
            
    except Exception as e:
        print(f"Error en configuración distribuida: {e}")
        raise
    
    malla_dispositivo = init_device_mesh("cuda", (tamaño_mundo,))
    
    configuracion_modelo = ModelArgs(dim=argumentos.dim, n_layers=argumentos.n_layers, n_heads=argumentos.n_heads, vocab_size=32000)
    modelo = Transformer.from_model_args(configuracion_modelo).to(dispositivo)
    modelo.init_weights()

    try:
        modelo = parallelize_module(
            modelo, malla_dispositivo,
            {"tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
             "norm": SequenceParallel(),
             "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())}
        )
        
        for bloque_transformador in modelo.layers:
            capa_atencion = bloque_transformador.attention
            capa_atencion.n_heads = capa_atencion.n_heads // malla_dispositivo.size()
            capa_atencion.n_kv_heads = capa_atencion.n_kv_heads // malla_dispositivo.size()

            plan_tp_capa = {
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(input_layouts=(Shard(1), None), desired_input_layouts=(Replicate(), None)),
                "attention.wq": ColwiseParallel(), 
                "attention.wk": ColwiseParallel(), 
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
                "feed_forward.w1": ColwiseParallel(), 
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)), 
                "feed_forward.w3": ColwiseParallel(),
            }
            parallelize_module(module=bloque_transformador, device_mesh=malla_dispositivo, parallelize_plan=plan_tp_capa)
            
    except Exception as e:
        if rango == 0:
            print(f"Error en la paralelización: {e}")
        raise
    
    tokenizador = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizador.pad_token = tokenizador.eos_token
    
    try:
        conjunto_datos = load_dataset(
            argumentos.dataset_name, 
            argumentos.dataset_config if argumentos.dataset_config != 'default' else None, 
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        if rango == 0:
            print(f"Error cargando dataset {argumentos.dataset_name}: {e}")
        raise
    
    if argumentos.dataset_fraction < 1.0:
        if rango == 0: 
            print(f"Usando {int(argumentos.dataset_fraction * 100)}% del dataset.")
        conjunto_datos = conjunto_datos.train_test_split(train_size=argumentos.dataset_fraction, shuffle=False, seed=42)['train']

    longitud_secuencia = 256
    def procesar_tokenizacion(ejemplos):
        return tokenizador(ejemplos["text"], truncation=True, padding="max_length", max_length=longitud_secuencia, return_tensors="pt")
    
    dataset_tokenizado = conjunto_datos.map(procesar_tokenizacion, batched=True, remove_columns=["text"])
    dataset_tokenizado.set_format("torch")
    
    cargador_datos = DataLoader(dataset_tokenizado, batch_size=argumentos.batch_size, shuffle=True)
    
    tasa_aprendizaje = 3e-3
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje, foreach=True)
    
    if rango == 0:
        tiempo_inicio_entrenamiento = time.time()
        
    iterador_datos = iter(cargador_datos)
    optimizador.zero_grad()

    for i in range(argumentos.num_iterations):
        try:
            lote = next(iterador_datos)
        except StopIteration:
            iterador_datos = iter(cargador_datos)
            lote = next(iterador_datos)

        entrada = lote['input_ids'].to(dispositivo)
        
        salida = modelo(entrada)
        perdida = salida.sum() / argumentos.grad_accumulation_steps
        perdida.backward()

        if (i + 1) % argumentos.grad_accumulation_steps == 0:
            optimizador.step()
            optimizador.zero_grad()

    if rango == 0:
        tiempo_fin_entrenamiento = time.time()
        duracion_entrenamiento = tiempo_fin_entrenamiento - tiempo_inicio_entrenamiento
        print(f"Tiempo puro de entrenamiento: {duracion_entrenamiento:.4f} segundos")

    dist.barrier()

if __name__ == "__main__":
    main()