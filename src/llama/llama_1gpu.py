import sys
import os
import torch
import torch.nn as nn
import argparse
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from llama2_model import Transformer, ModelArgs

def main():

    analizador = argparse.ArgumentParser(description="Benchmark PyTorch 1GPU")
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
    
    configuracion_modelo = ModelArgs(dim=argumentos.dim, n_layers=argumentos.n_layers, n_heads=argumentos.n_heads, vocab_size=32000)
    modelo = Transformer.from_model_args(configuracion_modelo).to("cuda")
    modelo.init_weights()
    
    tokenizador = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizador.pad_token = tokenizador.eos_token
    
    conjunto_datos = load_dataset(
        argumentos.dataset_name, 
        argumentos.dataset_config if argumentos.dataset_config != 'default' else None, 
        split="train",
        trust_remote_code=True
    )

    if argumentos.dataset_fraction < 1.0:
        print(f"Usando {int(argumentos.dataset_fraction * 100)}% del dataset.")
        conjunto_datos = conjunto_datos.train_test_split(train_size=argumentos.dataset_fraction, shuffle=True, seed=42)['train']

    longitud_secuencia = 256
    def procesar_tokenizacion(ejemplos):
        return tokenizador(ejemplos["text"], truncation=True, padding="max_length", max_length=longitud_secuencia, return_tensors="pt")
    
    dataset_tokenizado = conjunto_datos.map(procesar_tokenizacion, batched=True, remove_columns=["text"])
    dataset_tokenizado.set_format("torch")
    cargador_datos = DataLoader(dataset_tokenizado, batch_size=argumentos.batch_size, shuffle=True)

    tasa_aprendizaje = 3e-3
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=tasa_aprendizaje, foreach=True)
    
    tiempo_inicio_entrenamiento = time.time()
    
    iterador_datos = iter(cargador_datos)
    optimizador.zero_grad() 

    for i in range(argumentos.num_iterations):
        try:
            lote = next(iterador_datos)
        except StopIteration:
            iterador_datos = iter(cargador_datos)
            lote = next(iterador_datos)

        entrada = lote['input_ids'].to("cuda")
        
        salida = modelo(entrada)
        perdida = salida.sum() / argumentos.grad_accumulation_steps
        
        perdida.backward()
        
        if (i + 1) % argumentos.grad_accumulation_steps == 0:
            optimizador.step()
            optimizador.zero_grad()

    tiempo_fin_entrenamiento = time.time()
    duracion_entrenamiento = tiempo_fin_entrenamiento - tiempo_inicio_entrenamiento
    print(f"Tiempo puro de entrenamiento: {duracion_entrenamiento:.4f} segundos")

if __name__ == "__main__":
    main()