"""
    Trabalho IA:    Algoritmo Imunológico (CLONALG) - Otimização de Função
                    Usando a função multimodal Alpine02 para encontrar seu máximo global.

                Autores:    Alexander Neves Barbosa Júnior
                Davi Paulino Laboissiere Dantas
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# --- Hiperparâmetros ---
TAM_POP = 100           
GERACOES = 50           
N_SELECAO = 15          
FATOR_CLONE = 0.5       
RHO_MUTACAO = 2.5        
N_SUBSTITUICAO = 10     
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Parâmetros do Problema ---
DIMENSOES = 2
LIMITE_INFERIOR = 0.0
LIMITE_SUPERIOR = 10.0
PASTA_SAIDA = 'graficos_alpine'

# --- Funções do Algoritmo Imunológico ---
def calcular_afinidade(anticorpo):

    x1, x2 = anticorpo

    if x1 < 0 or x2 < 0: return -100

    return (math.sqrt(x1) * math.sin(x1)) * (math.sqrt(x2) * math.sin(x2))

def inicializar_populacao(tamanho, dimensoes, lim_inf, lim_sup):
 
    return [[random.uniform(lim_inf, lim_sup) for _ in range(dimensoes)] for _ in range(tamanho)]

def evoluir_alpine():

    populacao = inicializar_populacao(TAM_POP, DIMENSOES, LIMITE_INFERIOR, LIMITE_SUPERIOR)
    
    # Listas para guardar o histórico para os gráficos
    historico_fitness = []
    melhores_anticorpos_geracao = []
    populacoes_por_geracao = []

    for geracao in range(GERACOES):

        # 1. Avaliação da afinidade de toda a população
        afinidades = [calcular_afinidade(ac) for ac in populacao]
        populacao_com_afinidade = list(zip(afinidades, populacao))
        
        # 2. Seleção e Clonagem
        populacao_ordenada = sorted(populacao_com_afinidade, key=lambda item: item[0], reverse=True)
        anticorpos_selecionados = populacao_ordenada[:N_SELECAO]

        populacao_clones_com_pais = []
        for i, (afinidade, anticorpo) in enumerate(anticorpos_selecionados):
            num_clones = int((FATOR_CLONE * TAM_POP) / (i + 1))
            for _ in range(num_clones):
                populacao_clones_com_pais.append({'clone': anticorpo.copy(), 'afinidade_pai': afinidade})

        # 3. Maturação por Hipermutação
        afinidades_selecionadas = [ac[0] for ac in anticorpos_selecionados]
        max_afinidade = max(afinidades_selecionadas) if afinidades_selecionadas else 0
        min_afinidade = min(afinidades_selecionadas) if afinidades_selecionadas else 0

        populacao_mutada = []

        for item in populacao_clones_com_pais:

            clone, afinidade_pai = item['clone'], item['afinidade_pai']
            afinidade_normalizada = (afinidade_pai - min_afinidade) / (max_afinidade - min_afinidade) if max_afinidade != min_afinidade else 1.0
            alpha = np.exp(-RHO_MUTACAO * afinidade_normalizada) # Mutação inversamente proporcional à afinidade
            
            clone_mutado = []
            
            for valor_dimensao in clone:

                novo_valor = valor_dimensao + alpha * random.gauss(0, 1)
                novo_valor = np.clip(novo_valor, LIMITE_INFERIOR, LIMITE_SUPERIOR) # Garante que o anticorpo não saia do domínio
                clone_mutado.append(novo_valor)

            populacao_mutada.append(clone_mutado)

        # 4. Seleção da Próxima Geração
        afinidades_mutadas = [calcular_afinidade(ac) for ac in populacao_mutada]
        pool_candidatos = populacao_com_afinidade + list(zip(afinidades_mutadas, populacao_mutada))
        pool_ordenado = sorted(pool_candidatos, key=lambda item: item[0], reverse=True)

        num_sobreviventes = TAM_POP - N_SUBSTITUICAO
        sobreviventes = [item[1] for item in pool_ordenado[:num_sobreviventes]]
        novos_aleatorios = inicializar_populacao(N_SUBSTITUICAO, DIMENSOES, LIMITE_INFERIOR, LIMITE_SUPERIOR)
        populacao = sobreviventes + novos_aleatorios

        # 5. Salvar histórico para visualização
        melhor_da_geracao = pool_ordenado[0]
        media_fitness = np.mean([item[0] for item in populacao_com_afinidade])
        historico_fitness.append((melhor_da_geracao[0], media_fitness))
        melhores_anticorpos_geracao.append(melhor_da_geracao[1])
        populacoes_por_geracao.append(populacao.copy())
        
        print(f"Geração {geracao+1}/{GERACOES} | Melhor Afinidade: {melhor_da_geracao[0]:.4f}")

    return historico_fitness, melhores_anticorpos_geracao, populacoes_por_geracao

# --- Gráficos ---
def plotar_evolucao_fitness(historico_fitness, nome_pasta=PASTA_SAIDA):
    os.makedirs(nome_pasta, exist_ok=True)
    geracoes = range(1, len(historico_fitness) + 1)
    melhor_fit = [f[0] for f in historico_fitness]
    media_fit = [f[1] for f in historico_fitness]

    plt.figure(figsize=(10, 6))
    plt.plot(geracoes, melhor_fit, label="Melhor Afinidade (Max)", linewidth=2.5, color='dodgerblue')
    plt.plot(geracoes, media_fit, label="Afinidade Média da População", linestyle='--', color='darkorange')
    
    plt.xlabel("Geração")
    plt.ylabel("Afinidade (Valor da Função)")
    plt.title("Evolução da Afinidade ao Longo das Gerações")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nome_pasta}/evolucao_fitness_alpine.png", dpi=150)
    plt.show()
    plt.close()
    print(f"\nGráfico de fitness salvo em: {nome_pasta}/evolucao_fitness_alpine.png")

# --- Animação ---
def animar_otimizacao(populacoes, melhores_anticorpos, nome_pasta=PASTA_SAIDA):
    os.makedirs(nome_pasta, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Criar o fundo com o mapa de contorno da função
    x = np.linspace(LIMITE_INFERIOR, LIMITE_SUPERIOR, 200)
    y = np.linspace(LIMITE_INFERIOR, LIMITE_SUPERIOR, 200)
    X, Y = np.meshgrid(x, y)
    Z = (np.sqrt(X) * np.sin(X)) * (np.sqrt(Y) * np.sin(Y))
    
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis_r')
    fig.colorbar(contour, ax=ax, label='Valor da Função (Afinidade)')

    # 2. Preparar os pontos que serão animados
    pontos_populacao = ax.scatter([], [], color='black', s=15, alpha=0.6, label='Anticorpos')
    melhor_ponto = ax.scatter([], [], color='cyan', s=100, marker='o', edgecolor='black', linewidth=1, label='Melhor Anticorpo')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Evolução dos Anticorpos na Função Alpine02")
    ax.legend()
    ax.set_xlim(LIMITE_INFERIOR, LIMITE_SUPERIOR)
    ax.set_ylim(LIMITE_INFERIOR, LIMITE_SUPERIOR)

    def atualizar(frame):
        # Atualiza a posição dos pontos da população
        pop_coords = np.array(populacoes[frame])
        pontos_populacao.set_offsets(pop_coords)
        
        # Atualiza a posição do melhor anticorpo
        melhor_anticorpo = melhores_anticorpos[frame]
        melhor_ponto.set_offsets(melhor_anticorpo)
        
        # Atualiza o título do gráfico para refletir a geração atual
        ax.set_title(f'Geração {frame + 1}/{len(populacoes)}')
        return pontos_populacao, melhor_ponto

    anim = FuncAnimation(fig, atualizar, frames=len(populacoes), interval=333, blit=False, repeat=False)
    anim.save(f"{nome_pasta}/otimizacao_alpine.mp4", fps=3, dpi=150, writer='ffmpeg')
    
    print(f"Animação salva em: {nome_pasta}/otimizacao_alpine.mp4")
    plt.show()
    plt.close()

# --- Main ---
def main():
    # Executa o algoritmo para obter os resultados
    historico_fitness, melhores, populacoes = evoluir_alpine()
    
    # Encontra o melhor resultado global de todas as gerações
    melhor_afinidade_geral = max([f[0] for f in historico_fitness])
    melhor_anticorpo_geral = melhores[np.argmax([f[0] for f in historico_fitness])]
    
    print(f"\n Valor Máximo (Afinidade) Encontrado: {melhor_afinidade_geral:.5f}")
    print(f"Solução (x*) Encontrada: [{melhor_anticorpo_geral[0]:.5f}, {melhor_anticorpo_geral[1]:.5f}]")

    # Gera as visualizações (gráfico e animação)
    plotar_evolucao_fitness(historico_fitness)
    animar_otimizacao(populacoes, melhores)

if __name__ == "__main__":
    main()