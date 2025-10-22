"""
    Trabalho IA:    Algoritmo Imunológico (CLONALG) - IRIS 
                    Usando eixo 2D: petal length x petal width
                    Com Hipermutação nos clones

    Autores:    Alexander Neves Barbosa Júnior
                Davi Paulino Laboissiere Dantas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

NUM_CLONES = 8
GERACOES = 30
TAM_POP = 300
TAXA_MUTACAO = 0.1
SEED = 42
np.random.seed(SEED)

FEATURE_INDICES = [2, 3]  # X = petal length, Y = petal width
N_BEST = 5
PASTA_SAIDA = 'graficos_iris'

def carregar_dataset_2d():
    iris = load_iris()
    X_full, y = iris.data[:, FEATURE_INDICES], iris.target
    nomes = iris.target_names
    return X_full, y, nomes

def dividir_dataset(X, y, teste=0.3):
    return train_test_split(X, y, test_size=teste, shuffle=True, random_state=SEED)

def inicializar_populacao(tamanho, num_atributos, num_classes, mins, maxs):
    pop = []
    for _ in range(tamanho):
        clf = np.zeros((num_classes, num_atributos))
        for f in range(num_atributos):
            clf[:, f] = np.random.uniform(mins[f], maxs[f], size=(num_classes,))
        pop.append(clf)
    return pop

def fitness(individuo, X, y):
    return sum(np.linalg.norm(individuo[y[i]] - X[i]) for i in range(len(X)))

def fitness_por_classe(individuo, X, y, num_classes):
    fit_classes = np.zeros(num_classes)
    for c in range(num_classes):
        Xc = X[y == c]
        fit_classes[c] = sum(np.linalg.norm(individuo[c] - x) for x in Xc)
    return fit_classes

def mutar_hipermutacao(individuo, mins, maxs, parent_fitness, fitness_min, fitness_max, base_sigma=TAXA_MUTACAO, eps=1e-9):
    # normaliza: 0 para melhor (fitness_min), 1 para pior (fitness_max)
    norm = (parent_fitness - fitness_min) / (fitness_max - fitness_min + eps)
    # escala da mutação aumenta para piores pais
    sigma = base_sigma * (1.0 + norm * 4.0)
    ruido = np.random.randn(*individuo.shape) * sigma
    novo = individuo + ruido
    novo = np.clip(novo, mins, maxs)
    return novo

def selecionar_melhores(pop, X, y, k):
    avaliacao = [(ind, fitness(ind, X, y)) for ind in pop]
    avaliacao.sort(key=lambda x: x[1])
    return [ind for ind, _ in avaliacao[:k]]

def evoluir(X_treino, y_treino, X_teste, y_teste, num_classes, num_atributos, mins, maxs, nomes_classes):
    populacao = inicializar_populacao(TAM_POP, num_atributos, num_classes, mins, maxs)
    historico_fitness = []
    historico_fitness_classes = []
    melhores_classificadores = []
    populacoes_por_geracao = []

    for geracao in range(GERACOES):
        melhores = selecionar_melhores(populacao, X_treino, y_treino, N_BEST)
        melhor_fit = fitness(melhores[0], X_treino, y_treino)
        media_fit = np.mean([fitness(ind, X_treino, y_treino) for ind in populacao])

        fit_classes = fitness_por_classe(melhores[0], X_treino, y_treino, num_classes)
        historico_fitness.append((melhor_fit, media_fit))
        historico_fitness_classes.append(fit_classes)

        melhores_classificadores.append(melhores[0].copy())
        populacoes_por_geracao.append(populacao.copy())

        # --- clonagem com hipermutação ---
        fitness_melhores = [fitness(clf, X_treino, y_treino) for clf in melhores]
        fmin, fmax = min(fitness_melhores), max(fitness_melhores)

        clones = []
        for i, clf in enumerate(melhores):
            pf = fitness_melhores[i]
            for _ in range(NUM_CLONES):
                clones.append(mutar_hipermutacao(clf, mins, maxs, pf, fmin, fmax))

        populacao = melhores + clones
        if len(populacao) > TAM_POP:
            populacao = selecionar_melhores(populacao, X_treino, y_treino, TAM_POP)

        # Print com fitness por classe
        print(f"Geração {geracao+1}/{GERACOES} | Melhor Fitness: {melhor_fit:} | Média: {media_fit}")
        for i in range(num_classes):
            print(f"{nomes_classes[i]} = {fit_classes[i]}")
        print()

    return historico_fitness, historico_fitness_classes, melhores_classificadores, populacoes_por_geracao

def plotar_fitness(historico_fitness, historico_fitness_classes, nomes_classes, nome_pasta=PASTA_SAIDA):
    os.makedirs(nome_pasta, exist_ok=True)
    geracoes = range(1, len(historico_fitness)+1)
    melhor = [f[0] for f in historico_fitness]
    media = [f[1] for f in historico_fitness]

    plt.figure(figsize=(10,5))
    plt.plot(geracoes, melhor, label="Melhor Fitness Total", linewidth=2)
    plt.plot(geracoes, media, label="Média Fitness Total", linewidth=2)

    cores = ["#FF7F0E", "#2CA02C", "#1F77B4"]
    for i, nome in enumerate(nomes_classes):
        fit_classe = [f[i] for f in historico_fitness_classes]
        plt.plot(geracoes, fit_classe, label=f"Fitness {nome}", linestyle='--', color=cores[i])

    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Evolução do Fitness (Total e por Classe)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{nome_pasta}/evolucao_fitness_por_classe.png", dpi=150)
    plt.show()
    plt.close()
    print(f"Gráfico de fitness salvo em: {nome_pasta}/evolucao_fitness_por_classe.png")

def animar_prototipos_2d_com_populacao(X, y, nomes_classes, populacoes, melhores_classificadores, nome_pasta=PASTA_SAIDA):
    os.makedirs(nome_pasta, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,6))
    cores = ["#FF7F0E", "#2CA02C", "#1F77B4"]

    for i in range(len(nomes_classes)):
        ax.scatter(X[y==i,1], X[y==i,0], alpha=0.5, s=30, color=cores[i], label=nomes_classes[i])

    pontos_geracao = ax.scatter([], [], color='black', s=40, alpha=0.5)
    melhor_plots = [ax.scatter([], [], s=140, color=cores[i], edgecolor='black', linewidth=1.2, marker='X') for i in range(len(nomes_classes))]
    texts = [ax.text(0,0,'', fontsize=9, weight='bold') for _ in range(len(nomes_classes))]

    ax.set_xlabel('Petal width (cm)')
    ax.set_ylabel('Petal length (cm)')
    ax.set_title('Evolução dos Protótipos com População')
    ax.legend()
    ax.grid(True)

    x_min, x_max = X[:,1].min(), X[:,1].max()
    y_min, y_max = X[:,0].min(), X[:,0].max()
    dx = (x_max - x_min) * 0.15
    dy = (y_max - y_min) * 0.15
    ax.set_xlim(x_min - dx, x_max + dx)
    ax.set_ylim(y_min - dy, y_max + dy)

    def atualizar(frame):
        pop = populacoes[frame]
        melhor = melhores_classificadores[frame]

        # população
        pop_coords = np.vstack([np.column_stack((p[:,1], p[:,0])) for p in pop])
        pontos_geracao.set_offsets(pop_coords)

        # melhores protótipos
        for i, scatter_best in enumerate(melhor_plots):
            scatter_best.set_offsets([melhor[i,1], melhor[i,0]])
            texts[i].set_position((melhor[i,1]+0.02*(x_max-x_min), melhor[i,0]+0.02*(y_max-y_min)))
            texts[i].set_text(nomes_classes[i])

        ax.set_title(f'Geração {frame+1} / {len(melhores_classificadores)}')
        return [pontos_geracao] + melhor_plots + texts

    anim = FuncAnimation(fig, atualizar, frames=len(melhores_classificadores), interval=300, repeat=False)
    anim.save(f"{nome_pasta}/evolucao_prototipos_com_populacao.mp4", fps=5, dpi=150)
    print(f"Animação salva em: {nome_pasta}/evolucao_prototipos_com_populacao.mp4")
    plt.show()
    plt.close()

def main():
    X, y, nomes = carregar_dataset_2d()
    X_treino, X_teste, y_treino, y_teste = dividir_dataset(X, y)

    num_classes = len(np.unique(y))
    num_atributos = X.shape[1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    historico_fitness, historico_fitness_classes, melhores, populacoes_por_geracao = evoluir(
        X_treino, y_treino, X_teste, y_teste,
        num_classes, num_atributos, mins, maxs, nomes
    )

    # gráficos
    plotar_fitness(historico_fitness, historico_fitness_classes, nomes)
    # animação
    animar_prototipos_2d_com_populacao(X_teste, y_teste, nomes, populacoes_por_geracao, melhores)

if __name__ == "__main__":
    main()
