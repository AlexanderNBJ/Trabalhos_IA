"""
    Trabalho IA: Algoritmo Imunológico (CLONALG) - IRIS
    Nomes:  Alexander Neves Barbosa Júnior
            Davi Paulino Laboissiere Dantas
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
import os

NUM_CLONES = 10
GERACOES = 30
TAM_POP = 500
TAXA_MUTACAO = 0.2
FATOR_SUPERMUTACAO = 5

def carregar_dataset():
    iris = load_iris()
    X, y = iris.data, iris.target
    nomes_classes = iris.target_names

    return X, y, nomes_classes

def dividir_dataset(X, y, teste=0.3):
    return train_test_split(X, y, test_size=teste, shuffle=True, random_state=42)

def inicializar_populacao(tamanho, num_atributos, num_classes):
    return [np.random.uniform(0, 8, size=(num_classes, num_atributos)) for _ in range(tamanho)]

def afinidade(classificador, X, y):
    predicoes = []
    for amostra in X:
        dist = np.linalg.norm(classificador - amostra, axis=1)
        predicoes.append(np.argmin(dist))
    return accuracy_score(y, predicoes)

def mutar(classificador, intensidade=1.0):
    ruido = np.random.randn(*classificador.shape) * TAXA_MUTACAO * intensidade
    return classificador + ruido

def prever(classificador, X):
    predicoes = []
    for amostra in X:
        dist = np.linalg.norm(classificador - amostra, axis=1)
        predicoes.append(np.argmin(dist))
    return np.array(predicoes)

def evoluir(X_treino, y_treino, X_teste, y_teste, num_classes, num_atributos):
    populacao = inicializar_populacao(TAM_POP, num_atributos, num_classes)

    historico_fitness = []
    historico_treino = []
    historico_teste = []
    melhores_classificadores = []

    for geracao in range(GERACOES):
        afinidades = [afinidade(clf, X_treino, y_treino) for clf in populacao]
        melhores_idx = np.argsort(afinidades)[::-1]
        melhores = [populacao[i] for i in melhores_idx[:5]]

        melhor_fit = max(afinidades)
        media_fit = np.mean(afinidades)
        historico_fitness.append((melhor_fit, media_fit))

        melhor_clf = melhores[0]
        historico_treino.append(afinidade(melhor_clf, X_treino, y_treino))
        historico_teste.append(afinidade(melhor_clf, X_teste, y_teste))
        melhores_classificadores.append(melhor_clf.copy())

        clones = []
        for clf in melhores:
            for _ in range(NUM_CLONES):
                intensidade = FATOR_SUPERMUTACAO * (1 - afinidade(clf, X_treino, y_treino))
                clones.append(mutar(clf, intensidade=intensidade))

        populacao = melhores + clones
        populacao = sorted(populacao, key=lambda clf: afinidade(clf, X_treino, y_treino), reverse=True)
        populacao = populacao[:TAM_POP]

        print(f"Geração {geracao+1}/{GERACOES} | Melhor Fitness: {melhor_fit:.3f} | Média: {media_fit:.3f}")

    return historico_fitness, historico_treino, historico_teste, melhores_classificadores

def gerar_graficos(historico_fitness, historico_treino, historico_teste, nome_pasta="graficos"):
    os.makedirs(nome_pasta, exist_ok=True)

    geracoes = range(1, GERACOES + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(geracoes, [f[0] for f in historico_fitness], label="Melhor Fitness")
    plt.plot(geracoes, [f[1] for f in historico_fitness], label="Média Fitness")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Evolução do Fitness")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(geracoes, historico_treino, label="Treino")
    plt.plot(geracoes, historico_teste, label="Teste")
    plt.xlabel("Geração")
    plt.ylabel("Acurácia")
    plt.title("Acurácia por Geração")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{nome_pasta}/fitness_e_acuracia.png", dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {nome_pasta}/fitness_e_acuracia.png")

def animar_prototipos(X_teste, y_teste, nomes_classes, melhores_classificadores, nome_pasta="graficos"):
    os.makedirs(nome_pasta, exist_ok=True)

    pca = PCA(n_components=2)
    X_teste_pca = pca.fit_transform(X_teste)
    vetores_pca = [pca.transform(v) for v in melhores_classificadores]

    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ["red", "green", "blue"]

    for i in range(3):
        ax.scatter(X_teste_pca[y_teste == i, 0], X_teste_pca[y_teste == i, 1],
                   alpha=0.3, color=cores[i], label=nomes_classes[i])

    prot = ax.scatter([], [], marker='X', s=100, edgecolor='black', linewidth=1.5)
    titulo = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=14)

    ax.set_xlim(X_teste_pca[:, 0].min() - 3, X_teste_pca[:, 0].max() + 3)
    ax.set_ylim(X_teste_pca[:, 1].min() - 3, X_teste_pca[:, 1].max() + 3)
    ax.set_title("Evolução dos Vetores Protótipos (PCA)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    ax.grid(True)

    def atualizar(frame):
        pontos = vetores_pca[frame]
        prot.set_offsets(pontos)
        prot.set_color(cores)
        titulo.set_text(f"Geração {frame + 1}")
        return prot, titulo

    anim = FuncAnimation(fig, atualizar, frames=len(vetores_pca), interval=200, repeat=False)
    anim.save(f"{nome_pasta}/evolucao_prototipos.mp4", fps=5, dpi=150)
    plt.close()
    print(f"Animação salva em: {nome_pasta}/evolucao_prototipos.mp4")

def main():
    X, y, nomes_classes = carregar_dataset()
    X_treino, X_teste, y_treino, y_teste = dividir_dataset(X, y)

    num_classes = len(np.unique(y))
    num_atributos = X.shape[1]

    historico_fitness, hist_treino, hist_teste, melhores = evoluir(
        X_treino, y_treino, X_teste, y_teste, num_classes, num_atributos
    )

    melhor_gen = np.argmax(hist_teste)
    melhor_clf = melhores[melhor_gen]
    pred = prever(melhor_clf, X_teste)
    acc_final = accuracy_score(y_teste, pred)

    print(f"\nAcurácia final no teste: {acc_final:.2f}")
    print(f"Melhor geração: {melhor_gen + 1}\n\n")

    for i, nome in enumerate(nomes_classes):
        print(f"Classe {i} ({nome}): {melhor_clf[i]}")

    print('\n')
    gerar_graficos(historico_fitness, hist_treino, hist_teste)
    animar_prototipos(X_teste, y_teste, nomes_classes, melhores)


if __name__ == "__main__":
    main()
