"""
    Trabalho IA: Algoritmo Imunológico IRIS
    Nomes:  Alexander Neves Barbosa Júnior
            Davi Paulino Laboissiere Dantas
"""

# Fonte do dataset: https://archive.ics.uci.edu/dataset/53/iris
from ucimlrepo import fetch_ucirepo 


def main():
    # 150 amostras
    iris = fetch_ucirepo(id=53) 
    
    X = iris.data.features 
    y = iris.data.targets 


if __name__ == '__main__':
    main()