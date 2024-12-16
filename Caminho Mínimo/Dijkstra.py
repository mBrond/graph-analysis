import heapq
import matplotlib.pyplot as plt
import networkx as nx

class Grafo:
    def __init__(self):
        self.grafo = {}
    
    def adicionar_aresta(self, origem, destino, peso):
        if origem not in self.grafo:
            self.grafo[origem] = []
        self.grafo[origem].append((destino, peso))
        # Adiciona o nó de destino ao grafo, se não existir
        if destino not in self.grafo:
            self.grafo[destino] = []

    def dijkstra(self, inicio):
        # Inicializa as distâncias e a fila de prioridade
        distancias = {no: float('inf') for no in self.grafo}
        distancias[inicio] = 0
        fila_prioridade = [(0, inicio)]  # (distância, nó)
        visitados = set()

        # Define a posição dos nós uma única vez
        pos = nx.spring_layout(self.grafo)  # Posição dos nós

        i = 0
        while fila_prioridade:
            distancia_atual, no_atual = heapq.heappop(fila_prioridade)

            # Se a distância atual for maior que a registrada, ignora
            if distancia_atual > distancias[no_atual]:
                continue

            visitados.add(no_atual)
            self.plotar_grafo(visitados, distancias, pos, i)
            i=i+1

            # Atualiza as distâncias dos vizinhos
            for vizinho, peso in self.grafo.get(no_atual, []):
                if vizinho not in visitados:
                    distancia = distancia_atual + peso

                    if distancia < distancias[vizinho]:
                        distancias[vizinho] = distancia
                        heapq.heappush(fila_prioridade, (distancia, vizinho))

        return distancias

    def plotar_grafo(self, visitados, distancias, pos, iteration):
        G = nx.Graph()

        # Adiciona arestas ao grafo
        for origem, destinos in self.grafo.items():
            for destino, peso in destinos:
                G.add_edge(origem, destino, weight=peso)

        # Desenha o grafo
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_weight='bold')

        # Desenha as arestas com pesos
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

        # Destaca os nós visitados
        nx.draw_networkx_nodes(G, pos, nodelist=visitados, node_color='green')

        # Exibe as distâncias
        for no, distancia in distancias.items():
            plt.text(pos[no][0], pos[no][1] + 0.1, f'Distância: {distancia}', fontsize=12, ha='center')

        plt.title('Algoritmo de Dijkstra - Passo a Passo')
        plt.axis('off')

        plt.savefig(f'dijkstra_iteration_{iteration + 1}.png')

        plt.clf()  # Limpa a figura para o próximo passo

# Exemplo de uso
if __name__ == "__main__":
    grafo = Grafo()
    
    # Adicionando arestas (origem, destino, peso)
    grafo.adicionar_aresta('A', 'B', 2)
    grafo.adicionar_aresta('A', 'C', 4)
    grafo.adicionar_aresta('B', 'D', 7)
    grafo.adicionar_aresta('B', 'C', 1)
    grafo.adicionar_aresta('C', 'D', 2)
    grafo.adicionar_aresta('D', 'E', 2)
    grafo.adicionar_aresta('C', 'E', 10)


    # Executando o algoritmo de Dijkstra a partir do nó 'A'
    distancias = grafo.dijkstra('A')
    
    # Exibindo as distâncias mais curtas
    for no, distancia in distancias.items():
        print(f'Distância do nó A até o nó {no}: {distancia}')

    # plt.show()  # Exibe a última visualização