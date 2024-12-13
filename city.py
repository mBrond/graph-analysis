import networkx as nx
import matplotlib.pyplot as plt
import random


class CityFlowNetwork:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        self.flow = [[0] * num_nodes for _ in range(num_nodes)]
        self.pos = None  # Layout dos vértices

    def add_edge(self, u, v, capacity):
        self.adj_matrix[u][v] = capacity

    def generate_city_layout(self):
        # Gera um layout em grade para simular ruas e cruzamentos
        size = int(self.num_nodes**0.5)  # Tamanho da grade
        G = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
        self.pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Inverte o eixo y para parecer mais "urbano"
        return G
 
    def visualize_city_graph(self, frame_number=None):
        # Cria o grafo baseado na matriz de adjacência
        G = nx.DiGraph()
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if self.adj_matrix[u][v] > 0:
                    G.add_edge(u, v, capacity=self.adj_matrix[u][v], flow=self.flow[u][v])

        # Gera um layout de cidade (grade) se não estiver definido
        if self.pos is None:
            size = int(self.num_nodes**0.5)  # Tamanho da grade
            city_graph = nx.grid_2d_graph(size, size, create_using=nx.DiGraph)
            self.pos = {i: (x, -y) for i, (x, y) in enumerate(city_graph.nodes())}

        # Capacidade das arestas
        capacities = nx.get_edge_attributes(G, 'capacity')
        edge_labels = {edge: f"{cap}" for edge, cap in capacities.items()}

        plt.figure(figsize=(10, 8))
        nx.draw(
            G, pos=self.pos, with_labels=True, node_size=700, node_color='lightblue',
            font_size=10, arrowsize=20, edge_color='black'
        )
        nx.draw_networkx_edge_labels(G, pos=self.pos, edge_labels=edge_labels)

        plt.title(f"Mapa Urbano - Iteração {frame_number}" if frame_number else "Mapa Urbano")
        plt.savefig(f"city_graph_{frame_number if frame_number else 'final'}.png")
        plt.close()


# Exemplo de uso
if __name__ == "__main__":
    num_nodes = 25  # Reduzimos para um mapa urbano pequeno (5x5)
    flow_network = CityFlowNetwork(num_nodes)

    # Adiciona arestas com capacidades aleatórias
    for _ in range(50):  # Número de arestas
        u, v = random.sample(range(num_nodes), 2)
        capacity = random.randint(1, 20)
        flow_network.add_edge(u, v, capacity)

    # Visualiza o grafo em formato de mapa urbano
    flow_network.visualize_city_graph()
