import matplotlib.pyplot as plt
import networkx as nx
import os

class FlowNetworkDFS:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        self.flow = [[0] * num_nodes for _ in range(num_nodes)]
        self.frames_dir = "frames_dfs"
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        self.pos = None  # Armazenar as posições fixas dos vértices

    def add_edge(self, u, v, capacity):
        self.adj_matrix[u][v] = capacity

    # DFS para encontrar o caminho de aumento
    def dfs(self, source, sink, parent, visited):
        if source == sink:
            return True
        
        visited[source] = True
        
        for neighbor in range(self.num_nodes):
            # Se o vizinho não foi visitado e há capacidade residual
            if not visited[neighbor] and self.adj_matrix[source][neighbor] > self.flow[source][neighbor]:
                parent[neighbor] = source
                if self.dfs(neighbor, sink, parent, visited):
                    return True
        
        return False

    def visualize_graph(self, frame_number, parent=None, total_flow=None):
        G = nx.DiGraph()
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if self.adj_matrix[u][v] > 0:  # Só adicionar arestas com capacidade
                    G.add_edge(u, v, capacity=self.adj_matrix[u][v], flow=self.flow[u][v])

        if self.pos is None:
            # Definindo as posições fixas manualmente para os vértices
            self.pos = {
                0: (0, 0),  # Nó 0
                1: (1, 2),  # Nó 1
                2: (2, 1),  # Nó 2
                3: (3, 2),  # Nó 3
                4: (1, 0),  # Nó 4
                5: (3, 1),  # Nó 5
                6: (4, 1),  # Nó 6
                7: (5, 0)   # Nó 7
            }

        capacities = nx.get_edge_attributes(G, 'capacity')
        flow = nx.get_edge_attributes(G, 'flow')

        # Destacar o caminho atual percorrido
        if parent is not None:
            path_edges = []
            node = self.num_nodes - 1  # Começa do sink
            while parent[node] != -1:
                path_edges.append((parent[node], node))
                node = parent[node]
            path_edges.reverse()

            edge_colors = ['red' if (u, v) in path_edges else 'black' for u, v in G.edges()]
        else:
            edge_colors = ['black'] * len(G.edges())

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos=self.pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold', arrowsize=20, edge_color=edge_colors)

        edge_labels = {(u, v): f"{flow.get((u, v), 0)}/{capacities.get((u, v), 0)}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos=self.pos, edge_labels=edge_labels)

        # Exibir o valor total de fluxo acima do nó de destino (sink)
        if total_flow is not None:
            plt.text(self.pos[self.num_nodes - 1][0], self.pos[self.num_nodes - 1][1] + 0.1,
                     f"Max Flow: {total_flow}", fontsize=12, ha='center')

        plt.title(f"Flow Network DFS (Ford-Fulkerson) - Iteration {frame_number}")
        plt.savefig(f"{self.frames_dir}/frame_{frame_number}.png")
        plt.close()

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        parent = [-1] * self.num_nodes
        frame_number = 0
        self.visualize_graph(frame_number)  # Visualizar o grafo antes da execução

        while True:
            visited = [False] * self.num_nodes
            if not self.dfs(source, sink, parent, visited):
                break

            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.adj_matrix[parent[s]][s] - self.flow[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

            # Atualiza os fluxos residuais
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

            frame_number += 1
            self.visualize_graph(frame_number, parent, max_flow)  # Visualizar o grafo após cada iteração

        return max_flow


# Exemplo de uso:
fn_dfs = FlowNetworkDFS(8)
fn_dfs.add_edge(0, 1, 10)
fn_dfs.add_edge(0, 4, 10)
fn_dfs.add_edge(1, 3, 10)
fn_dfs.add_edge(1, 2, 10)
fn_dfs.add_edge(2, 5, 5)
fn_dfs.add_edge(3, 5, 10)
fn_dfs.add_edge(4, 2, 10)
fn_dfs.add_edge(4, 3, 5)
fn_dfs.add_edge(5, 6, 15)
fn_dfs.add_edge(5, 7, 10)
fn_dfs.add_edge(6, 7, 10)

source = 0
sink = 7
max_flow_dfs = fn_dfs.ford_fulkerson(source, sink)
print(f"Max Flow using DFS (Ford-Fulkerson): {max_flow_dfs}")
