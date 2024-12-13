import concurrent
import matplotlib.pyplot as plt
import networkx as nx
import random
import os


class FlowNetwork:
    def __init__(self, num_nodes, algorithm):
        self.num_nodes = num_nodes
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        self.flow = [[0] * num_nodes for _ in range(num_nodes)]
        self.frames_dir = f"frames_{algorithm.lower()}"
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        self.pos = None  # Layout dos vértices
        self.algorithm = algorithm

    def add_edge(self, u, v, capacity):
        self.adj_matrix[u][v] = capacity

    def bfs(self, source, sink, parent):
        visited = [False] * self.num_nodes
        queue = [source]
        visited[source] = True

        while queue:
            node = queue.pop(0)
            for neighbor, capacity in enumerate(self.adj_matrix[node]):
                if not visited[neighbor] and capacity > self.flow[node][neighbor]:
                    parent[neighbor] = node
                    if neighbor == sink:
                        return True
                    queue.append(neighbor)
                    visited[neighbor] = True
        return False

    def dfs(self, source, sink, parent, visited):
        if source == sink:
            return True

        visited[source] = True
        for neighbor, capacity in enumerate(self.adj_matrix[source]):
            if not visited[neighbor] and capacity > self.flow[source][neighbor]:
                parent[neighbor] = source
                if self.dfs(neighbor, sink, parent, visited):
                    return True

        return False

    def visualize_graph(self, frame_number, parent=None):
        G = nx.DiGraph()
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if self.adj_matrix[u][v] > 0:
                    G.add_edge(u, v, capacity=self.adj_matrix[u][v], flow=self.flow[u][v])

        if self.pos is None:
            self.pos = nx.circular_layout(G)  # Disposição fixa e uniforme

        capacities = nx.get_edge_attributes(G, 'capacity')
        flow = nx.get_edge_attributes(G, 'flow')

        edge_colors = ['red' if parent and (parent[v] == u) else 'black' for u, v in G.edges()]
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, pos=self.pos, with_labels=True, node_size=700, node_color='lightblue',
            font_size=10, arrowsize=20, edge_color=edge_colors
        )
        edge_labels = {(u, v): f"{flow.get((u, v), 0)}/{capacities.get((u, v), 0)}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos=self.pos, edge_labels=edge_labels)

        plt.title(f"{self.algorithm} - Iteração {frame_number}")
        plt.savefig(f"{self.frames_dir}/frame_{frame_number}.png")
        plt.close()

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        parent = [-1] * self.num_nodes
        frame_number = 0
        self.visualize_graph(frame_number)

        while True:
            if self.algorithm == "Edmonds-Karp":
                path_found = self.bfs(source, sink, parent)
            elif self.algorithm == "Ford-Fulkerson":
                visited = [False] * self.num_nodes
                path_found = self.dfs(source, sink, parent, visited)
            else:
                raise ValueError("Algoritmo desconhecido!")

            if not path_found:
                break

            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.adj_matrix[parent[s]][s] - self.flow[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

            frame_number += 1
            self.visualize_graph(frame_number, parent)

        return max_flow
import time
import random
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')

class FlowNetwork:
    def __init__(self, num_nodes, algorithm):
        self.num_nodes = num_nodes
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        self.flow = [[0] * num_nodes for _ in range(num_nodes)]
        self.algorithm = algorithm

    def add_edge(self, u, v, capacity):
        self.adj_matrix[u][v] = capacity

    def bfs(self, source, sink, parent):
        visited = [False] * self.num_nodes
        queue = [source]
        visited[source] = True

        while queue:
            node = queue.pop(0)
            for neighbor, capacity in enumerate(self.adj_matrix[node]):
                if not visited[neighbor] and capacity > self.flow[node][neighbor]:
                    parent[neighbor] = node
                    if neighbor == sink:
                        return True
                    queue.append(neighbor)
                    visited[neighbor] = True
        return False

    def ford_fulkerson(self, source, sink):
        max_flow = 0
        parent = [-1] * self.num_nodes

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.adj_matrix[parent[s]][s] - self.flow[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

        return max_flow

    def edmonds_karp(self, source, sink):
        max_flow = 0
        parent = [-1] * self.num_nodes

        while self.bfs(source, sink, parent):
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.adj_matrix[parent[s]][s] - self.flow[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = parent[v]

        return max_flow

def generate_edges(num_nodes, density, capacity_range):
    edges = set()
    max_edges = int(density * num_nodes * (num_nodes - 1) / 2)
    while len(edges) < max_edges:
        u, v = random.sample(range(num_nodes), 2)
        if (u, v) not in edges and (v, u) not in edges:
            capacity = random.randint(*capacity_range)
            edges.add((u, v, capacity))
    return list(edges)

def run_test(scenario, algorithm, num_nodes):
    density = scenario.get("density", 0.5)
    capacity_range = scenario.get("capacity_range", (10, 50))

    # Generate edges
    edges = generate_edges(num_nodes, density, capacity_range)

    # Create graph
    fn = FlowNetwork(num_nodes, algorithm)
    for u, v, capacity in edges:
        fn.add_edge(u, v, capacity)

    # Choose source and sink
    source = random.randint(0, num_nodes // 2)
    sink = random.randint(num_nodes // 2, num_nodes - 1)

    # Execute the algorithm
    start_time = time.time()
    if algorithm == "Ford-Fulkerson":
        max_flow = fn.ford_fulkerson(source, sink)
    elif algorithm == "Edmonds-Karp":
        max_flow = fn.edmonds_karp(source, sink)
    end_time = time.time()

    return {
        "scenario": scenario["description"],
        "algorithm": algorithm,
        "max_flow": max_flow,
        "time": end_time - start_time,
    }

if __name__ == "__main__":
    num_nodes = 1500
    scenarios = [
        {"description": "Grafo esparso", "density": 0.2},
        {"description": "Grafo denso", "density": 0.7},
        {"description": "Grandes capacidades", "capacity_range": (50, 100), "density": 0.4},
        {"description": "Pequenas capacidades", "capacity_range": (1, 10), "density": 0.4},
    ]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_test, scenario, algorithm, num_nodes)
            for scenario in scenarios
            for algorithm in ["Ford-Fulkerson", "Edmonds-Karp"]
        ]

        for future in futures:
            results.append(future.result())

    for result in results:
        print(f"Teste: {result['scenario']}, Algoritmo: {result['algorithm']}")
        print(f"Fluxo Máximo: {result['max_flow']}")
        print(f"Tempo de execução: {result['time']:.4f} segundos")
        print("-" * 50)
