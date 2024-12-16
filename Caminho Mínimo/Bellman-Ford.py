import matplotlib.pyplot as plt
import networkx as nx

class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, weight))
        # Ensure the destination node is also in the graph
        if v not in self.graph:
            self.graph[v] = []

    def bellman_ford(self, start):
        # Initialize distances from start to all other vertices as infinite
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        num_vertices = len(self.graph)

        # Define the layout and style for the graph
        pos = nx.spring_layout(self.graph)  # Fixed layout for consistency

        # Relax edges repeatedly
        for i in range(num_vertices - 1):
            for u in self.graph:
                for v, weight in self.graph[u]:
                    if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
            self.plot_graph(distances, pos, i)

        # Check for negative-weight cycles
        for u in self.graph:
            for v, weight in self.graph[u]:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    print("Graph contains a negative-weight cycle")
                    return None

        return distances

    def plot_graph(self, distances, pos, iteration):
        G = nx.DiGraph()  # Use nx.DiGraph for directed edges

        # Add edges to the graph
        for u, destinations in self.graph.items():
            for v, weight in destinations:
                G.add_edge(u, v, weight=weight)

        # Draw the graph
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_weight='bold', arrowsize=20)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

        # Display distances
        for node, distance in distances.items():
            plt.text(pos[node][0], pos[node][1] + 0.1, f'Distância: {distance}', fontsize=12, ha='center')

        # Add the "passo X" text at the top with a specific color
        plt.text(0.5, 1.05, f'Passo {iteration + 1}', fontsize=16, ha='center', va='center', 
                 transform=plt.gca().transAxes, color='blue')  # Cor da fonte alterada para azul

        plt.title(f'Algoritmo de Bellman-Ford - Iteração {iteration + 1}')

        plt.axis('off')

        # Save the plot as an image
        plt.savefig(f'bellman_ford_iteration_{iteration + 1}.png')  # Save as PNG
        plt.clf()  # Clear the figure for the next step

# Example usage
if __name__ == "__main__":
    graph = Graph()
    
    # Adding edges (source, destination, weight)
    graph.add_edge('S', 'A', 10)
    graph.add_edge('S', 'E', 8)
    graph.add_edge('E', 'D', 1)
    graph.add_edge('D', 'A', -4)
    graph.add_edge('D', 'C', -1)
    graph.add_edge('C', 'B', -2)
    graph.add_edge('B', 'A', 1)
    graph.add_edge('A', 'C', 2)

    # Running the Bellman-Ford algorithm from node 'A'
    distances = graph.bellman_ford('S')
    
    # Displaying the shortest distances
    if distances:
        for node, distance in distances.items():
            print(f'Distância do nó S até o nó {node}: {distance}')

    # plt.show()  # Show the final visualization