import networkx as nx
import json
import matplotlib.pyplot as plt
from itertools import combinations 

# the graph and things
def build_graph(dataset):
    '''
    this takes a list of dictionaries, and builds a graph based on those datasets, chosing a new color for each dataset. 
    '''
    graph = nx.Graph()
    colors = ['blue', 'red']
    edge_colors = ['lightblue', 'pink']
    sizes = [25, 100]
    weights = [0.2,50]
    for i in range(len(dataset)):
        papers = dataset[i]
        size = sizes[i]
        color = colors[i]
        edge_color = edge_colors[i]
        weight = weights[i]
        for authorIDs in papers.values():
            for authorID in authorIDs:
                graph.add_node(authorID, color = color, size = size)
            for combo in combinations(authorIDs, 2):
                graph.add_edge(combo[0], combo[1], weight=weight, color = edge_color)
    return graph

def draw_graph(graph):
    pos = nx.spring_layout(graph, weight='weight') 
    edge_colors = [graph.edges[n].get('color', 'gray') for n in graph.edges]
    node_colors = [graph.nodes[n].get('color', 'gray') for n in graph.nodes]
    sizes = [graph.nodes[n].get("size", 300) for n in graph.nodes]

    nx.draw(graph,pos, with_labels=False, node_color=node_colors, node_size=sizes, edge_color=edge_colors)
    plt.show()


if __name__ == '__main__':
    with open('round1.json', 'r') as f:
        papers_round_1 = json.load(f)
    with open('round2.json', 'r') as f:
        papers_round_2 = json.load(f)
    print(len(papers_round_2))

    graph = build_graph([papers_round_2,papers_round_1])
    draw_graph(graph)