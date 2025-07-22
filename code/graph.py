import networkx as nx
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import combinations 
import pickle as pkl
import pdb

# the graph and things
def build_graph(top_cell_complex, vertex_dict, special_nodes = set()):
    '''
    arguments:
    - top cell complex (dictionary of sets): the top cell complex containing the vertex ids

    returns:
    - graph (nx.Graph object): the graph
    '''
    graph = nx.Graph()
    node_color = 'blue'
    speical_color = 'red'
    edge_color = 'lightblue'
    node_size = 1
    special_size = 3
    edge_size = 0.2
    node_weight = 1
    edge_weight = 0.5

    for simplex in top_cell_complex.values():
        for vertex_id in simplex:
            # pdb.set_trace()
            if bool(vertex_dict[vertex_id].intersection(special_nodes)):
                graph.add_node(vertex_id, color = speical_color, size = special_size, node_weight = node_weight)
            else:
                graph.add_node(vertex_id, color = node_color, size = node_size, node_weight = node_weight)
        for edge in combinations(simplex, 2):
            graph.add_edge(edge[0], edge[1], weight=edge_weight, edge_size = edge_size, color=edge_color)
    return graph

def draw_graph(graph, top_cell_complex, save_location):
    graph.add_node(-1, color = 'grey', size = 0, node_weight = 0.5)
    pos = nx.spring_layout(graph, weight='weight', k = 10/len(top_cell_complex.values()), iterations=20) 

    _, ax = plt.subplots()
    for simplex in top_cell_complex.values():
        pts = [pos[v] for v in simplex]
        poly = Polygon(pts, closed=True, facecolor='grey', alpha=0.5, edgecolor=None)
        ax.add_patch(poly)
    edge_colors = [graph.edges[n].get('color', 'gray') for n in graph.edges]
    node_colors = [graph.nodes[n].get('color', 'gray') for n in graph.nodes]
    sizes = [graph.nodes[n].get("size", 25) for n in graph.nodes]

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=sizes, ax=ax)
    ax.set_axis_off()
    plt.savefig(save_location, dpi = 1000)

def main(data_location, save_location, name, special_nodes = set()):
    top_cell_location = f'{data_location}{name}_simplex_maps.pkl'
    save_location = f'{save_location}{name}_graph.png'
    vertex_dict_location = f'{data_location}{name}_vertex_dict.pkl'

    with open(vertex_dict_location, 'rb') as f:
        vertex_dict = pkl.load(f)

    with open(top_cell_location, 'rb') as f:
        top_cell_complex = pkl.load(f)

    graph = build_graph(top_cell_complex, vertex_dict, special_nodes=special_nodes)
    draw_graph(graph, top_cell_complex, save_location)

if __name__ == '__main__':
    # testing on the 1st test
    main(data_location=f'../data/simplex_tests/', save_location='../data/graph_tests/' ,name=f'test0')

    # testing on small_sloths0
    # main(data_location=f'../data/sloths/',save_location='../data/graph_tests/', name= f'small_sloths0')

    # testing on Kate_Meyer0
    main(data_location=f'../data/people/',save_location='../data/graph_tests/', name= f'Kate_Meyer0', special_nodes={'A5029009134'})