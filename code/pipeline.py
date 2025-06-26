import json
import pdb
import sys
import os

from query_open_alex import *
from graph import *
from simplicial_complex import *


def pipeline(query_url, name, data_location):

    input_file = f'{data_location}/{name}.json'
    if os.path.isfile(input_file) == False or os.stat(input_file).st_size == 0:
        results = query(f'https://api.openalex.org/works?filter={query_url}')
        paper_dict = format_papers(results)
        paper_dict = {key: set(paper_dict[key]) for key in paper_dict.keys() if len(paper_dict[key]) > 1}
        with open(input_file, 'w') as f:
            json.dump(paper_dict, f)


    with open(input_file, 'r') as f:
        dataset = json.load(f)
    print(f'dataset size: {len(dataset.keys())}')

    complex = SimplicialComplex(dataset, data_location = data_location, name = name, verbose = True, save=True)
    complex.calculate_all(save = True, verbose = True)
    complex.build_perseus_simplex()
    print(complex.betti_numbers)

    # graph = build_graph([dataset])
    # draw_graph(graph)


if __name__ == "__main__":

    # small sloths
    pipeline('title.search:Choloepus', 'small_sloths', '../data/sloths')

    # big sloths
    pipeline('default.search:two%252520toed%252520sloth%7CCholoepus', 'big_sloths', '../data/sloths')

    # sloths
    pipeline('title.search:two%252520toed%252520sloth%7CCholoepus,abstract.search:two%252520toed%252520sloth%7CCholoepus', 'sloths', '../data/sloths')

