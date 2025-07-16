import json
import pdb
import sys
import os

from query_open_alex import *
from graph import *
from simplicial_complex_2 import *



class Pipeline:
    def __init__(self, data_location = "../data", name = "temp", overwrite = False, verbose = False, results = False):
        self.data_location = data_location
        self.name = name
        self.overwrite = overwrite
        self.file_path = f'{self.data_location}/{self.name}.json'
        self.verbose = verbose
        self.results = results

    def get_data(self, query_url):    
        query_url = f'https://api.openalex.org/works?filter={query_url}'
        f = open(self.file_path, 'w+')
        f.write('{')
        not_first = False

        for results in query(query_url): 
            paper_dict = format_papers(results)
            for key,value in paper_dict.items():
                if not_first:
                    f.write(', ')
                else:
                    not_first = True
                json.dump(key, f)  
                f.write(': ')
                json.dump(value, f)
        f.write('}')     
        f.close()

    def load_data(self):
        with open(self.file_path, 'r') as f:
            dataset = json.load(f)
            self.dataset = {key: frozenset(dataset[key]) for key in dataset.keys()}
            if self.verbose or self.results:
                print(f'dataset size: {len(dataset.keys())}')


    def run_analysis(self, dataset, level = None):
        complex = SimplicialComplex(top_cell_complex = dataset, data_location = self.data_location, name = f'{self.name}_level{level}', verbose = self.verbose, results = self.results, save=True)
        complex.calculate_all(save = True, verbose= self.verbose)


    def raise_bar(self, max_bar_level):
        # a list of dictionaries,
        # (set of colaborators : set of papers)
        levels = [dict() for i in range(max_bar_level)]
        levels[0] = {colabs : {paper} for paper, colabs in self.dataset.items()}
        
        for level in range(1, max_bar_level):
            if self.verbose or self.results:
                print(f'now building level {level} of {max_bar_level}, there are {len(levels[level -1])} elements in the last level')

            for colabs1, papers1 in levels[level - 1].items():
                for colabs2, papers2 in levels[0].items():
                    if bool(colabs1.intersection(colabs2)):
                        if len(papers1.union(papers2)) >= level:
                            if colabs1.intersection(colabs2) in levels[level].keys():
                                levels[level][colabs1.intersection(colabs2)] = levels[level][colabs1.intersection(colabs2)].union(papers1.union(papers2))
                            else:
                                levels[level][colabs1.intersection(colabs2)] = papers1.union(papers2)

        for level_num, level in enumerate(levels):
            if self.verbose:
                print(f'now analyzing level {level_num}')
            dataset = frozenset( frozenset(colabs) for colabs in level.keys() )
            yield dataset

    def main(self, query_url = None, max_bar_level = 1):
        # build the dataset
        if os.path.isfile(self.file_path) == False or os.stat(self.file_path).st_size == 0 or self.overwrite == True:
            if query_url != None:
                self.get_data(query_url)

        self.load_data()
        for level, dataset in enumerate(self.raise_bar(max_bar_level)):
            # if self.verbose:
            #     print(dataset)
            # pdb.set_trace()
            if dataset:
                self.run_analysis(dataset= dataset, level=level)



if __name__ == "__main__":

    # small sloths
    # pipeline = Pipeline(name = 'small_sloths', data_location= '../data/sloths', verbose=True, overwrite=False)
    # pipeline.main(query_url='title.search:Choloepus', max_bar_level= 15)

    # # big sloths
    # pipeline = Pipeline(name = 'big_sloths', data_location='../data/sloths', overwrite= False, verbose=False, results = True)
    # pipeline.main(query_url= 'default.search:two%252520toed%252520sloth%7CCholoepus', max_bar_level= 10)

    # # sloths      
    # pipeline('title.search:two%252520toed%252520sloth%7CCholoepus,abstract.search:two%252520toed%252520sloth%7CCholoepus', 'sloths', '../data/sloths', overwrite=True)

    # math theory
    pipeline = Pipeline(name = 'math_theory',data_location= '../data/math', verbose = False, results=True)
    pipeline.main(query_url='primary_topic.field.id:fields/26,primary_topic.id:t12170', max_bar_level=10)

    # all of math
    # pipeline = Pipeline('primary_topic.field.id:fields/26', 'math', '../data/math')
    # pipeline.main()

    # # Topological and Geometric Data Analysis 
    # # from 2020 - 2025
    # pipeline = Pipeline(name = 'computational_topology', data_location='../data/math')
    # pipeline.main('publication_year:2000+-+2025,primary_topic.id:t12536', max_bar_level= 2)
