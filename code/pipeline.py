import json
import pdb
import sys
import os

from query_open_alex import *
import graph
from simplicial_complex import *



class Pipeline:
    """
    pipeline class to run the whole simplicial pipeline from pulling data from openAlex to building and running the simplicial complex

    warning:
    - if you plan on running both a distance and betti analysis, do the distance first or reload the dataset. Running the betti analysis reduces the complex

    arguments:
    - directory_path (path): path to the folder where data is saved and pulled from
    - name (string): name of base name for save and data files
    - overwrite (bool): whether to overwrite the current files
    - verbose (bool): whether to print updates on progress
    - results (bool): whwether to print results
    - save (bool): whether to save results and data to files
    """

    def __init__(self, directory_path = "../data/", name = "temp", overwrite = False, verbose = False, results = False, save = True):
        assert directory_path[-1] == '/'
        os.makedirs(directory_path, exist_ok=True) 

        self.directory_path = directory_path
        self.name = name
        self.overwrite = overwrite
        self.dataset_path = f'{self.directory_path}{self.name}.json'
        self.verbose = verbose
        self.results = results
        self.save = save


    def load_data(self, top_cell_complex = None, dataset_path = None):
        """
        by default looks at the dataset_path
        arguments:
        - top_cell_complex (dictionary of sets): if you desire to load the dataset by giving it the top_cell_complex
        - dataset_path (path): path of the dataset, if none uses the class dataset by default

        returns:
        - None
        """
        if top_cell_complex != None:
            self.dataset = self.load_user_data(top_cell_complex)
        else:
            if dataset_path == None:
                dataset_path = self.dataset_path
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                self.dataset = {key: frozenset(dataset[key]) for key in dataset.keys()}
                if self.verbose or self.results:
                    print(f'dataset size: {len(dataset.keys())}')
    
    def load_user_data(self, top_cell_complex):
        """
        handles converting user top_cell_complex into correct data structure
        
        arguments: 
        - top_cell_complex (?): top cell complex, allowed types:
            - dictioanry of frozensets
            - list of lists
            - list of frozensets
        
        returns:
        - top_cell_complex (dictionary of frozensets): properly formatted dictionary
        """
        if isinstance(top_cell_complex, dict):
            if isinstance(next(iter(top_cell_complex.values())), frozenset):
                return top_cell_complex
        elif isinstance(top_cell_complex, list):
            if isinstance(top_cell_complex[0], list):
                return {i: frozenset(top_cell_complex[i]) for i in range(len(top_cell_complex))}
            if isinstance(top_cell_complex[0], set):
                return {i: frozenset(top_cell_complex[i]) for i in range(len(top_cell_complex))}
        raise NotImplemented("this user datatype has not been implemented")
            
            
            

    def run_betti_analysis(self, dataset = None, directory_path = None, name = None, max_bar_level = 1):
        """
        runs the betti analysis on the given data

        warning:
        - if you give this function a dataset, it will make this dataset the default for the Pipeline class moving forward

        arguments:
        - dataset (set of sets): the top cell complex
        - directory_path (path): the place where the folder for the save files is
        - name (string): base name for save files
        - max_bar_level (int): the bar to raise up to for the filtering process

        returns:
        - None
        """
        if directory_path == None:
            directory_path = self.directory_path
        if name == None:
            name = self.name

        if dataset == None:
            dataset = self.dataset
        f = open(f'{directory_path}{name}_betti_results.txt', 'w+')

        for level, dataset in enumerate(self.raise_bar(max_bar_level)):
            if self.verbose:
                print(f'Now beginning betti analysis of level {level}')         
            complex = SimplicialComplex(top_cell_complex = dataset, directory_path = self.directory_path, name = f'{self.name}', level=level, verbose = self.verbose, results = self.results, save=self.save)
            complex.run_betti()
            f.write(f'level: {level}, betii_numbers: {complex.betti_numbers}')
        
        f.close()

    def run_distance_analysis(self, colab1, colab2, directory_path = None, name = None, dataset = None, max_width = 0, max_bar_level = 1):
        """
        runs a distance analysis on two colaborations. Has two dials for stability width and bar level 


        arguments:
        - colab1 (set): the set of user_labels to check distance with
        - colab2 (set): the set of user_labels to check distance with
        - directory_path (path): the place where the folder for the save files is
        - name (string): base name for save files
        - dataset (file path): path of the dataset to run on, by default will look in class directory_path
        - max_width (int): the maximum width to check for paths
        - max_bar_level (int): the bar to raise up to for the filtering process

        returns:
        - None
        """

        if dataset == None:
            dataset = self.dataset

        if directory_path == None:
            directory_path = self.directory_path
        if name == None:
            name = self.name

        f = open(f'{directory_path}{name}_distance_results.txt', 'w+')

        for level, dataset in enumerate(self.raise_bar(max_bar_level)):
            if self.verbose:
                print(f'Now beginning distance analysis of level {level}') 
            width = 0
            distance = 0
            complex = SimplicialComplex(top_cell_complex = dataset, directory_path = self.directory_path, name = f'{self.name}', verbose = self.verbose, results = self.results, save=self.save)
            while width <= max_width and distance != -1:
                distance = complex.run_colab_distance(colab1, colab2, width = width)
                f.write(f"level: {level}, width: {width}, distance: {distance}")
                width += 1
            return distance

    def raise_bar(self, max_bar_level):
        """
        raises the bar and yields the datasets

        arguments:
        - max_bar_level (int): the maximum bar level to yield

        yields:
        - dataset (frozenset of frozensets): the top cell complex at the given level
        """
        # a list of dictionaries,
        # (set of colaborators : set of papers)
        levels = [dict() for i in range(max_bar_level)]
        levels[0] = {colabs : {paper} for paper, colabs in self.dataset.items()}
        
        assert max_bar_level >= 0
        if max_bar_level > 0:
            for level in range(1, max_bar_level):
                if self.verbose or self.results:
                    print(f'now building level {level} of {max_bar_level}, there are {len(levels[level -1])} elements in the last level')

                for colabs1, papers1 in levels[level - 1].items():
                    for colabs2, papers2 in levels[0].items():
                        if bool(colabs1.intersection(colabs2)):
                            if len(papers1.union(papers2)) > level:
                                if colabs1.intersection(colabs2) in levels[level].keys():
                                    levels[level][colabs1.intersection(colabs2)] = levels[level][colabs1.intersection(colabs2)].union(papers1.union(papers2))
                                else:
                                    levels[level][colabs1.intersection(colabs2)] = papers1.union(papers2)

        for level in levels:
            dataset = frozenset( frozenset(colabs) for colabs in level.keys() )
            if bool(dataset):
                yield dataset
            else:
                print(f'there is no data at level {level}')


if __name__ == "__main__":

    try:
        tests = [0, 1, 2 ,3 ,4 ]


        
        if 0 in tests:
            # small sloths
            pipeline = Pipeline(name = 'small_sloths', directory_path= '../data/sloths/', verbose=True, overwrite=True, save = True)
            papers_by_topic(query_filter='title.search:Choloepus')
            pipeline.load_data()
            pipeline.run_distance_analysis({'A5103447570'}, {'A5070985763'}, max_bar_level=5, max_width=5)
            pipeline.run_betti_analysis(max_bar_level=10)


        if 1 in tests:
            # big sloths
            name = 'big_sloths'
            directory_path = "../data/sloths/"
            query_filter = 'default.search:two%252520toed%252520sloth%7CCholoepus'

            pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=False, results = True)
            papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = name)
            pipeline.load_data()
            pipeline.run_betti_analysis(max_bar_level= 10)


        if 2 in tests:
            # sloths
            name = 'sloths'
            directory_path = "../data/sloths/"
            query_filter = 'title.search:two%252520toed%252520sloth%7CCholoepus,abstract.search:two%252520toed%252520sloth%7CCholoepus'

            pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=False, results = True)
            papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = name)
            pipeline.load_data()
            pipeline.run_betti_analysis(max_bar_level= 10)


        if 3 in tests:
            # Kate Papers
            name = 'Kate_Meyer'
            author_id = 'A5029009134'
            directory_path = '../data/people'
            papers_by_author(verbose = True)

            pipeline = Pipeline(name = name, directory_path= directory_path, verbose=True, results=True, overwrite=True, save = True)
            pipeline.load_data()
            pipeline.run_betti_analysis(max_bar_level=2)
            for level in range(2):
                graph.main(directory_path=directory_path, save_path=directory_path, name= f'{name}_level{level}', special_nodes={author_id})

        
        if 4 in tests:
            name = 'Jeremy_Reiter'
            author_id = 'A5068565988'
            directory_path = '../data/people'
            papers_by_author(verbose = True)

            pipeline = Pipeline(name = name, directory_path= directory_path, verbose=True, results=True, overwrite=True, save = True)
            pipeline.load_data()
            pipeline.run_betti_analysis(max_bar_level=2)
            for level in range(2):
                graph.main(directory_path=directory_path, save_path=directory_path, name= f'{name}_level{level}', special_nodes={author_id})

    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()    
