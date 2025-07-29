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

    arguments:
    - folder_location (path): location to the folder where data is saved and pulled from
    - name (string): name of base name for save and data files
    - overwrite (bool): whether to overwrite the current files
    - verbose (bool): whether to print updates on progress
    - results (bool): whwether to print results
    - save (bool): whether to save results and data to files
    """

    def __init__(self, folder_location = "../data/", name = "temp", overwrite = False, verbose = False, results = False, save = False):
        self.data_location = folder_location
        self.name = name
        self.overwrite = overwrite
        self.dataset_location = f'{self.data_location}/{self.name}.json'
        self.verbose = verbose
        self.results = results
        self.save = save


    def load_data(self, dataset_location = None):
        """
        arguments:
        - dataset_location (path): location of the dataset, if none uses the class dataset by default

        returns:
        - None
        """
        if dataset_location == None:
            dataset_location = self.dataset_location
        with open(dataset_location, 'r') as f:
            dataset = json.load(f)
            self.dataset = {key: frozenset(dataset[key]) for key in dataset.keys()}
            if self.verbose or self.results:
                print(f'dataset size: {len(dataset.keys())}')

    def run_betti_analysis(self, folder_location = None, name = None, dataset = None, max_bar_level = 1):
        """
        runs the betti analysis on the given data

        arguments:
        - folder_location (path): the place where the folder for the save files is
        - name (string): base name for save files
        - dataset (file path): location of the dataset to run on, by default will look in class folder_location
        - max_bar_level (int): the bar to raise up to for the filtering process

        returns:
        - None
        """
        if folder_location == None:
            folder_location = self.data_location
        if name == None:
            name = self.name

        if dataset == None:
            dataset = self.dataset
        f = open(f'{folder_location}{name}_betti_results.txt', 'w+')

        for level, dataset in enumerate(self.raise_bar(max_bar_level)):
            if self.verbose:
                print(f'Now beginning betti analysis of level {level}')         
            complex = SimplicialComplex(top_cell_complex = dataset, data_location = self.data_location, name = f'{self.name}', level=level, verbose = self.verbose, results = self.results, save=self.save)
            complex.run_betti()
            f.write(f'level: {level}, betii_numbers: {complex.betti_numbers}')
        
        f.close()

    def run_distance_analysis(self, colab1, colab2, folder_location = None, name = None, dataset = None, max_width = 0, max_bar_level = 1):
        """
        runs a distance analysis on two colaborations. Has two dials for stability width and bar level 


        arguments:
        - colab1 (set): the set of user_labels to check distance with
        - colab2 (set): the set of user_labels to check distance with
        - folder_location (path): the place where the folder for the save files is
        - name (string): base name for save files
        - dataset (file path): location of the dataset to run on, by default will look in class folder_location
        - max_width (int): the maximum width to check for paths
        - max_bar_level (int): the bar to raise up to for the filtering process

        returns:
        - None
        """

        if dataset == None:
            dataset = self.dataset

        if folder_location == None:
            folder_location = self.data_location
        if name == None:
            name = self.name

        f = open(f'{folder_location}{name}_distance_results.txt', 'w+')

        for level, dataset in enumerate(self.raise_bar(max_bar_level)):
            if self.verbose:
                print(f'Now beginning distance analysis of level {level}') 
            width = 0
            distance = 0
            complex = SimplicialComplex(top_cell_complex = dataset, data_location = self.data_location, name = f'{self.name}', verbose = self.verbose, results = self.results, save=self.save)
            while width <= max_width and distance != -1:
                distance = complex.run_colab_distance(colab1, colab2, width = width)
                f.write(f"level: {level}, width: {width}, distance: {distance}")
                width += 1

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

    erdos_id =  "A5035271865"
    deanna_id = 'A5039705998'
    kate_id = 'A5029009134'

    # Pipeline = Pipeline(name = 'deanna_erdos',data_location= '../data/people/', verbose=False, results=True, overwrite=True, save = True)
    # Pipeline.load_data(dataset_location='../data/math/math_theory.json')
    # distance = Pipeline.run_distance_analysis(colab1 = {deanna_id}, colab2 = {erdos_id}, width = 0)

    # print(f'distance between deanna and erdos is {distance}')


    # Kate Papers
    # papers_by_author(verbose = True)

    pipeline = Pipeline(name = 'Kate_Meyer', folder_location= '../data/people/', verbose=True, results=True, overwrite=True, save = True)
    pipeline.load_data()
    pipeline.run_betti_analysis(max_bar_level=2)
    for level in range(2):
        graph.main(data_location=f'../data/people/', save_location=f'../data/people/', name= f'Kate_Meyer{level}', special_nodes={'A5029009134'})

    

    # Jeremy Papers
    # pipeline = Pipeline(name = 'Jeremy_Reiter', data_location= '../data/people', verbose=True, results=True, overwrite=False, save = True)
    # papers_by_author(seedID='A5068565988', name="Jeremy_Reiter")
    # pipeline.main(max_bar_level=1)
    # for level in range(1):
    #     graph.main(f'../data/people/Jeremy_Reiter{level}_top_cell_complex.pkl',f'../data/people/Jeremy_Reiter{level}_graph.png')

    # small sloths
    # pipeline = Pipeline(name = 'small_sloths', folder_location= '../data/sloths/', verbose=True, overwrite=True, save = True)
    # papers_by_topic(query_filter='title.search:Choloepus')
    # pipeline.load_data()
    # pipeline.run_betti_analysis(max_bar_level=10)
    # pipeline.run_distance_analysis({'A5103447570'}, {'A5070985763'}, max_bar_level=5, max_width=5)

    # # big sloths
    # pipeline = Pipeline(name = 'big_sloths', data_location='../data/sloths', overwrite= False, verbose=False, results = True)
    # pipeline.main(query_url= 'default.search:two%252520toed%252520sloth%7CCholoepus', max_bar_level= 10)

    # # sloths      
    # pipeline('title.search:two%252520toed%252520sloth%7CCholoepus,abstract.search:two%252520toed%252520sloth%7CCholoepus', 'sloths', '../data/sloths', overwrite=True)

    # math theory
    # pipeline = Pipeline(name = 'math_theory',data_location= '../data/math', verbose = False, results=True)
    # pipeline.papers_by_topic(query_url='primary_topic.field.id:fields/26,primary_topic.id:t12170', max_bar_level=10)

    # all of math
    # pipeline = Pipeline('primary_topic.field.id:fields/26', 'math', '../data/math')
    # pipeline.main()

    # # Topological and Geometric Data Analysis 
    # # from 2020 - 2025
    # pipeline = Pipeline(name = 'computational_topology', data_location='../data/math')
    # pipeline.main('publication_year:2000+-+2025,primary_topic.id:t12536', max_bar_level= 2)
