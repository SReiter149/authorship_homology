from itertools import combinations
import pdb
import numpy as np 
from linear_algebra import *
import subprocess
import logging
import os
import random

class SimplicialComplex:
    '''
    simplicial complex class! 
    has all methods for 
    - constructing simplicial complexes 
    - calculating betti numbers

    
    - data_location (relative location of data)
    - name (string)
    - verbose (bool)
    - save (bool)
    - simplex (a set of vertices)
    - top_cell_complex (a set of simplicies)
    - maximum dimension (integer)
    '''

    def __init__(self, top_cell_complex, data_location = '../data', name = 'simplicial_complex', verbose = False, save = False):
        self.data_location = data_location
        self.name = name
        self.verbose = verbose
        self.save = save

        # parses top_cell_complex
        if isinstance(top_cell_complex, list):
            if isinstance(top_cell_complex[0], list):
                self.top_cell_complex = set()
                for simplex in top_cell_complex:
                    self.top_cell_complex.add(frozenset(simplex))
            else:
                raise ValueError('invalid input type')
            

        elif isinstance(top_cell_complex, dict):
            self.top_cell_complex = set()
            for key in top_cell_complex.keys():
                if isinstance(top_cell_complex[key], list):
                    self.top_cell_complex.add(set(top_cell_complex[key]))
                elif isinstance(top_cell_complex[key], frozenset):
                    self.top_cell_complex.add(top_cell_complex[key])
                else:
                    raise ValueError('invalid input type')
    

        elif isinstance(top_cell_complex, frozenset):
            if isinstance(next(iter(top_cell_complex)), frozenset):
                self.top_cell_complex = top_cell_complex
            else:
                raise ValueError('invalid input type')
        
        else:
            raise ValueError('invalid input type')

        # converts the elements to little integers
        vertex_dict = dict()
        clean_top_cell_complex = dict()
        for simplex_id, simplex in enumerate(self.top_cell_complex):
            for vertex in simplex:
                if vertex not in vertex_dict.keys():
                    vertex_dict[vertex] = len(vertex_dict.keys())
            clean_top_cell_complex[simplex_id] = frozenset({vertex_dict[vertex] for vertex in simplex})
        self.top_cell_complex = clean_top_cell_complex

        self.max_dimension = max([len(simplex) for simplex in self.top_cell_complex.values()])
        self.vertex_count = len(vertex_dict.keys())
        self.max_vertex_id = self.vertex_count
        self.max_simplex_id = max(self.top_cell_complex.keys())

    def __repr__(self):
        '''
        returns:
            output (str): a string containing
                - name
                - max dimension
                - top cell complex
                - size of the k-skeletons (if calculated)
                - dimension matrices (if calculated)
                - betti numbers (if calculated)
        '''
        output = f'\n--------{self.name}-----------\n '

        # name

        # max dimension
        output += f'max dimension: {self.max_dimension}\n'

        # top cell complex
        # output += f'top cell complex: {self.top_cell_complex}\n'

        try:
            for idx, k_skeleton in self.simplicial_complex.items():
                output += f'in the {idx}-skeleton, there are {len(k_skeleton)} elements \n'
        except:
            pass

        # dimension matrices
        try:
            if self.dimension_matrices:
                for dimension_matrix in self.dimension_matrices:
                    output += str(dimension_matrix.shape)
                ouput += '\n'

        except:
            pass
        
        # betti numbers
        try:
            if self.betti_numbers:
                for i in range(len(self.betti_numbers)):
                    output += f'betti number {i}: {self.betti_numbers[i]}\n'
        except:
            pass

        return output[:-1]
    
    def save_top_complex(self): 
        with open(f'{self.data_location}/{self.name}_top_cell_complex.txt', 'w') as f:
            for complex in self.top_cell_complex.values(): 
                f.write(str(complex))
                f.write(f'\n')
            f.close()

    def build_vertex_simplex_maps(self):
        """
        set up the data structure used through the rest of the class. Two dictionaries, that have information about which simplex a vertex is part of and which vertices a simplex has. 

        arguments:
        - None

        returns:
        -None
        """
        simplex_maps = dict()
        vertex_maps = {i:set() for i in range(self.vertex_count)}
        for simplex_id, simplex in self.top_cell_complex.items():
            simplex_maps[simplex_id] = set(simplex)
            for vertex in simplex:
                vertex_maps[vertex].add(simplex_id)

        self.simplex_maps = simplex_maps
        self.vertex_maps = vertex_maps

    def perform_strong_collapses(self):
        """
        Uses strong face reduction to reduce a simplex in a faster way. Will be passed off to edge collapse after. 

        arguments:
        - none

        returns:
        - none
        """

        # setting up queues
        vertex_queue = [vertex_id for vertex_id in self.vertex_maps.keys()]
        simplex_queue = []
        while vertex_queue: 
            while vertex_queue:
                # retrieving relivant data
                vertex_id = vertex_queue.pop(0)
                vertex = self.vertex_maps[vertex_id]
                nonzero_simplex_id = self.simplex_maps[next(iter(vertex))]

                for simplex2_id in nonzero_simplex_id:

                    # checking if vertex is dominated by vertex2
                    if vertex < self.vertex_maps[simplex2_id]:
                        # removing vertex from relivant places
                        self._remove_vertex(vertex_id)
                        
                        # pushing all non-zero columns
                        simplex_queue.extend(vertex)
                        break

            # deleting repeated elements in column queue 
            simplex_queue = list(set(simplex_queue))

            while simplex_queue:
                # retrieving relivant data
                simplex_id = simplex_queue.pop(0)
                simplex = self.simplex_maps[simplex_id]
                nonzero_vertex_id = self.vertex_maps[next(iter(simplex))]

                for vertex_2_id in nonzero_vertex_id:

                    # checking if simplex is dominated by simplex2
                    if simplex < self.simplex_maps[vertex_2_id]:

                        # removing simplex from relivant places
                        self._remove_simplex(simplex_id)
                        vertex_queue.extend(simplex)
                        break

            vertex_queue = list(set(vertex_queue))

    def star(self, face, complex = None):
        '''
        finds all simplices in the complex for which the given face is a face in the complex

        arguments: 
        - face (set of vertex ids): the face to check the star of 
        - complex (list of simplex ids) (optional): if the search space is smaller than the whole complex

        returns:
        - star_set (a set of simplex ids): implementation of the star operation from Fellegara 2020 Homology 
        '''

        star_set = set()

        # Case 1: if the search space is the whole complex
        if complex == None:
            for simplex_id in self.vertex_maps[next(iter(face))]:
                if set(face).issubset(self.simplex_maps[simplex_id]):
                    star_set.add(simplex_id)
        
        # Case 2: if the search space is the given complex
        else:
            print(f'the face is {face}, the complex I am looking in is {complex}')
            print(f'the state of simplex maps is {self.simplex_maps}')
            for simplex_id in complex:
                simplex = self.simplex_maps[simplex_id]
                if set(face).issubset(simplex):
                    star_set.add(simplex_id)
        return star_set
    
    def boundary_face(self, face, complex):
        """
        checks whether the given face is a face in the given complex

        arguments:
        - face (list)
        - complex (list of simplex ids): the simplicial complex to check

        returns:
        - bool: whether the face is a face in the complex or not
        """
        for simplex_id in complex:
            simplex = self.simplex_maps[simplex_id]
            if set(face) < set(simplex):
                return True
        return False
    
    def link_condition(self, star_v1, star_v2, star_edge):
        """
        Fellegara 2020 Homology

        checks the link condition, basically whether it is safe to remove the given edge or not

        arguments:
        - star_v1 (set of simplex ids): the set of simplices that v1 is in
        - star_v2 (set of simplex ids): the set of simplices that v2 is in
        - star_edge (set of simplex ids): the set of simplices that edge is in

        returns:
        - bool: whether the link condition is satisfied or not
        """

        T1 = star_v1 - star_edge
        T2 = star_v2 - star_edge

        for id1 in T1:
            simplex1 = self.simplex_maps[id1]
            for id2 in T2:
                simplex2 = self.simplex_maps[id2]
                shared_face = simplex1.intersection(simplex2)
                if bool(shared_face):
                    if not self.boundary_face(shared_face, star_edge):
                        return False
        return True

    def contract(self, v1,v2,star_v1, star_v2, star_edge):
        """
        Fellegara 2020 Homology

        Removes edge (v1,v2) from the simplicial complex
        turns all instances of v1 into v2

        arguments:
        - v1 (1-set): the first vertex id, the one to be removed
        - v2 (1-set): the second vertex id, the one to be combined into
        - star_v1 (set of simplex ids): the star of v1
        - star_v2 (set of simplex ids): the star of v2
        - star_edge (set of simplex ids): the star of the edge (v1,v2)

        returns:
        - None  
        """   
        for top_simplex_id in star_edge:
            top_simplex = self.simplex_maps[top_simplex_id]
            gamma_1 = top_simplex - v1
            gamma_2 = top_simplex - v2

            star_gamma_1 = self.star(gamma_1, complex = star_v2)
            star_gamma_2 = self.star(gamma_2, complex = star_v1)

            if star_gamma_1.union(star_gamma_2) == top_simplex:
                self._add_simplex(gamma_1)
            self._remove_simplex(top_simplex_id)
            star_v2.remove(top_simplex_id)
            star_v1.remove(top_simplex_id)

        for top_simplex_id in star_v1 - star_edge:
            # pdb.set_trace()
            self.simplex_maps[top_simplex_id] = (self.simplex_maps[top_simplex_id] - v1).union(v2)
        
        self.vertex_maps[next(iter(v2))].union(self.vertex_maps[next(iter(v1))])
        self.vertex_maps.pop(next(iter(v1)))

    def perform_edge_contraction(self):
        """
        reduces the top cell complex using the above functions

        very unoptimal

        arguments:
        - none

        returns:
        - none
        """

        current_simplex_id = 0
        while current_simplex_id <= self.max_simplex_id:
            if current_simplex_id in self.simplex_maps.keys():
                vertex1_options = self.simplex_maps[current_simplex_id]
                while vertex1_options:
                    vertex1_id = vertex1_options.pop()
                    vertex2_options = vertex1_options.copy()
                    while vertex2_options:
                        vertex2_id = vertex2_options.pop()
                        star_edge = self.star({vertex1_id, vertex2_id})
                        if bool(star_edge):
                            star_v1 = self.star({vertex1_id})
                            star_v2 = self.star({vertex2_id})
                            if self.link_condition(star_v1, star_v2, star_edge):
                                print(f"contracting edge {vertex1_id, vertex2_id}")
                                self.contract({vertex1_id},{vertex2_id},star_v1, star_v2, star_edge)
                                vertex1_options.remove(vertex2_id)
            current_simplex_id += 1

    def _add_simplex(self, simplex):
        """
        helper function to add a simplex to the data structure. Assumes that all vertices already are in the simplex. Will give the next available id to the simplex. 

        arguments:
        - simplex (set of vertex ids): simplex to add

        returns:
        - none
        """
        new_simplex_id = self.max_simplex_id + 1

        self.simplex_maps[new_simplex_id] = simplex
        
        for vertex_id in simplex:
            self.vertex_maps[vertex_id].add(new_simplex_id)

        self.max_simplex_id += 1

    def _remove_simplex(self, simplex_id):
        """
        helper_function to remove a simplex from the data structure.

        arguments:
        - simplex_id (int): the id of the simplex to remove

        returns:
        - none
        """
        print(f"now removing {simplex_id}")
        simplex = self.simplex_maps.pop(simplex_id)
        for vertex_id in simplex:
            self.vertex_maps[vertex_id].remove(simplex_id)
        
    def _remove_vertex(self, vertex_id):
        """
        helper_function to remove a vertex from the data structure.

        arguments:
        - vertex_id (int): the id of the vertex to remove

        returns:
        - none
        """

        vertex = self.vertex_maps.pop(vertex_id)
        for simplex_id in vertex:
            self.simplex_maps[simplex_id].remove(vertex_id)


    def calculate_all(self, save = False, verbose = None):
        '''
        parameters:
            self
            verbose (bool): if true, print statements to see what is happening
        returns:
            None
        
        runs all functions on the simplicial complex for 
        - building it
        - elementary collapsing it

        specifically this function will build the simplicialt complex, then build the dimension matrices, finally calculate the betti numbers. 
        '''

        if verbose:
            self.verbose = verbose
        
        if save:
            self.save = save

        self.build_vertex_simplex_maps()
        # self.perform_strong_collapses()
        self.perform_edge_contraction()

if __name__ == '__main__':
    def test(top_cell_complex, answer, name, *args, **kwargs):
        data_location='../data/simplex_tests'
        complex = SimplicialComplex(top_cell_complex, data_location = data_location, name = name, *args, **kwargs)
        complex.calculate_all()

        # assert complex.betti_numbers == answer
        
        # complex.calculate_euler_characteristic()
        # complex.calculate_betti_sum()

        # assert complex.betti_sum == complex.euler_characteristic

        # compare with perseus
        # complex.build_perseus_simplex()
        # subprocess.run(["arch", "-x86_64", "./perseus",
        #     "nmfsimtop",
        #     f"{data_location}/{name}_perseus.txt",
        #     f"{data_location}/{name}_perseus"
        #     ],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     check= True
        #     )
        
        # with open(f"{data_location}/{name}_perseus_betti.txt", 'r') as f:
        #     for line in f:
        #         perseus_betti = line
        #     perseus_betti = perseus_betti.split(" ")[2:-1]
        #     perseus_betti = [int(x) for x in perseus_betti]
        
        # while len(perseus_betti) < len(answer):
        #     perseus_betti.append(0)
        # print(f'perseus betti: {perseus_betti}')
        # assert perseus_betti == complex.betti_numbers



    # my example
    # answer should be [1,3,0,0] 
    answer = [1,3,0,0,0]
    top_cell_complex = [[1,2,3,4],[4,5,6,7],[2,5,7],[1,5],[7,8],[8,9],[9,10],[8,9,10],[7,8,9,10],[1,3,5,7],[2,4,6,8],[10,11,12,13,14]] 
    test(top_cell_complex, answer, name = 'test1', verbose = True , save = True)

    # Chad example
    # answer should be [1,1,0]
    # answer = [1,1,0]
    # top_cell_complex = [[1,2,5],[2,3],[3,4],[4,5]] 
    # test(top_cell_complex, answer, name = 'test2' , verbose = True, save = True)

    # Chad exercise 7
    # answer should be [1,2,0]
    # answer = [1,2,0]
    # top_cell_complex = [[1,2],[2,3,7],[3,4],[4,5],[5,6],[6,3],[7,8],[8,1]] 
    # test(top_cell_complex, answer, name = 'test3', verbose = True, save = True)

    # three triangles that have a 2 dimensional hole in the middle
    # answer should be [1,1,0]
    # answer = [1,1,0]
    # top_cell_complex = [[1,2,3],[2,4,5],[3,5,6]] 
    # test(top_cell_complex, answer, name = 'test4', save = True)

    # three open triangles with a triangle in the middle
    # answer should be [1,4]
    # answer = [1,4]
    # top_cell_complex = [[1,2],[1,3],[2,3],[2,4],[2,5],[4,5],[3,5],[3,6],[5,6]] 
    # test(top_cell_complex, answer, name = 'test5', verbose = False, save = True)

    # the small sloths test
    # answer = [81,0,0,0,0,0]
    # top_cell_complex = [[1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16], [17, 18], [19, 20], [21, 22, 23, 24], [25, 26, 27, 17, 18], [21, 22, 23, 24], [14, 16], [28, 29, 30], [31, 32], [33, 18, 34, 35], [36, 37], [38, 39], [21, 23, 40, 41, 24], [18, 42, 43], [44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58], [59, 60, 61], [62, 63, 64, 65], [66, 67, 68, 69], [70, 71], [72, 73], [74, 75, 23], [76, 77, 78, 79], [80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91], [92, 93, 94], [95, 96, 97], [98, 99, 100, 101], [102, 103, 104, 105, 106], [39, 38], [107, 18], [108, 109], [110, 111, 112], [113, 114, 115, 116, 117], [118, 119, 120], [121, 122], [123, 124, 80], [125, 126, 127, 128], [129, 130], [131, 132, 133, 134], [135, 136, 137, 138], [139, 140, 141, 142, 143, 144], [145, 146, 147, 148, 149, 150], [151, 152, 153], [154, 155, 156], [157, 158, 159, 160, 161, 162], [163, 164], [165, 166], [167, 168, 169, 170, 171], [81, 172, 173, 174, 80], [175, 176, 177, 178], [179, 18, 180], [181, 182, 183, 18], [184, 185, 186, 187], [188, 189, 190, 191, 192], [193, 194, 195, 196], [197, 17], [198, 18, 199], [200, 201], [202, 185, 186, 187], [203, 204, 205, 132], [206, 207, 208, 95], [209, 210], [211, 186, 185, 187], [212, 213, 214, 215, 216, 217], [218, 18], [219, 220, 221], [222, 223, 224, 225], [226, 227, 228, 229], [230, 231], [232, 233, 234], [235, 236], [237, 238, 239], [240, 241], [200, 201], [242, 243], [244, 245], [246, 247, 248], [249, 250, 251, 252], [106, 102, 253, 254], [255, 44, 46, 256, 257, 49], [258, 259], [18, 260], [261, 18], [107, 18], [262, 263], [264, 265, 266], [267, 268, 269, 270], [72, 73], [264, 265, 266], [271, 272, 273, 274], [275, 276, 277], [264, 278, 279], [280, 281], [282, 283, 284, 285, 286, 287], [288, 289, 290, 291], [292, 293], [294, 295, 296], [34, 297], [298, 299, 300], [301, 302, 250, 303, 252, 251], [304, 305, 306, 307], [308, 309], [310, 311, 312], [313, 314], [315, 316], [33, 297, 34, 18], [297, 33, 18, 34]]
    # test(top_cell_complex, answer, name = 'small_sloth', verbose = True, save = True)
            


            
            

    




    