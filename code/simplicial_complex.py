'''

'''

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
        clean_top_cell_complex = set()
        for complex in self.top_cell_complex:
            for vertex in complex:
                if vertex not in vertex_dict.keys():
                    vertex_dict[vertex] = len(vertex_dict.keys()) + 1
            clean_top_cell_complex.add(frozenset({vertex_dict[vertex] for vertex in complex}))
   
        self.top_cell_complex = clean_top_cell_complex
        self.max_dimension = max([len(simplex) for simplex in self.top_cell_complex])

    def __str__(self):
        output = f"Simplicial Complex of dimension {self.max_dimension}"
        return output

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

    def save_complete_complex(self):
        with open(f'{self.data_location}/{self.name}_complete_complex.txt', 'w') as f:
            for complex in self.simplicial_complex: 
                f.write(str(complex))
                f.write(f'\n')
            f.close()

    def save_top_complex(self): 
        with open(f'{self.data_location}/{self.name}_top_cell_complex.txt', 'w') as f:
            for complex in self.top_cell_complex: 
                f.write(str(complex))
                f.write(f'\n')
            f.close()

    def save_incidence_matrices(self):
        with open(f'{self.data_location}/{self.name}_incidence_matrices.txt', 'w') as f:
            f.write(f'transpose on import, rows are columns!\n')
            for incidence_matrix in self.incidence_matrices:
                for column_index in range(incidence_matrix.shape[1]):
                    column = incidence_matrix[:, column_index]
                    for x in column:
                        f.write(f'{x} ')
                    f.write(f'\n')
                f.write(f"\n")

    def save_betti_numbers(self):
        with open(f'{self.data_location}/{self.name}_betti_numbers.txt', 'w') as f:
            f.write(str(self.betti_numbers))
            f.close()
        
    def star(self, face, complex = None):
        '''
        finds all simplices in the complex for which the given face is a face in the complex

        arguments: 
        - face (set): the face to check the star of 
        - complex (set of simplicies) (optional): if the search space is smaller than the whole complex

        returns:
        - star_set (set of simplicies): implementation of the star operation from Fellegara 2020 Homology 
        '''
        star_set = set()

        # Case 1: if the search space is the whole complex
        if complex == None:
            for simplex in self.top_cell_complex:
                if face.issubset(simplex):
                    star_set.add(simplex)

        # Case 2: if the search space is the given complex
        else:
            for simplex in complex:
                if face.issubset(simplex):
                    star_set.add(simplex)
        return star_set

    def boundary_face(self, face, complex):
        """
        checks whether the given face is a face in the given complex

        arguemtns:
        - face (set)
        - complex (simplicial complex): the simplicial complex to check

        returns:
        - bool: whether the face is a face in the complex or not
        """
        for simplex in complex:
            # is there a proper subset function?
            if face.issubset(simplex):
                if len(face) < len(simplex):
                    return True
        return False

    def link_condition(self, star_v1, star_v2, star_edge):
        """
        Fellegara 2020 Homology

        checks the link condition, basically whether it is safe to remove the given edge or not

        arguments:
        - star_v1 (set of simplices): the set of simplices that v1 is in
        - star_v2 (set of simplices): the set of simplices that v2 is in
        - star_edge (set of simplices): the set of simplices that edge is in

        returns:
        - bool: whether the link condition is satisfied or not
        """
        T1 = star_v1 - star_edge
        T2 = star_v2 - star_edge
        for t1 in T1:
            for t2 in T2:
                shared_face = t1.intersection(t2)
                if bool(shared_face):
                    if not self.boundary_face(shared_face, star_edge):
                        return False
        return True
    
    def contract(self, v1, v2, star_v1, star_v2, star_edge):
        """
        Fellegara 2020 Homology

        Removes edge (v1,v2) from the simplicial complex
        turns all instances of v1 into v2

        arguments:
        - v1 (1-set): the first vertex, the one to be removed
        - v2 (1-set): the second vertex, the one to be combined into
        - star_v1 (set of simplices): the star of v1
        - star_v2 (set of simplices): the star of v2
        - star_edge (set of simplicies): the star of the edge (v1,v2)

        returns:
        - None
        
        """
        for top_simplex in star_edge:
            gamma_1 = top_simplex - v1
            gamma_2 = top_simplex - v2

            # set of simplices
            star_gamma_1 = self.star(gamma_1, star_v2)
            star_gamma_2 = self.star(gamma_2, star_v1)
            pdb.set_trace()
            if star_gamma_1.union(star_gamma_2) == {top_simplex}:
                self.top_cell_complex.add(gamma_1)
            self.top_cell_complex.remove(top_simplex)


        for top_simplex in star_v1 - star_edge:
            self.top_cell_complex.remove(top_simplex)
            self.top_cell_complex.add((top_simplex - v1).union(v2))

    def reduce_top_cell_complex(self):
        """
        reduces the top cell complex using the above functions

        very unoptimal

        arguments:
        - none

        returns:
        - none
        
        """

        # creates edge set
        edges = set()
        for simplex in self.top_cell_complex:
            for edge in combinations(simplex, 2):
                edges.add(frozenset(edge))

        for edge in edges:
            v1, v2 = [{v} for v in edge]
            star_edge = self.star(edge)
            # check if edge is still in the complex 
            # (if there is a way I don't need to generate all the edges so I don't need this check that is likely more optimal)
            if bool(star_edge):
                star_v1 = self.star(v1)
                star_v2 = self.star(v2)
                if self.link_condition(star_v1, star_v2, star_edge):
                    self.contract(v1,v2,star_v1, star_v2, star_edge)

    def build_simplicial_complex(self):
        '''
        builds the simplicial complex from the top cell complex
        takes each top cell complex and adds each sub combination to the simplicial complex
        stores each dimension in its own entry in the simplicial complex list
        each dimension is sorted lexigraphically 
        '''


        simplicial_complex = {i: set() for i in range(self.max_dimension)}

        # builds all subsimplices from the top cell complexes
        for simplex in self.top_cell_complex:
            for skeleton_dimension in range(len(simplex)):
                for subsimplex in combinations(simplex,skeleton_dimension + 1):
                    simplicial_complex[skeleton_dimension].add(frozenset(subsimplex)) # GRRRRRRRRRRRRRRRRRRRRRRRRRRRRR

        self.simplicial_complex = simplicial_complex

    def build_incidence_matrices(self):
        '''
        builds dimension matrices from the simplicial complex
        saves dimension matrices as a list of list of lists
        - the matrices
        - rows of the dimension matrix
        - columns of the dimension matrix

        there will be one fewer matrix than the max dimension in the simplicial complex
        '''

        incidence_matrices = []
       
        for dimension in range(0, self.max_dimension - 1):
            if self.verbose:
                print(f'now on dimension {dimension} of {len(incidence_matrices)}')
            simplex_idx_dict = {simplex: idx for idx, simplex in enumerate(self.simplicial_complex[dimension])}
            incidence_matrix = np.zeros((len(self.simplicial_complex[dimension]), len(self.simplicial_complex[dimension+1])), dtype = int)
            values = [(-1) ** (i+1) for i in range(dimension + 2)]

            for simplex_idx, simplex in enumerate(self.simplicial_complex[dimension + 1]):
                indices = []
                for vertex in sorted(list(simplex)):
                    subsimplex = simplex - {vertex}
                    indices.append(simplex_idx_dict[subsimplex])
                incidence_matrix[indices, simplex_idx] = values
            incidence_matrices.append(incidence_matrix)
        self.incidence_matrices = incidence_matrices
    
    def build_perseus_simplex(self):
        with open(f'{self.data_location}/{self.name}_perseus.txt', 'w+') as f:
            f.write(f'1 \n')
            for complex in self.top_cell_complex:
                line = f'{len(complex) - 1} '
                line += ' '.join([str(x) for x in complex])
                line += f' {len(complex)} \n'
                # pdb.set_trace()
                f.write(line)
        print(f'perseus location: {f"{self.data_location}/{self.name}_perseus.txt"}')

    def calculate_betti_numbers(self):
        '''
        calculates betti numbers using the dimension matrices
        there will be the same number of betti numbers as the max dimension in the simplicial complex
        saves the betti numbers as a list
        '''

        betti_numbers = []
        dim_kers = []
        dim_ims = []
        for i, matrix in enumerate(self.incidence_matrices):
            if matrix.size != 0:
                dim_ker, dim_im = calc_dim_ker_im(matrix)
                try:
                    assert np.linalg.matrix_rank(matrix) == dim_im
                except:
                    print("catching numpy and my linear algebra disagree")
                    pdb.set_trace()
            else:
                dim_ker, dim_im = (0,0)
            dim_kers.append(dim_ker)
            dim_ims.append(dim_im)
            if self.verbose:
                print(f"for matrix {i}, the dimension of the kernal is {dim_ker} and the dimension of the image is {dim_im}")
        # 0th betti number ( #verticies - dim(im(D1))) to calculate number of objects )
        betti_numbers.append(len(self.simplicial_complex[0]) -  dim_ims[0])
        if self.verbose:
            print(f'the {0} betti number is {betti_numbers[-1]}')

        # the rest of the betti numbers
        for i in range(self.max_dimension-2):
            betti_numbers.append(dim_kers[i] - dim_ims[i+1])
            if self.verbose:
                print(f'the {i+1} betti number is {betti_numbers[-1]}')

        # the last betti number is done here because there is no higher dimension matrix thus no image
        betti_numbers.append(dim_kers[-1])
        if self.verbose:
            print(f'the {self.max_dimension - 1} betti number is {betti_numbers[-1]}')
        self.betti_numbers = betti_numbers

    def calculate_betti_sum(self):
        """
        combined with the euler characteristic calculation this can be a sanity check

        arguments: 
        - None

        Returns:
        - None
        """
        betti_sum = 0
        for idx, betti_num in enumerate(self.betti_numbers):
            betti_sum += betti_num * ((-1) ** idx)
        self.betti_sum = betti_sum

    def calculate_euler_characteristic(self):
        """
        combined with the euler characteristic calculation this can be a sanity check
        
        arguments: 
        - None

        Returns:
        - None
        """
        euler_characteristic = 0
        for idx, k_skeleton in self.simplicial_complex.items():
                euler_characteristic += len(k_skeleton) * (-1) ** idx
        self.euler_characteristic = euler_characteristic

    def calculate_all(self, save = False, verbose = None):
        '''
        parameters:
            self
            verbose (bool): if true, print statements to see what is happening
        returns:
            None
        
        runs all functions for building and calculating betti numbers for the simplicilal complex

        specifically this function will build the simplicialt complex, then build the dimension matrices, finally calculate the betti numbers. 
        '''
        if verbose:
            self.verbose = verbose
        
        if save:
            self.save = save
        if self.save:
            self.save_top_complex()

        self.reduce_top_cell_complex()
        self.build_simplicial_complex()
        if self.save:
            self.save_complete_complex()
        # if self.verbose:
        #     print("finished building complex")
        #     print(repr(self))

        self.build_incidence_matrices()
        if self.save:
            self.save_incidence_matrices()
        if self.verbose:
            print(repr(self))

        self.calculate_betti_numbers()
        if self.save:
            self.save_betti_numbers() 


if __name__ == '__main__':
    def test(top_cell_complex, answer, name, *args, **kwargs):
        data_location='../data/simplex_tests'
        complex = SimplicialComplex(top_cell_complex, data_location = data_location, name = name, *args, **kwargs)
        complex.calculate_all()

        assert complex.betti_numbers == answer
        
        complex.calculate_euler_characteristic()
        complex.calculate_betti_sum()

        assert complex.betti_sum == complex.euler_characteristic

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
    # # answer should be [1,3,0,0] 
    # answer = [1,3,0,0,0]
    # top_cell_complex = [[1,2,3,4],[4,5,6,7],[2,5,7],[1,5],[7,8],[8,9],[9,10],[8,9,10],[7,8,9,10],[1,3,5,7],[2,4,6,8],[10,11,12,13,14]] 
    # test(top_cell_complex, answer, name = 'test1', verbose = True , save = True)

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
    answer = [81,0,0,0,0,0]
    top_cell_complex = [[1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16], [17, 18], [19, 20], [21, 22, 23, 24], [25, 26, 27, 17, 18], [21, 22, 23, 24], [14, 16], [28, 29, 30], [31, 32], [33, 18, 34, 35], [36, 37], [38, 39], [21, 23, 40, 41, 24], [18, 42, 43], [44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58], [59, 60, 61], [62, 63, 64, 65], [66, 67, 68, 69], [70, 71], [72, 73], [74, 75, 23], [76, 77, 78, 79], [80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91], [92, 93, 94], [95, 96, 97], [98, 99, 100, 101], [102, 103, 104, 105, 106], [39, 38], [107, 18], [108, 109], [110, 111, 112], [113, 114, 115, 116, 117], [118, 119, 120], [121, 122], [123, 124, 80], [125, 126, 127, 128], [129, 130], [131, 132, 133, 134], [135, 136, 137, 138], [139, 140, 141, 142, 143, 144], [145, 146, 147, 148, 149, 150], [151, 152, 153], [154, 155, 156], [157, 158, 159, 160, 161, 162], [163, 164], [165, 166], [167, 168, 169, 170, 171], [81, 172, 173, 174, 80], [175, 176, 177, 178], [179, 18, 180], [181, 182, 183, 18], [184, 185, 186, 187], [188, 189, 190, 191, 192], [193, 194, 195, 196], [197, 17], [198, 18, 199], [200, 201], [202, 185, 186, 187], [203, 204, 205, 132], [206, 207, 208, 95], [209, 210], [211, 186, 185, 187], [212, 213, 214, 215, 216, 217], [218, 18], [219, 220, 221], [222, 223, 224, 225], [226, 227, 228, 229], [230, 231], [232, 233, 234], [235, 236], [237, 238, 239], [240, 241], [200, 201], [242, 243], [244, 245], [246, 247, 248], [249, 250, 251, 252], [106, 102, 253, 254], [255, 44, 46, 256, 257, 49], [258, 259], [18, 260], [261, 18], [107, 18], [262, 263], [264, 265, 266], [267, 268, 269, 270], [72, 73], [264, 265, 266], [271, 272, 273, 274], [275, 276, 277], [264, 278, 279], [280, 281], [282, 283, 284, 285, 286, 287], [288, 289, 290, 291], [292, 293], [294, 295, 296], [34, 297], [298, 299, 300], [301, 302, 250, 303, 252, 251], [304, 305, 306, 307], [308, 309], [310, 311, 312], [313, 314], [315, 316], [33, 297, 34, 18], [297, 33, 18, 34]]
    test(top_cell_complex, answer, name = 'small_sloth', verbose = True, save = True)
