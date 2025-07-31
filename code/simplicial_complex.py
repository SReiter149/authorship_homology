"""
[1 1 0 1
1 1 1 0
0 1 0 0]
behaves funny in strong collapse
"""

from itertools import combinations
import pdb
import numpy as np 
import linear_algebra as la
import sparse_linear_algebra as sla

import pickle as pkl
import traceback

class SimplicialComplex:
    '''
    simplicial complex class! 
    has all methods for 
    - constructing simplicial complexes 
    - calculating betti numbers

    arguments:
    - top_cell_complex (a set of simplicies)
    - directory_path (path of data)
    - name (string)
    - level (int)
    - verbose (bool)
    - results (bool)
    - save (bool)
    
    '''

    def __init__(self, top_cell_complex, directory_path = '../data', name = 'abstract_complex', level = None, verbose = False, results = False, save = False):
        self.name = name
        self.verbose = verbose
        self.results = results
        self.save = save
        self.level = level
        self.user_vertex_dict = dict()
        self.vertex_dict = dict()

        self.save_path = f'{directory_path}{self.name}'
        if self.level != None:
            self.save_path += f'_level_{level}'

        self.top_cell_complex = dict()
        self.parse_user_top_cell(top_cell_complex)

        # converts the elements to little integers

        clean_top_cell_complex = dict()
        for simplex_id, simplex in enumerate(self.top_cell_complex):
            clean_top_cell_complex[simplex_id] = self.transform_user_simplex(simplex)
            self.max_simplex_id = simplex_id

        self.top_cell_complex = clean_top_cell_complex

        self.max_dimension = max([len(simplex) for simplex in self.top_cell_complex.values()])
        self.vertex_count = len(self.user_vertex_dict.keys())
        self.max_vertex_id = self.vertex_count

    def parse_user_top_cell(self, top_cell_complex):
        """
        builds the top_cell complex from the users top_cell_complex

        arguments:
        - top_cell_complex (user input): the user input top_cell complex. Can parse: 
            - list of lists
            - dictionary of lists
            - dictionary of frozen sets
            - frozen set of frozen sets

        returns:
        - None
        """
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

    def transform_user_simplex(self, user_simplex):
        """
        arguments:
        - user_simplex (iterable): the simplex with the users naming convention

        returns:
        - new_simplex (frozenset): the simplex with internal naming convention
        """
        for vertex_name in user_simplex:
            if vertex_name not in self.user_vertex_dict.keys():
                vertex_id = len(self.user_vertex_dict.keys())
                self.user_vertex_dict[vertex_name] = vertex_id
                self.vertex_dict[vertex_id] = {vertex_name}

        new_simplex = frozenset({self.user_vertex_dict[vertex_name] for vertex_name in user_simplex})

        return new_simplex   

    def __repr__(self):
        '''
        string object for developers

        arguments:
        - None

        returns:
            output (str): a string containing
                - name
                - max dimension
                - size of the k-skeletons (if calculated)
                - dimension matrices (if calculated)
                - betti numbers (if calculated)
        '''
        output = f'\n--------{self.name}-----------\n '

        # max dimension
        output += f'max dimension: {self.max_dimension}\n'

        # top cell complex
        # output += f'top cell complex: {self.top_cell_complex}\n'

        try:
            for idx, k_skeleton in self.abstract_complex.items():
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

    def __str__(self):
        """
        note:
        - the user should rewrite this function for what they would find the most useful

        arguments:
        - None

        returns:
        - string object for the user
            - simplex maps
            - vertex maps
        """
        str = ""
        try:
            str += f"simplex maps: {self.simplex_maps}\n"
            str += f"vertex maps: {self.simplex_maps}\n "
        except:
            pass
        return str


    # ----------SMALL FUNCTIONS----------
    def _add_simplex(self, simplex):
        """
        helper function to add a simplex to the complex. Will give the next available id to the simplex. 
        
        Warning:
        - Assumes that all vertices in the given simplex already are in the complex. 

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

    def _replace_vertex(self, add_vertex_id, remove_vertex_id):
        """
        arguments:
        - add_vertex_id (int): the vertex id for everything in the removed vertex to be added to
        - remove_vertex_id (int): the vertex to remove

        returns:
        - None
        """
        # changes to the simplex_maps and vertex_maps
        for simplex_id in self.vertex_maps[remove_vertex_id]:
            self.simplex_maps[simplex_id] = (self.simplex_maps[simplex_id] - {remove_vertex_id}).union({add_vertex_id})
        
        # merging items in vertex_maps and vertex_dict
        self.vertex_maps[add_vertex_id] = self.vertex_maps[add_vertex_id].union(self.vertex_maps[remove_vertex_id])
        self.vertex_dict[add_vertex_id] = self.vertex_dict[add_vertex_id].union(self.vertex_dict[remove_vertex_id])

        self.vertex_maps.pop(remove_vertex_id)
        self.vertex_dict.pop(remove_vertex_id)

    def _remove_simplex(self, simplex_id):
        """
        helper function to remove a simplex from the data structure.

        arguments:
        - simplex_id (int): the id of the simplex to remove

        returns:
        - none
        """
        
        simplex = self.simplex_maps.pop(simplex_id)
        for vertex_id in simplex:
            self.vertex_maps[vertex_id].remove(simplex_id)
            if not bool(self.vertex_maps[vertex_id]):
                self._remove_vertex(vertex_id)

        self._check_consistency()
       
    def _remove_vertex(self, vertex_id, dominating_vertex_id = None):
        """
        helper_function to remove a vertex from the data structure.

        arguments:
        - vertex_id (int): the id of the vertex to remove

        returns:
        - none
        """
        # changes to the vertex_maps and simplex_maps
        vertex = self.vertex_maps.pop(vertex_id)
        for simplex_id in vertex:
            self.simplex_maps[simplex_id].remove(vertex_id)
            if not bool(self.simplex_maps[simplex_id]):
                self.simplex_maps.pop(simplex_id)

        # changes to the vertex_dict
        if dominating_vertex_id:
            self.vertex_dict[dominating_vertex_id] = (self.vertex_dict[dominating_vertex_id]).union(self.vertex_dict[vertex_id])  
        self.vertex_dict.pop(vertex_id)
        
    def _check_consistency(self):
        """
        debug function that will raise errors

        helper funcion to check that the vertex_maps and simplex_maps agree

        arguments:
        - None

        returns:
        - None
        """

        # ensures everything in the vertex_maps is in the simplex maps
        for vertex_id, simplex_ids in self.vertex_maps.items():
            for simplex_id in simplex_ids:
                if vertex_id not in self.simplex_maps[simplex_id]:
                    print()
                    print(f"Inconsistency detected: vertex {vertex_id} references simplex {simplex_id} but not the other way")
                    # print(f'simplex maps: {self.simplex_maps}')
                    # print(f'vertex maps: {self.vertex_maps}')
                    raise AssertionError(f"Inconsistency detected: vertex {vertex_id} references simplex {simplex_id} but not the other way")

        # ensures everything in the simplex_maps is in the vertex_maps     
        for simplex_id, vertex_ids in self.simplex_maps.items():
            for vertex_id in vertex_ids:
                if simplex_id not in self.vertex_maps[vertex_id]:
                    print()
                    print(f"Inconsistency detected: simplex {simplex_id} references a {vertex_id} but not the other way")
                    print(f'simplex maps: {self.simplex_maps}')
                    print(f'vertex maps: {self.vertex_maps}')
                    raise AssertionError(f"Inconsistency detected: simplex {simplex_id} references a {vertex_id} but not the other way")
                
        # ensures that vertex_dict and vertex_maps have the same vertices
        assert self.vertex_maps.keys() == self.vertex_dict.keys(), pdb.set_trace()

    # -----------SAVE THINGS----------
    def save_top_complex(self): 
        """
        saves the top_cell_compelx to a file
        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_top_cell_complex.pkl', 'wb') as f:
                pkl.dump(self.top_cell_complex, f)
        except:
            pass
        try:
            with open(f'{self.save_path}_top_cell_complex.pkl', 'wb') as f:
                pkl.dump(self.top_cell_complex, f)
        except:
            pass
    def save_reduced_complex(self):
        """
        saves the reduced complex 

        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_reduced_complex.pkl', 'wb') as f:
                pkl.dump(self.abstract_complex, f)
        except:
            pass
        try:
            with open(f'{self.save_path}_reduced_complex.pkl', 'wb') as f:
                pkl.dump(self.abstract_complex, f)
        except:
            pass

    def save_vertex_dict(self):
        """
        saves the vertex dictionary to a file 

        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_vertex_dict.pkl', 'wb') as f:
                pkl.dump(self.vertex_dict, f)
        except:
            pass
        try:
            with open(f'{self.save_path}_vertex_dict.pkl', 'wb') as f:
                pkl.dump(self.vertex_dict, f)
        except:
            pass
    def save_simplex_maps(self):
        """
        saves the simplex maps to a file

        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_simplex_maps.pkl', 'wb') as f:
                pkl.dump(self.simplex_maps, f)
        except:
            pass
        try:
            with open(f'{self.save_path}_simplex_maps.pkl', 'wb') as f:
                pkl.dump(self.simplex_maps, f)
        except:
            pass

    def save_betti_numbers(self):
        """
        saves the betti numbers to a file

        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_betti_numbers.txt', 'w') as f:
                f.write(f'{self.betti_numbers}\n')
                f.close()
        except:
            pass

    def save_complex(self):
        """
        saves the complex and all the stuff to external files. 
        arguments:
        - None

        returns:
        - None
        """
        self.save_top_complex()
        self.save_reduced_complex()
        self.save_vertex_dict()
        self.save_simplex_maps()
        self.save_betti_numbers()
        self.save_betti_results()

    def save_betti_results(self):
        """
        writes all the results from the whole betti number computation to the given path

        arguments:
        - None

        returns:
        - None
        """
        try:
            with open(f'{self.save_path}_results.txt', 'w') as f:
                
                f.write(f'\nfor level {self.level}:\n')
                f.write(f'the maximum dimension before reduction was {self.max_dimension} \n')
                f.write(f'there were {self.vertex_count} vertices\n')
                f.write(f'there were {len(self.top_cell_complex)} simplices in the top cell complex\n')
                f.write(f'the betti numbers are {self.betti_numbers}\n')
        except:
            pass
        
    # ----------HELPER FUNCTION----------

    # -----FOR EDGE CONTRACTION-----
    def star(self, face, complex = None):
        '''
        finds all simplices in the complex for which the given face is a face in the complex

        arguments: 
        - face (set of vertex ids): the face to check the star of 
        - complex (list of simplex ids) (optional): if the search space is smaller than the whole complex

        returns:
        - star_set (a set of simplex ids): implementation of the star operation from Fellegara 2020 Homology 
        '''

        star_set = set.intersection(*[self.vertex_maps[vertex] for vertex in face])

        if complex != None:
            star_set = star_set.intersection(complex)
        
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

    def contract(self,vertex1_id, vertex2_id, star_v1, star_v2, star_edge):
        """
        Fellegara 2020 Homology

        Removes edge (v1,v2) from the simplicial complex
        turns all instances of v1 into v2

        arguments:
        - vertex1_id (int): the first vertex id, the one to be removed
        - vertex2_id (int): the second vertex id, the one to be combined into
        - star_v1 (set of simplex ids): the star of v1
        - star_v2 (set of simplex ids): the star of v2
        - star_edge (set of simplex ids): the star of the edge (v1,v2)

        returns:
        - None  
        """   
        assert type(vertex1_id) == int
        assert type(vertex2_id) == int
        for top_simplex_id in star_edge:
            top_simplex = self.simplex_maps[top_simplex_id]
            gamma_1 = top_simplex - {vertex1_id}
            gamma_2 = top_simplex - {vertex2_id}
            # set of simplex ids
            star_gamma_1 = self.star(gamma_1, complex = star_v2)
            star_gamma_2 = self.star(gamma_2, complex = star_v1)
            if star_gamma_1.union(star_gamma_2) == {top_simplex_id}:
                self._add_simplex(gamma_1)
            self._remove_simplex(top_simplex_id)
            star_v2.remove(top_simplex_id)
            star_v1.remove(top_simplex_id)

        if vertex1_id in self.vertex_dict.keys():
            self._replace_vertex(remove_vertex_id=vertex1_id, add_vertex_id=vertex2_id)
        


    # -----BUILDING------
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

    def build_abstract_complex(self):
        """
        builds the simplicial complex from the simplex map
        takes each top simplex and adds each sub combination to the simplicial complex
        stores each dimension in its own entry in the simplicial complex list
        each dimension is sorted lexigraphically 
        """
        dimension = max(len(simplex) for simplex in self.simplex_maps.values())
        abstract_complex = {i: set() for i in range(dimension)}

        # builds all subsimplices from the top cell complexes
        for simplex in self.simplex_maps.values():
            for skeleton_dimension in range(len(simplex)):
                for subsimplex in combinations(simplex,skeleton_dimension + 1):
                    abstract_complex[skeleton_dimension].add(frozenset(subsimplex)) # GRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
        
        self.complex_dimension = dimension
        self.abstract_complex = abstract_complex

    def build_dense_incidence_matrices(self):
        '''
        builds dimension matrices from the simplicial complex
        saves dimension matrices as a list of list of lists
        - the matrices
        - rows of the dimension matrix
        - columns of the dimension matrix

        there will be one fewer matrix than the max dimension in the simplicial complex
        '''

        incidence_matrices = []
        for dimension in range(0, self.complex_dimension - 1):
            if self.verbose:
                print(f'now building dimension {dimension} of {self.complex_dimension - 1} of the incidence matrices')
                
            simplex_idx_dict = {simplex: idx for idx, simplex in enumerate(self.abstract_complex[dimension])}
            incidence_matrix = np.zeros((len(self.abstract_complex[dimension]), len(self.abstract_complex[dimension+1])), dtype = int)
            values = [(-1) ** (i+1) for i in range(dimension + 2)]

            for simplex_idx, simplex in enumerate(self.abstract_complex[dimension + 1]):
                indices = []
                for vertex in sorted(list(simplex)):
                    subsimplex = simplex - {vertex}
                    indices.append(simplex_idx_dict[subsimplex])
                incidence_matrix[indices, simplex_idx] = values
            incidence_matrices.append(incidence_matrix)
        self.incidence_matrices = incidence_matrices

    def build_sparse_incidence_matrices(self):
        '''
        builds dimension matrices from the simplicial complex
        saves dimension matrices as a list of list of lists
        - the matrices
        - rows of the dimension matrix
        - columns of the dimension matrix

        there will be one fewer matrix than the max dimension in the simplicial complex
        '''

        sparse_incidence_matrices = []
        for dimension in range(0, self.complex_dimension - 1):
            if self.verbose:
                print(f'now building dimension {dimension} of {len(sparse_incidence_matrices)} of the sparse incidence matrices')
                

            subsimplex_idx_dict = dict()


            Matrix = sla.Matrix(verbose = self.verbose)
            values = [(-1) ** (i+1) for i in range(dimension + 2)]

            for simplex_idx, simplex in enumerate(self.abstract_complex[dimension + 1]):
                indices = []
                for vertex in sorted(list(simplex)):
                    subsimplex = simplex - {vertex}
                    if subsimplex not in subsimplex_idx_dict.keys():
                        subsimplex_idx_dict[subsimplex] = len(subsimplex_idx_dict.keys())
                    indices.append(subsimplex_idx_dict[subsimplex])
                Matrix.add_column(indices, simplex_idx, values)
            Matrix.convert()
            sparse_incidence_matrices.append(Matrix)
        self.sparse_incidence_matrices = sparse_incidence_matrices


    # ----------MAIN FUNCTIONS---------- 
    
    def perform_strong_collapses(self):
        """
        Uses strong face reduction to reduce a simplex in a faster way. Will be passed off to edge collapse after. 

        arguments:
        - none

        returns:
        - none
        """
        if self.verbose:
            print(f"before strong collapse\n {repr(self)}")

        # setting up queues
        vertex_queue = [vertex_id for vertex_id in self.vertex_maps.keys()]
        simplex_queue = []
        while vertex_queue: 
            self._check_consistency()
            while vertex_queue:
                # retrieving relivant data
                vertex_id = vertex_queue.pop(0)
                vertex = self.vertex_maps[vertex_id]
                nonzero_simplex_id = self.simplex_maps[next(iter(vertex))]

                for vertex2_id in nonzero_simplex_id:

                    # checking if vertex is dominated by vertex2
                    if vertex_id != vertex2_id and vertex <= self.vertex_maps[vertex2_id]:
                        if self.verbose:
                            print(f'vertex {vertex_id} is being dominated by {vertex2_id}')
                        # removing vertex from relivant places
                        self._remove_vertex(vertex_id, dominating_vertex_id=vertex2_id)
                        
                        # pushing all non-zero columns
                        simplex_queue.extend(vertex)
                        break

            # deleting repeated elements in column queue 
            simplex_queue = [simplex_id for simplex_id in list(set(simplex_queue)) if simplex_id in self.simplex_maps.keys()]
            self._check_consistency()
            while simplex_queue:
                # retrieving relivant data
                simplex_id = simplex_queue.pop(0)
                simplex = self.simplex_maps[simplex_id]
                nonzero_vertex_id = self.vertex_maps[next(iter(simplex))]

                for simplex2_id in nonzero_vertex_id:

                    # checking if simplex is dominated by simplex2
                    if simplex_id != simplex2_id and simplex <= self.simplex_maps[simplex2_id]:
                        if self.verbose:
                            print(f'simplex {simplex_id} is being dominated by {simplex2_id}')
                        # removing simplex from relivant places
                        self._remove_simplex(simplex_id)
                        vertex_queue.extend(simplex)
                        break

            vertex_queue = [vertex_id for vertex_id in list(set(vertex_queue)) if vertex_id in self.vertex_maps.keys()]

    def perform_edge_contraction(self):
        """
        reduces the top cell complex using the above functions

        very unoptimal

        arguments:
        - none

        returns:
        - none
        """
        if self.verbose:
            print(f'before edge contractions \n{repr(self)}')
        current_simplex_id = 0
        while current_simplex_id <= self.max_simplex_id:
            if current_simplex_id in self.simplex_maps.keys():
                vertex2_options = self.simplex_maps[current_simplex_id].copy()
                while vertex2_options:
                    vertex2_id = vertex2_options.pop()
                    vertex1_options = vertex2_options.copy()
                    while vertex1_options:
                        vertex1_id = vertex1_options.pop()
                        star_edge = self.star({vertex1_id, vertex2_id})
                        if bool(star_edge):
                            star_v1 = self.star({vertex1_id})
                            star_v2 = self.star({vertex2_id})
                            
                            if self.link_condition(star_v1, star_v2, star_edge):
                                if self.verbose:
                                    print(f"contracting edge {vertex1_id, vertex2_id}")
                                self.contract(vertex1_id,vertex2_id,star_v1, star_v2, star_edge)
                                if vertex1_id in vertex2_options:
                                    vertex2_options.remove(vertex1_id)
            current_simplex_id += 1
        if self.verbose:
            print(f"after edge contractions {repr(self)}")
    
    def calculate_betti_numbers(self, sparse = True):
        '''
        calculates betti numbers using the dimension matrices
        there will be the same number of betti numbers as the max dimension in the simplicial complex
        saves the betti numbers as a list
        arguments:
        - sparse (bool): whether the incidence matrix is sparse or dense

        returns:
        - None
        '''
        if self.complex_dimension > 1:
            betti_numbers = []
            dim_kers = []
            dim_ims = []
            if sparse:
                """
                this part is done with my own sparse linear algebra package. Is likely a large source of slow down
                """
                for i, matrix in enumerate(self.sparse_incidence_matrices):
                    if matrix.shape() != (0,0):
                        dim_ker, dim_im = matrix.dim_ker_im()
                    else:
                        dim_ker, dim_im = (0,0)

                    dim_kers.append(dim_ker)
                    dim_ims.append(dim_im)
                    if self.verbose or self.results:
                        print(f"for matrix {i}, the dimension of the kernal is {dim_ker} and the dimension of the image is {dim_im}")
                    
            else:
                for i, matrix in enumerate(self.incidence_matrices):
                    if matrix.size != 0:
                        dim_im = np.linalg.matrix_rank(matrix)
                        dim_ker = matrix.shape[1] - dim_im
                        """
                        Here is my attempt to do all the dense linear algebra by myself. Was a cool project but turned out to be several times slower than the state of the art
                        """
                        # dim_ker, dim_im = la.calc_dim_ker_im(matrix)
                        # try:
                        #     assert np.linalg.matrix_rank(matrix) == dim_im
                        # except:
                        #     print("catching numpy and my linear algebra disagree")
                        #     pdb.set_trace()
                    else:
                        dim_ker, dim_im = (0,0)
                    dim_kers.append(dim_ker)
                    dim_ims.append(dim_im)
                    if self.verbose or self.results:
                        print(f"for matrix {i}, the dimension of the kernal is {dim_ker} and the dimension of the image is {dim_im}")
            # 0th betti number ( #verticies - dim(im(D1))) to calculate number of objects )
            betti_numbers.append(len(self.abstract_complex[0]) -  dim_ims[0])
            if self.verbose or self.results:
                print(f'the {0} betti number is {betti_numbers[-1]}')

            # the rest of the betti numbers
            for i in range(self.complex_dimension-2):
                betti_numbers.append(dim_kers[i] - dim_ims[i+1])
                if self.verbose or self.results:
                    print(f'the {i+1} betti number is {betti_numbers[-1]}')

            # the last betti number is done here because there is no higher dimension matrix thus no image
            betti_numbers.append(dim_kers[-1])
            if self.verbose or self.results:
                print(f'the {self.complex_dimension - 1} betti number is {betti_numbers[-1]}')
            while betti_numbers[-1] == 0:
                betti_numbers = betti_numbers[:-1]
            self.betti_numbers = betti_numbers
        else:
            self.betti_numbers =  [len(self.abstract_complex[0])]
            if self.verbose or self.results:
                print(f"there are only single independent nodes, so the betti number is {self.betti_numbers[0]}")
    
    def find_colab_distance(self, colab1, colab2, width = 0):
        """
        finds the distance measure between the two sets of colaborations through the given width
        uses the star operation to make sure that the inersection at each step in the path is atleast dimension width

        note:
        - probably can optimize by searching from both sides, and halting as soon as a path is found
        - by going out in rings you can assure that you halt at the smallest distance

        note:
        - returns -1 if no path is found

        arguments:
        - colab1 (set of ints): the ids of the authors in the first colaboration
        - colab2 (set of ints): the ids of the authors in the second colaboration
        - width (int): the width of the path

        returns:
        - distance (int): the distance between the two colaborations
        """

        colab1 = self.transform_user_simplex(colab1)
        colab2 = self.transform_user_simplex(colab2)

        check1 = False
        check2 = False

        for simplex in self.top_cell_complex.values():
            if colab1 <= simplex:
                check1 = True
            if colab2 <= simplex:
                check2 = True

        if not check1 or not check2:
            return -1

        distance_dict = {simplex_id: 0 for simplex_id in set.intersection(*[self.vertex_maps[vertex] for vertex in colab1])}
        queue = [key for key in distance_dict.keys()]

        while queue:
            current_simplex_id = queue.pop(0)
            next_simplexes = set()
            for vertex_ids in combinations(self.simplex_maps[current_simplex_id], width + 1):
                next_simplexes = next_simplexes.union(self.star(set(vertex_ids)))
            for next_simplex_id in next_simplexes:
                if next_simplex_id in distance_dict.keys():
                    if distance_dict[next_simplex_id] > distance_dict[current_simplex_id] + 1:
                        distance_dict[next_simplex_id] = distance_dict[current_simplex_id] + 1
                        queue.append(next_simplex_id)
                else:
                    distance_dict[next_simplex_id] = distance_dict[current_simplex_id] + 1
                    queue.append(next_simplex_id)


        result = float('inf')
        for simplex_id, vertex_ids in self.top_cell_complex.items():
            if colab2 <= vertex_ids:
                if simplex_id in distance_dict.keys():
                    result = min(result, distance_dict[simplex_id])

        if result == float('inf'):
            result = -1
        return result
        

    # ----------SANITY CHECKS----------
    def calculate_euler_characteristic(self):
        """
        combined with the euler characteristic calculation this can be a sanity check
        
        arguments: 
        - None

        Returns:
        - None
        """
        euler_characteristic = 0
        for idx, k_skeleton in self.abstract_complex.items():
                euler_characteristic += len(k_skeleton) * (-1) ** idx
        self.euler_characteristic = euler_characteristic

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
    
    def build_perseus_simplex(self):
        """
        build the perseus simplex for comparing to the perseus package

        arguments:
        - None

        returns:
        - None
        """
        with open(f'{self.save_path}_perseus.txt', 'w+') as f:
            f.write(f'1 \n')
            for simplex in self.simplex_maps.values():
                # dimension, elements, birthtimes
                line = f'{len(simplex) - 1} '
                line += ' '.join([str(x) for x in simplex])
                line += f' {len(simplex)} \n'
                f.write(line)
        print(f'perseus path: {f"{self.save_path}_perseus.txt"}')

    # ----------RUN THIS----------
    def run_colab_distance(self,colab1, colab2, width = None):
        """
        the callable function to find distance between colaborations

        arguments:
        - colab1 (set of ints): the ids of the authors in the first colaboration
        - colab2 (set of ints): the ids of the authors in the second colaboration
        - width (int): the width of the path

        returns:
        - distance (int): the distance between the two colaborations

        """
        self.build_vertex_simplex_maps()
        if width:
            distance = self.find_colab_distance(colab1, colab2, width)
        else:
            distance = self.find_colab_distance(colab1, colab2)
        return distance

    def run_betti(self,sparse = True, save = None, verbose = None, results = None):
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

        if self.verbose or self.results:
            print(f"\n\nBeginning the calcuations for {self.name}\n")
        
        if save:
            self.save = save

        try:
            self.build_vertex_simplex_maps()
            for _ in range(2):
                self.perform_strong_collapses()
                self.perform_edge_contraction()

            self.build_abstract_complex()
            # self.build_perseus_simplex()
            if sparse:
                self.build_sparse_incidence_matrices()
            else:
                self.build_incidence_matrices()
            self.calculate_betti_numbers(sparse=sparse)
        except Exception:
            print(traceback.format_exc())
            pdb.post_mortem()

        if self.save:
            self.save_complex()


            self.save_complex()



if __name__ == '__main__':
    """
    my random testing functions, feel free to use
    """
    def test(top_cell_complex, answer, name,sparse = True, colab1 = False, colab2 = False, width = 0, *args, **kwargs):
        try:
            directory_path='../data/simplex_tests/'
            complex = SimplicialComplex(top_cell_complex, directory_path = directory_path, name = name, *args, **kwargs)

            if bool(colab1) & bool(colab2):
                distance = complex.run_colab_distance(colab1, colab2, width)
                print(f'distance between {colab1}, and {colab2} is {distance}')

            complex.run_betti(sparse = sparse)

            assert complex.betti_numbers == answer
            
            complex.calculate_euler_characteristic()
            complex.calculate_betti_sum()

            assert complex.betti_sum == complex.euler_characteristic

        except Exception:
            print(traceback.format_exc())
            pdb.post_mortem()
        # compare with perseus
        # complex.build_perseus_simplex()
        # subprocess.run(["arch", "-x86_64", "./perseus",
        #     "nmfsimtop",
        #     f"{directory_path}/{name}_perseus.txt",
        #     f"{directory_path}/{name}_perseus"
        #     ],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     check= True
        #     )
        
        # with open(f"{directory_path}/{name}_perseus_betti.txt", 'r') as f:
        #     for line in f:
        #         perseus_betti = line
        #     perseus_betti = perseus_betti.split(" ")[2:-1]
        #     perseus_betti = [int(x) for x in perseus_betti]
        
        # while len(perseus_betti) < len(answer):
        #     perseus_betti.append(0)
        # print(f'perseus betti: {perseus_betti}')
        # assert perseus_betti == complex.betti_numbers

    tests = [0,1,2,3,4,5,6,7,8]

    if 0 in tests:
        # can't be strong reduced must be edge collapsed
        answer = [1]
        top_cell_complex = [[1,2,3],[2,3,4]] 
        test(top_cell_complex, answer, name = 'test0', colab1 = {1,2}, colab2 = {3,4}, verbose = True , save = True)

    if 1 in tests:
    # my example
    # answer should be [1,3,0,0] 
        answer = [1,3]
        top_cell_complex = [[1,2,3,4],[4,5,6,7],[2,5,7],[1,5],[7,8],[8,9],[9,10],[8,9,10],[7,8,9,10],[1,3,5,7],[2,4,6,8],[10,11,12,13,14]] 
        test(top_cell_complex, answer, colab1 = {12,13,14}, colab2 = {1,2,3}, name = 'test1', verbose = True , save = True)


    if 2 in tests:
        # Chad example
        # answer should be [1,1,0]
        answer = [1,1]
        top_cell_complex = [[1,2,5],[2,3],[3,4],[4,5]] 
        test(top_cell_complex, answer, name = 'test2' , verbose = True, save = True)

    if 3 in tests:
        # Chad exercise 7
        # answer should be [1,2,0]
        answer = [1,2]
        top_cell_complex = [[1,2],[2,3,7],[3,4],[4,5],[5,6],[6,3],[7,8],[8,1]] 
        test(top_cell_complex, answer, name = 'test3', verbose = True, save = True)

    if 4 in tests:
        # three triangles that have a 2 dimensional hole in the middle
        # answer should be [1,1,0]
        answer = [1,1]
        top_cell_complex = [[1,2,3],[2,4,5],[3,5,6]] 
        test(top_cell_complex, answer, name = 'test4', save = True)

    if 5 in tests:
        # three open triangles with a triangle in the middle
        # answer should be [1,4]
        answer = [1,4]
        top_cell_complex = [[1,2],[1,3],[2,3],[2,4],[2,5],[4,5],[3,5],[3,6],[5,6]] 
        test(top_cell_complex, answer, name = 'test5', verbose = False, save = True)

    if 6 in tests:
        # the small sloths test
        answer = [81]
        top_cell_complex = [[1, 2], [3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16], [17, 18], [19, 20], [21, 22, 23, 24], [25, 26, 27, 17, 18], [21, 22, 23, 24], [14, 16], [28, 29, 30], [31, 32], [33, 18, 34, 35], [36, 37], [38, 39], [21, 23, 40, 41, 24], [18, 42, 43], [44, 45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58], [59, 60, 61], [62, 63, 64, 65], [66, 67, 68, 69], [70, 71], [72, 73], [74, 75, 23], [76, 77, 78, 79], [80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91], [92, 93, 94], [95, 96, 97], [98, 99, 100, 101], [102, 103, 104, 105, 106], [39, 38], [107, 18], [108, 109], [110, 111, 112], [113, 114, 115, 116, 117], [118, 119, 120], [121, 122], [123, 124, 80], [125, 126, 127, 128], [129, 130], [131, 132, 133, 134], [135, 136, 137, 138], [139, 140, 141, 142, 143, 144], [145, 146, 147, 148, 149, 150], [151, 152, 153], [154, 155, 156], [157, 158, 159, 160, 161, 162], [163, 164], [165, 166], [167, 168, 169, 170, 171], [81, 172, 173, 174, 80], [175, 176, 177, 178], [179, 18, 180], [181, 182, 183, 18], [184, 185, 186, 187], [188, 189, 190, 191, 192], [193, 194, 195, 196], [197, 17], [198, 18, 199], [200, 201], [202, 185, 186, 187], [203, 204, 205, 132], [206, 207, 208, 95], [209, 210], [211, 186, 185, 187], [212, 213, 214, 215, 216, 217], [218, 18], [219, 220, 221], [222, 223, 224, 225], [226, 227, 228, 229], [230, 231], [232, 233, 234], [235, 236], [237, 238, 239], [240, 241], [200, 201], [242, 243], [244, 245], [246, 247, 248], [249, 250, 251, 252], [106, 102, 253, 254], [255, 44, 46, 256, 257, 49], [258, 259], [18, 260], [261, 18], [107, 18], [262, 263], [264, 265, 266], [267, 268, 269, 270], [72, 73], [264, 265, 266], [271, 272, 273, 274], [275, 276, 277], [264, 278, 279], [280, 281], [282, 283, 284, 285, 286, 287], [288, 289, 290, 291], [292, 293], [294, 295, 296], [34, 297], [298, 299, 300], [301, 302, 250, 303, 252, 251], [304, 305, 306, 307], [308, 309], [310, 311, 312], [313, 314], [315, 316], [33, 297, 34, 18], [297, 33, 18, 34]]
        test(top_cell_complex, answer, name = 'small_sloth', verbose = True, save = True) 

    if 7 in tests:
        # distance test
        top_cell_complex = [[0,1,2,3,4,5,6,7],[4,5,6,7,8,9,10,11],[8,9,10,11,12,13,14,15],[12,13,14,15,16,17,18,19],[16,17,18,19,20,21,22,23]]
        test(top_cell_complex=top_cell_complex, answer = [1],colab1={1,2,3,4}, colab2 = {20,21,22,23},name = 'distance_test', width = 3)

    if 8 in tests:
        # distance test 2
        top_cell_complex = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[0,5]]
        test(top_cell_complex=top_cell_complex, answer = [1,1],colab1={0,1}, colab2 = {4,5},name = 'distance_test2', width = 2)
    