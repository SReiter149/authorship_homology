import numpy as np

import pdb
import traceback
from linear_algebra import calc_dim_ker_im as old_method


class Matrix:
    """
    There will be two forms of this matrix. The first being that to build the matrix, and the second to be that which computes row operations and row reductions.
    """

    def __init__(self, verbose = False, tolerance = 1e-9, warning = 1e-7):
        self.building = True
        self.computing = False
        self.rref_flag = False
        self.verbose = verbose
        self.tolerance = tolerance
        self.warning = warning

        self.nonzero_rows = []
        self.nonzero_columns = []
        self.nonzero_values = []

    def __repr__(self):
        return_string = ""

        if self.building:
            return_string += f"currently in building mode\n"

            return_string += f"values are (row, column, value): \n"
            for i in range(len(self.nonzero_values)):
                return_string += f"({self.nonzero_rows[i]}, {self.nonzero_columns[i]}, {self.nonzero_values[i]}), "
        
        elif self.computing:
            return_string += f"currently in computing mode\n"

            for row_idx, row in self.Matrix.items():
                return_string += f'row {row_idx}: {row}\n'

        return_string += f'\n'
        return return_string
    

    def shape(self):
        if self.building == True:
            return (len(self.nonzero_rows), len(self.nonzero_columns))
        elif self.computing == True:
            return (self.row_count, self.column_count)

    def add_nonzero_value(self, row, column, value):
        assert self.building == True
        if value != 0:
            self.nonzero_rows.append(row)
            self.nonzero_columns.append(column)
            self.nonzero_values.append(value)

    def add_column(self, row_idxs, column_idx, values):
        assert self.building == True
        assert len(row_idxs) == len(values)
        for i in range(len(row_idxs)):
            self.add_nonzero_value(row_idxs[i], column_idx, values[i])


    def convert(self):
        assert self.building == True
        self.building = False
        self.computing = True

        self.row_count = max(self.nonzero_rows) + 1
        self.column_count = max(self.nonzero_columns) + 1

        self.Matrix = {i:[] for i in range(self.row_count)}

        for i in range(len(self.nonzero_rows)):
            self.Matrix[self.nonzero_rows[i]].append([self.nonzero_columns[i], self.nonzero_values[i]])

        for row in self.Matrix.values():
            if not bool(row):
                row.append([-1,None])
            else:
                row = sorted(row)

        if self.verbose:
            print("conversion complete")

    def get_first_nonzero_column(self,row_idx):
        # constant time function
        assert self.computing
        return self.Matrix[row_idx][0][0]
        
    def update_row(self, row_idx, column_idxs, values):
        if self.computing:
            self.Matrix[row_idx] = list(zip(column_idxs, values))

        elif self.building:
            raise NotImplementedError("Hasn't been built yet")
        else:
            raise BrokenPipeError("You shouldn't be able to get here")
        
    def merge_rows(self, row1, row2, check = True):
        """
        row1 can have 0 values
        row2 should not have 0 values
        """
        merged_row = []

        row1_idx = 0
        row2_idx = 0

        while True:
            if row1[row1_idx][0] < row2[row2_idx][0]:
                if row1[row1_idx][1] != 0:
                    merged_row.append(row1[row1_idx])
                row1_idx += 1
                if row1_idx == len(row1):
                    for i in range(row2_idx, len(row2)):
                        merged_row.append(row2[i])
                    return merged_row           
            else:
                merged_row.append(row2[row2_idx])
                row2_idx += 1
                if row2_idx == len(row2):
                    for i in range(row1_idx, len(row1)):
                        merged_row.append(row1[i])
                    return merged_row
                

    def reduce_rows(self,row1_idx, row2_idx):
        """
        will reduce row1 into row2
        returns whether the row is all zero after
        """
        row1 = self.Matrix[row1_idx]
        row2 = self.Matrix[row2_idx]
        # row1_column_idxs, row1_values = self.get_row(row1_idx)
        # row2_column_idxs, row2_values = self.get_row(row2_idx)

        # if self.verbose:
        #     print(f'row 1 {row1_idx}: {row1}')
        #     print(f'row 2 {row2_idx}: {row2}')

        assert row1[0][0] == row2[0][0]

        factor = (-1) * (row2[0][1] / row1[0][1])

        row1_location = 1
        row2_location = 1

        row2[0][1] = 0
        merge_row = []

        while (row1_location < len(row1)) and (row2_location < len(row2)):
            if row1[row1_location][0] == row2[row2_location][0]:
                value = row2[row2_location][1] + factor * row1[row1_location][1]
                if abs(value) > self.tolerance:
                    row2[row2_location][1] = value  
                else:
                    row2[row2_location][1] = 0
                row2_location += 1
                row1_location += 1 
            elif row1[row1_location][0] < row2[row2_location][0]:
                value = factor * row1[row1_location][1]
                if abs(value) > self.tolerance:
                    merge_row.append([row1[row1_location][0], value])
                row1_location += 1    
            else:
                row2_location += 1

        # experiment here
        if bool(merge_row):
            row2 = self.merge_rows(row2[1:], merge_row)
        else:
            row2 = [entry for entry in row2 if entry[1] != 0]

        for i in range(row1_location, len(row1)):
            value = factor * row1[i][1]
            if abs(value) > self.tolerance:
                row2.append([row1[i][0], value])
            i += 1 



        self.Matrix[row2_idx] = row2

        if self.verbose:
            pass
            # print(f'row2 is now {row2}')

        return row2 == []
    
    def reduce_matrix(self):
        assert self.computing

        unchecked_rows = [True for i in range(self.row_count)]
        linearly_dependent_rows  = 0
        for column_idx in range(self.column_count):
            rows = [i for i, val in enumerate(unchecked_rows) if val]
            rows_to_check = len(rows)
            i = 0
            while i < rows_to_check:
                nonzero_column = self.get_first_nonzero_column(rows[i])
                # if nonzero_column == None:
                #     unchecked_rows[rows[i]] = False
                #     i += 1
                if nonzero_column == column_idx:
                    row1_idx = rows[i]
                    unchecked_rows[rows[i]] = False
                    i += 1
                    break  
                else:
                    i += 1
            if 'row1_idx' in locals():
                while i < rows_to_check:

                    nonzero_column = self.get_first_nonzero_column(rows[i])
                    zero_row = False
                    if nonzero_column == None:
                        unchecked_rows[rows[i]] = False
                        i += 1
                    if nonzero_column == column_idx:
                        zero_row = self.reduce_rows(row1_idx, rows[i])
                            
                    if zero_row:
                        linearly_dependent_rows += 1
                        unchecked_rows[rows[i]] = False
                    i += 1
                del row1_idx
            if not any(unchecked_rows):
                break
        return linearly_dependent_rows
    
    def count_pivots(self):
        pivot_columns = set()

        for row in self.Matrix.values():
            if bool(row):
                pivot_columns.add(row[0][0])
        if -1 in pivot_columns:
            pivot_columns.remove(-1)
        return len(pivot_columns)

    def get_sorted_values(self):
        values = []
        for row in self.Matrix:
            for element in row:
                values.append(element[1])
        return sorted(values).reverse()

    
    def dim_ker_im(self):
        assert self.computing

        if self.verbose:
            print(f"finding dim ker and im for matrix size: ({self.row_count}, {self.column_count})")
            # print(repr(self))


        self.reduce_matrix()
        dim_im = self.count_pivots()
        dim_ker = self.column_count - dim_im
        if sorted([abs(x) for x in self.nonzero_values], reverse = True)[-1] < self.warning:
            print(f"there are some values that might be computational arounding errors. For example maybe {sorted([abs(x) for x in self.nonzero_values], reverse = True)[-1]} should be 0")
        return dim_ker, dim_im
        
if __name__ == "__main__":
    import cProfile
    import pstats
    import numpy as np

    def compare_methods(M):
        assert M.computing == True
        row_count, column_count = M.shape()
        numpy_Matrix = np.zeros((row_count, column_count))
        for row_idx, row_values in M.Matrix.items():
            for entry in row_values:
                if entry[1] != None:
                    numpy_Matrix[row_idx, entry[0]] = entry[1]
        # pdb.set_trace()
        # _ = old_method(numpy_Matrix)
        return np.linalg.matrix_rank(numpy_Matrix) 


    def test(tests):
        if 0 in tests:
            print()
            print("--- beginning test 0 ---")
            M = Matrix()
            M.add_column([0],2,[1])
            M.add_column([1,2],3,[1,1])
            M.add_column([0],4,[1])
            print(repr(M))
            M.convert()
            M.reduce_matrix()
            print(repr(M))

        if 1 in tests:
            print()
            print("--- beginning test 1 ---")
            M = Matrix(verbose = True)
            M.add_column([0,3],2,[1,1])
            M.add_column([1,2],3,[1,1])
            M.add_column([0,3],4,[1,1])          
            print(repr(M))
            M.convert()
            numpy_dim_im = compare_methods(M)
            dim_ker, dim_im = M.dim_ker_im()
            assert numpy_dim_im == dim_im
            assert (dim_ker, dim_im) == (3,2)
            print(dim_ker, dim_im)
            print(repr(M))
        if 2 in tests: 
            print()
            print("--- beginning test 2 ---")
            import numpy.random as rand
            M = Matrix(verbose = True)
            for i in range(10):
                M.add_column([i], i, [1])      
            print(repr(M))
            M.convert()
            numpy_dim_im = compare_methods(M)
            dim_ker, dim_im = M.dim_ker_im()
            assert numpy_dim_im == dim_im
            assert dim_ker == 0
            print(dim_ker, dim_im)
            print(repr(M))

        if 3 in tests:
            print()
            print("--- beginning test 3 ---")
            import numpy.random as rand
            M = Matrix(verbose = False)
            dim = 5000
            nonzeros = 4
            for i in range(dim):
                row_idxs = rand.choice(dim,size =nonzeros, replace = False)
                M.add_column(row_idxs, i, [(-1)*j for j in range(nonzeros)])  
                
            # print(repr(M))
            M.convert()
            # print("computing numpy")
            # numpy_dim_im = compare_methods(M)

            print("computing my own")
            dim_ker, dim_im = M.dim_ker_im()

            print(f'the dim_im I found {dim_im}')
            # print(f'compare before {numpy_dim_im}')
            # assert numpy_dim_im == dim_im
            # print(repr(M))
        
        if 4 in tests:
            print()
            print("--- beginning test 4 ---")
            M = Matrix(verbose = True)
            M.add_column([0,2],0,[1,1])
            M.add_column([1,2],1,[1,-1])
            M.add_column([0,1,3],2,[1,1,1])  
            M.add_column([0,2],3,[1,1])   
            M.add_column([3],4,[1]) 
            print(repr(M))
            M.convert()
            dim_ker, dim_im = M.dim_ker_im()
            assert (dim_ker, dim_im) == (2,3)
            print(dim_ker, dim_im)
            print(repr(M))

        if 5 in tests:
            print()
            print("--- beginning test 5 ---")
            M = Matrix(verbose = True)
            M.add_column([0,1,2],0,[-1,1,-1])
            M.add_column([1,3,4],1,[1,-1,-1])
            print(repr(M))
            M.convert()
            dim_ker, dim_im = M.dim_ker_im()
            assert (dim_ker, dim_im) == (0,2)
            print(dim_ker, dim_im)
            # print(repr(M))

    try:
        profiler = cProfile.Profile()
        profiler.enable()
        test([3])
        # test([0,1,2,3,4,5])
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)
        # pdb.set_trace()

    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()    



        





