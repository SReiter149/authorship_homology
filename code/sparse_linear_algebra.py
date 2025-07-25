import numpy as np


import pdb
import traceback

class Matrix:
    """
    There will be two forms of this matrix. The first being that to build the matrix, and the second to be that which computes row operations and row reductions.
    """


    def __init__(self, verbose = False):
        self.building = True
        self.computing = False
        self.rref_flag = False
        self.verbose = verbose

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

            return_string += f"values are (row, column, value): \n"

            for row_idx in range(len(self.row_starts)):
                start_idx = self.row_starts[row_idx]
                try:
                    end_idx = self.row_starts[row_idx + 1]
                except:
                    end_idx = len(self.nonzero_values)

                for i in range(start_idx, end_idx):
                    return_string += f"({row_idx}, {self.nonzero_columns[i]}, {self.nonzero_values[i]}), "

        return return_string
    
    def _check_consistency(self):
        if self.building:
            assert len(self.nonzero_columns) == len(self.nonzero_rows) == len(self.nonzero_values)
        
        elif self.computing:
            assert len(self.nonzero_columns) == len(self.nonzero_values)

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
        for i in range(len(row_idxs)):
            self.add_nonzero_value(row_idxs[i], column_idx, values[i])


    def convert(self):
        assert self.building == True
        self.building = False
        self.computing = True

        self.row_count = max(self.nonzero_rows) + 1
        self.column_count = max(self.nonzero_columns) + 1

        matrix_tuples = sorted(list(set(list(zip(self.nonzero_rows, self.nonzero_columns, self.nonzero_values)))))

        self.nonzero_rows, self.nonzero_columns, self.nonzero_values = list(zip(*matrix_tuples))

        self.nonzero_rows = list(self.nonzero_rows)
        self.nonzero_columns = list(self.nonzero_columns)
        self.nonzero_values = list(self.nonzero_values)

        self.row_starts = [0 for _ in range(self.nonzero_rows[0] + 1)]
        for row_number in range(1,len(self.nonzero_rows)):
            self.row_starts.extend([row_number] * (self.nonzero_rows[row_number] - self.nonzero_rows[row_number-1]))
        # pdb.set_trace()
        self.first_nonzero = self.row_starts.copy()
        if self.verbose:
            print("conversion complete")

    def get_first_nonzero(self,row_idx):
        # constant time function
        assert self.computing

        nonzero_idx = self.first_nonzero[row_idx]
        if nonzero_idx != None:
            column_idx = self.nonzero_columns[nonzero_idx]
            value = self.nonzero_values[nonzero_idx]
        else:
            column_idx = None
            value = None
        return value, column_idx
    
    def update_first_nonzero(self, row_idx):
        # O(n) where n is the number of saved elements in the row
        assert self.computing

        current_idx = self.first_nonzero[row_idx]
        _, end_idx = self.get_row_start_end(row_idx)

        while self.nonzero_values[current_idx] == 0:
            current_idx += 1
            if current_idx >= end_idx:
                current_idx = None
                break
        self.first_nonzero[row_idx] = current_idx

    def iterate_over_rows(self):
        for row_idx in range(len(self.row_starts)):
            start_idx, end_idx = self.get_row_start_end(row_idx)
            yield start_idx, end_idx
    
    def get_row_start_end(self, row_idx):
        # O(1)
        start_idx = self.row_starts[row_idx]
        if row_idx + 1 < len(self.row_starts):
            end_idx = self.row_starts[row_idx + 1]
        else:
            end_idx = len(self.nonzero_values)
        return start_idx, end_idx

    
    def get_row(self, row_idx):
        """
        will return column_idx and values
        """
        if self.computing:
            start_idx, end_idx = self.get_row_start_end(row_idx)
            
            column_idxs = self.nonzero_columns[start_idx: end_idx]
            row_values = self.nonzero_values[start_idx: end_idx]
            return column_idxs, row_values
        elif self.building:
            raise NotImplementedError("Hasn't been built yet")
        else:
            raise BrokenPipeError("You shouldn't be able to get here")
        
    def update_row(self, row_idx, values):
        if self.computing:
            start_idx, end_idx = self.get_row_start_end(row_idx)
            
            assert len(values) == end_idx - start_idx

            self.nonzero_values[start_idx: end_idx] = values

        elif self.building:
            raise NotImplementedError("Hasn't been built yet")
        else:
            raise BrokenPipeError("You shouldn't be able to get here")

    def reduce_rows(self,row1_idx, row2_idx):
        """
        will reduce row1 into row2
        returns whether the row is all zero after
        """
        row1_column_idxs, row1_values = self.get_row(row1_idx)
        row2_column_idxs, row2_values = self.get_row(row2_idx)

        nonzero_value1, nonzero_column1 = self.get_first_nonzero(row1_idx)
        nonzero_value2, nonzero_column2 = self.get_first_nonzero(row2_idx)
        if self.verbose:
            print(f'row 1 {row1_idx}: values: {row1_values}, columns: {row1_column_idxs}')
            print(f'row 2 {row2_idx}: values: {row2_values}, columns: {row2_column_idxs}')

        assert nonzero_column1 == nonzero_column2

        factor = (-1) * (nonzero_value2 / nonzero_value1)

        # could be fancier w/ this
        row1_location = 0
        row2_location = 0

        row1_value_counts = len(row1_values)
        row2_value_counts = len(row2_values)

        while (row1_location < row1_value_counts) and (row2_location < row2_value_counts):
            if row1_column_idxs[row1_location] == row2_column_idxs[row2_location]:
                row2_values[row2_location] += factor * row1_values[row1_location]
                row1_location += 1 
                row2_location += 1
            elif row1_column_idxs[row1_location] < row2_column_idxs[row2_location]:
                row1_location += 1
            
            else:
                row2_location += 1

        self.update_row(row2_idx, row2_values)
        self.update_first_nonzero(row2_idx)

        if self.verbose:
            print(f'row2 is now {row2_values}')
        # pdb.set_trace()
        return self.first_nonzero[row2_idx] == None
    
    def reduce_matrix(self):
        assert self.computing

        unchecked_rows = [True for i in range(self.row_count)]
        linearly_dependent_rows  = 0
        for column_idx in range(self.column_count):
            rows = [i for i, val in enumerate(unchecked_rows) if val]
            i = 0
            while i < len(rows):
                _, nonzero_column = self.get_first_nonzero(rows[i])
                if nonzero_column == None:
                    unchecked_rows[rows[i]] = False
                    i += 1
                elif nonzero_column == column_idx:
                    row1_idx = rows[i]
                    unchecked_rows[rows[i]] = False
                    linearly_dependent_rows += 1
                    i += 1
                    break  
                else:
                    i += 1
            if 'row1_idx' in locals():
                while i < len(rows):

                    _, nonzero_column = self.get_first_nonzero(rows[i])
                    zero_row = False
                    if nonzero_column == None:
                        unchecked_rows[rows[i]] = False
                        i += 1
                    else:
                        if nonzero_column == column_idx:
                            zero_row = self.reduce_rows(row1_idx, rows[i])
                        if zero_row:
                            unchecked_rows[rows[i]] = False
                        i += 1
                del row1_idx
            if not any(unchecked_rows):
                break
        return linearly_dependent_rows
    
    def dim_ker_im(self):
        assert self.computing

        if self.verbose:
            print(f"finding dim ker and im for matrix size: ({self.row_count}, {self.column_count})")
            print(repr(self))


        dim_im = self.reduce_matrix()
        dim_ker = self.column_count - dim_im
        return dim_ker, dim_im
        
if __name__ == "__main__":
    import cProfile
    import pstats
    import numpy as np

    def compare_numpy(Matrix):
        assert Matrix.computing == True
        numpy_Matrix = np.zeros((Matrix.row_count, Matrix.column_count))
        for i in range(len(Matrix.nonzero_values)):
            numpy_Matrix[Matrix.nonzero_rows[i], Matrix.nonzero_columns[i]] = Matrix.nonzero_values[i]
        
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
            dim_ker, dim_im = M.dim_ker_im()
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
            dim_ker, dim_im = M.dim_ker_im()
            assert dim_ker == 0
            print(dim_ker, dim_im)
            print(repr(M))

        if 3 in tests:
            print()
            print("--- beginning test 3 ---")
            import numpy.random as rand
            M = Matrix(verbose = False)
            for i in range(40):
                col_idx = rand.randint(0,40,4)
                M.add_column(col_idx, i, [1,-1,1,-1])  


                
            # print(repr(M))
            M.convert()
            compare_before = compare_numpy(M)
            
            dim_ker, dim_im = M.dim_ker_im()

            compare_after = compare_numpy(M)

            print(f'the dim_im I found {dim_im}')
            print(f'compare before {compare_before}')
            print(f'compare after {compare_after}')
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
            assert (dim_ker, dim_im) == (1,4)
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
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("tottime")
        # stats.print_stats(20)

    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()    



        





