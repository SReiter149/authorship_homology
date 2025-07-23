import numpy as np


import pdb
import traceback

class Matrix:
    """
    There will be two forms of this matrix. The first being that to build the matrix, and the second to be that which computes row operations and row reductions.
    """


    def __init__(self):
        self.building = True
        self.computing = False
        self.rref_flag = False

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
                    end_idx = self.last_idx

                for i in range(start_idx, end_idx):
                    return_string += f"({row_idx}, {self.nonzero_columns[i]}, {self.nonzero_values[i]}), "

        return return_string

                


    
    def _check_consistency(self):
        if self.building:
            assert len(self.nonzero_columns) == len(self.nonzero_rows) == len(self.nonzero_values)
        
        elif self.computing:
            assert len(self.nonzero_columns) == len(self.nonzero_values)


    def add_nonzero_value(self, row, column, value):
        assert self.building == True
        self.nonzero_rows.append(row)
        self.nonzero_columns.append(column)
        self.nonzero_values.append(value)

    def convert(self):
        assert self.building == True
        self.building = False
        self.computing = True

        self.row_count = max(self.nonzero_rows) + 1
        self.column_count = max(self.nonzero_columns) + 1

        row_permutation = np.argsort(self.nonzero_rows).tolist()


        self.nonzero_columns = [self.nonzero_columns[i] for i in row_permutation]
        self.nonzero_rows = [self.nonzero_rows[i] for i in row_permutation]
        self.nonzero_values = [self.nonzero_values[i] for i in row_permutation]
        self.last_idx = len(self.nonzero_values)


        self.row_starts = [0 for _ in range(self.nonzero_rows[0] + 1)]
        for row_number in range(1,len(self.nonzero_rows)):
            self.row_starts.extend([row_number] * (self.nonzero_rows[row_number] - self.nonzero_rows[row_number-1]))

        for start_idx, end_idx in self.iterate_over_rows():
            columns_permutation = np.argsort(self.nonzero_columns[start_idx: end_idx])
            self.nonzero_columns[start_idx: end_idx] = [self.nonzero_columns[i + start_idx] for i in columns_permutation]
            self.nonzero_values[start_idx: end_idx] = [self.nonzero_values[start_idx + i] for i in columns_permutation]
    
    def iterate_over_rows(self):
        for row_idx in range(len(self.row_starts)):
            start_idx, end_idx = self.get_row_start_end(row_idx)
            yield start_idx, end_idx
    
    def get_row_start_end(self, row_idx):
        start_idx = self.row_starts[row_idx]
        try:
            end_idx = self.row_starts[row_idx + 1]
        except:
            end_idx = self.last_idx
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

        assert row1_column_idxs[0] == row2_column_idxs[0]

        factor = (-1) * (row2_values[0] / row1_values[0])

        row1_location = 0
        row2_location = 0

        row1_value_counts = len(row1_values)
        row2_value_counts = len(row2_values)

        while (row1_location < row1_value_counts) and (row2_location < row2_value_counts):
            if row1_column_idxs[row1_location] == row2_column_idxs[row2_location]:
                row2_values[row2_location] += factor * row1_values[row1_location]
                row1_location += 1 
            elif row1_column_idxs[row1_location] < row2_column_idxs[row2_location]:
                row1_location += 1
            
            else:
                row2_location += 1

        self.update_row(row2_idx, row2_values)

        return all(value == 0 for value in row2_values)
    
    def reduce_matrix(self):
        assert self.computing

        unchecked_rows = [i for i in range(self.row_count)]
        linearly_independent_rows_count = 0
        for column_idx in range(self.column_count):
            i = 0
            while i < len(unchecked_rows):
                row1_columns, row1_values = self.get_row(unchecked_rows[i])
                if row1_columns[0] == column_idx:
                    row1_idx = unchecked_rows[i]
                    unchecked_rows.pop(i)
                    linearly_independent_rows_count += 1
                    break  
                i += 1
            if 'row1_idx' in locals():
                while i < len(unchecked_rows):
                    row2_columns, row2_values = self.get_row(unchecked_rows[i])
                    zero_row = False
                    if row2_columns[0] == column_idx:
                        """
                        wasted computation here cuz we have columns and values for both rows so fix later maybe
                        """
                        zero_row = self.reduce_rows(row1_idx, unchecked_rows[i])
                    if zero_row:
                        unchecked_rows.pop(i)
                    else:
                        i += 1
                del row1_idx
            if not bool(unchecked_rows):
                break
        return linearly_independent_rows_count
    
    def dim_ker_im(self):
        assert self.computing

        dim_im = self.reduce_matrix()
        dim_ker = self.column_count - dim_im
        return dim_ker, dim_im
        
if __name__ == "__main__":
    try:
        tests = [1]

        if 0 in tests:
            M = Matrix()
            M.add_nonzero_value(0,2,1)
            M.add_nonzero_value(0,4,1)
            M.add_nonzero_value(1,3,1)
            M.add_nonzero_value(2,3,1)
            print(repr(M))
            M.convert()
            M.reduce_matrix()
            print(repr(M))

        if 1 in tests:
            M = Matrix()
            M.add_nonzero_value(0,4,1)
            M.add_nonzero_value(0,2,1)
            M.add_nonzero_value(1,3,1)
            M.add_nonzero_value(2,3,1)
            M.add_nonzero_value(3,4,1)
            M.add_nonzero_value(3,2,1)            
            print(repr(M))
            M.convert()
            dim_ker, dim_im = M.dim_ker_im()
            print(dim_ker, dim_im)
            print(repr(M))
    
    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()    



        





