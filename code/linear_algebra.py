import numpy as np



class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rref_flag = False

    def check_matrix(self):
        '''
        check matrix to make sure it is rectangular, of dimension 2 and is a numpy array
        '''
        assert isinstance(self.matrix, np.ndarray)
        assert len(self.matrix.shape) == 2
        self.matrix = self.matrix.astype(float)


    def find_non_zero_row(self, pivot_row_num, column_num):
        '''
        finds the first row after the pivot row that has a non-zero element in the column
        returns None if no such row exists 
        '''
        row_num = pivot_row_num + 1
        for row_num in range(pivot_row_num, len(self.matrix)):
            row = self.matrix[row_num]
            if row[column_num] != 0:
                return row_num
        return None

    def swap_rows(self, row1, row2):
        '''
        returns a matrix with row1 and row2 swapped 
        '''
        self.matrix[[row1,row2]] = self.matrix[[row2,row1]]

    def eliminate_down(self, pivot_row_num, column_num):
        '''
        reduces every other row beloew by the pivot row of the matrix
        saves the reduced matrix
        '''
        pivot = self.matrix[pivot_row_num, column_num]
        factors = self.matrix[pivot_row_num+1: , column_num] / pivot
        self.matrix[pivot_row_num + 1: ] -= np.outer(factors, self.matrix[pivot_row_num])
    
    '''
    for RREf
    findnonzero-row with column non-zero
    make it the first row
    eliminate down
    '''
    def rref(self):
        '''
        calculate the row reduced row echilon form of the matrix
        '''
        self.check_matrix()
        pivot_row_num = 0
        for column_num in range(len(self.matrix[0])):
            row_num = self.find_non_zero_row(pivot_row_num, column_num)
            if row_num != None:
                self.swap_rows(row_num, pivot_row_num)
                self.eliminate_down(pivot_row_num, column_num)
                pivot_row_num += 1
        self.rref_flag = True


    '''
    calc dim of image
    transpose matrix
    rref matrix
    count non-zero rows
    '''

    def dim_im(self):
        if not self.rref_flag:
            self.rref()
        dim = 0
        for row in self.matrix:
            if np.count_nonzero(row) > 0:
                dim += 1
        return dim

    '''
    for calc dim of kernal
    find dim of image
    use rank nullity thm
    '''

    def dim_ker(self):
        column_count = len(self.matrix[0])
        rank = self.dim_im()
        return column_count - rank
    
def calc_dim_ker_im(matrix):
    '''
    parameters: 
    - matrix (np.array): the input matrix
    returns: 
    - dim_ker (int): dimension of the kernal of the matrix
    - dim_im (int): dimension of the image of the matrix

    uses the Matrix class to calculate the dimension of the kernal and image for the matrix. Wrapper function so that rref only needs to be calculated once. 
    '''
    matrix = Matrix(matrix)
    matrix.rref()
    dim_ker = matrix.dim_ker()
    dim_im = matrix.dim_im()
    return dim_ker, dim_im


if __name__ == '__main__':
    def test(test_matrix, answer):
        return calc_dim_ker_im(test_matrix) == answer

    test_matrix = np.array([[1,2,3],[4,5,6]])
    assert test(test_matrix, (1,2))

    test_matrix = np.array([[1,0,2,-1],[0,1,3,2],[1,1,5,1],[2,1,7,1]])    
    assert test(test_matrix, (1,3))

    test_matrix = np.array([[1,2],[2,4],[3,6],[4,8]])
    assert test(test_matrix, (1,1))