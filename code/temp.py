import pickle as pkl 
import pdb
path = '../data/examples/example2_level_0_top_cell_complex.pkl'

with open(path, 'rb') as f:
    x = pkl.load(f)

print(x)
pdb.set_trace()