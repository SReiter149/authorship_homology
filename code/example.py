from pipeline import Pipeline
from simplicial_complex import SimplicialComplex
from query_open_alex import papers_by_author, papers_by_topic


# Example 1: if you already have your dataset

directory_path = '../data/examples/'
name = 'example1'

# dataset: one filled in triangle connected to a holed triangle
top_cell_complex = [[0,1,2],[0,3],[3,4],[4,2]]

# creates teh pipeline class and loads the data into it
example_pipeline = Pipeline(directory_path=directory_path, name=name)
example_pipeline.load_data(top_cell_complex= top_cell_complex)

# runs the betti analysis and the distance analysis
example_pipeline.run_distance_analysis(colab1={1}, colab2 = {4})
example_pipeline.run_betti_analysis()


