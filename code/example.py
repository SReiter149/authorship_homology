from pipeline import Pipeline
from simplicial_complex import SimplicialComplex
from query_open_alex import papers_by_author, papers_by_topic


# Example 1: if you already have your dataset
directory_path = '../data/examples/'
name = 'example1'

# dataset: one filled in triangle connected to a holed triangle
top_cell_complex = [[0,1,2],[0,3],[3,4],[4,2]]

# creates the pipeline class and loads the data into it
example_pipeline = Pipeline(directory_path=directory_path, name=name)
example_pipeline.load_data(top_cell_complex= top_cell_complex)

# runs the betti analysis and the distance analysis
example_pipeline.run_distance_analysis(colab1={1}, colab2 = {4})
example_pipeline.run_betti_analysis()


# Example 2: if you want to query open Alex
name = 'example2'
directory_path = '../data/examples/'
query_filter = 'title.search:Choloepus'

# creates the pipeline class and loads the data into it
pipeline = Pipeline(name = name, directory_path= directory_path)
papers_by_topic(query_filter=query_filter, directory_path=directory_path, name = name)
pipeline.load_data()

# runs the betti analysis and the distance analysis
pipeline.run_distance_analysis({'A5009166574'}, {'A5013791380'}, max_bar_level=3, max_width=2)
pipeline.run_betti_analysis(max_bar_level=3)

