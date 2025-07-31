from pipeline import Pipeline
from query_open_alex import papers_by_author, papers_by_topic
import graph


# ----- Example 1 -----
# you already have your dataset

directory_path = '../data/examples/'
name = 'example1'

# dataset: one filled in triangle connected to a holed quadralateral
top_cell_complex = [[0,1,2],[0,3],[3,4],[4,2]]

# creates the pipeline class and loads the data into it
example_pipeline = Pipeline(directory_path=directory_path, name=name)
example_pipeline.load_data(top_cell_complex= top_cell_complex)

# runs the betti analysis and the distance analysis
# by default run_distance_analysis runs with max_bar_level 1 and passes through vertices (similar to graph theory)
example_pipeline.run_distance_analysis(colab1={1}, colab2 = {4})

# by default run_bett_analysis runs with max_bar_level of 1
example_pipeline.run_betti_analysis()


# ----- Example 2 -----
# you want to query openAlex for a topic

name = 'example2'
directory_path = '../data/examples/'
query_filter = 'title.search:Choloepus'
author1_id = 'A5009166574'
author2_id = 'A5013791380'

# creates the pipeline class and loads the data into it
pipeline = Pipeline(name = name, directory_path= directory_path)
papers_by_topic(query_filter=query_filter, directory_path=directory_path, name = name)
pipeline.load_data()

# runs the distance analysis and makes plots to visualize the distance
pipeline.run_distance_analysis({author1_id}, {author2_id}, max_bar_level=3, max_width=2)
for level in range(3):
    graph.main(data_directory_path=directory_path, save_directory_path=directory_path, name = f'{name}_level_{level}', special_nodes= {author1_id, author2_id})

# runs the betti analysis and saves to files
pipeline.run_betti_analysis(max_bar_level=3)

# ----- Example 3 -----
# you want to query openAlex for an author

name = 'example3'
directory_path = '../data/examples/'
author_id = 'A5060197447'

# query and load the data for the single author
pipeline = Pipeline(name = name, directory_path= directory_path)
papers_by_author(seed_id = author_id, name = name, directory_path=directory_path, max_rounds = 2)
pipeline.load_data()

# run betti analysis
pipeline.run_betti_analysis(max_bar_level=2)
