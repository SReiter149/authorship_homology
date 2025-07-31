# authorship Homology

## About this project
This is a project that is based on finding holes and distances amungst authorship data. Specifically I used tools from homology to find the betti numbers and simplicial distance in biological and mathematical authorship networks. 
### Data Source

This project uses data from **OpenAlex**—a fully‑open index of scholarly works, authors, venues, institutions, and concepts.
Priem, J., Piwowar, H., & Orr, R. (2022). *OpenAlex: A fully‑open index of scholarly works, authors, venues, institutions, and concepts*. arXiv.  [oai_citation:1‡OpenAlex](https://help.openalex.org/hc/en-us/articles/28761511652247-How-can-I-cite-OpenAlex?utm_source=chatgpt.com) [oai_citation:2‡OpenAlex Documentation](https://docs.openalex.org/?utm_source=chatgpt.com)

## Getting started 
make sure you have
```
numpy
matplotlib
networkx
```
then clone and cd into the project
```
git clone https://github.com/SReiter149/authorship_homology.git
cd authorship-homology
```

## output file names
by convention I have the base name of the file be "name" or "name_level_x" if you are working with various level. These base names with different extensions based on what type of file it is. The important ones are:
- _betti_results.txt are the betti numbers
- _distance_results.txt are the distance results
- _graph.png the pictures for the distance visualization
- _results the summary statistics for the level


### betti_results
Contains information about the betti numbers. In the list of betti results, the first number is always H_0, if there are hole at dimension x, then there will be 0s after H_0 until the x-th element of the list, which will contain the quantity of holes. 

### distance_results


## Quick Start
Example.py (also in the code directory): 
```
from pipeline import Pipeline
from query_open_alex import papers_by_author, papers_by_topic
import graph


# ----- Example 1 -----
# you already have your dataset

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
pipeline.run_distance_analysis({'A5009166574'}, {'A5013791380'}, max_bar_level=3, max_width=2)
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
```
