from pipeline import Pipeline
from query_open_alex import papers_by_author, papers_by_topic


tests = [0,1]

if 0 in tests:
    # distance on deanna and erdos
    # using only math theory papers
    name = 'deanna_distance'
    directory_path = "../data/math/"
    query_filter = 'primary_topic.field.id:fields/26,primary_topic.id:t12170'

    erdos_id =  "A5035271865"
    deanna_id = 'A5039705998'

    pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=True, results = True)
    papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = 'math_theory')
    pipeline.load_data(dataset_path=f'{directory_path}math_theory.json')
    distance = pipeline.run_distance_analysis(colab1 = {deanna_id}, colab2 = {erdos_id}, max_bar_level=1, max_width = 0)
    print(f'distance between deanna and erdos is {distance}')

if 1 in tests:
    # math theory
    name = 'math_theory'
    directory_path = "../data/math/"
    query_filter = 'primary_topic.field.id:fields/26,primary_topic.id:t12170'

    pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=True, results = True)
    papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = name)
    pipeline.load_data()
    pipeline.run_betti_analysis(max_bar_level= 10)

if 2 in tests:
    # Topological and Geometric Data Analysis 
    # from 2020 - 2025
    name = 'computational_topology'
    directory_path = "../data/math/"
    query_filter = 'publication_year:2000+-+2025,primary_topic.id:t12536'

    pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=False, results = True)
    papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = name)
    pipeline.load_data()
    pipeline.run_betti_analysis(max_bar_level= 10)

if 3 in tests:
    # all of math
    name = 'math'
    directory_path = "../data/math/"
    query_filter = 'primary_topic.field.id:fields/26'

    pipeline = Pipeline(name = name, directory_path=directory_path, overwrite= False, verbose=False, results = True)
    papers_by_topic(query_filter=query_filter, directory_path= directory_path, name = name)
    pipeline.load_data()
    pipeline.run_betti_analysis(max_bar_level= 10)

