import requests
import json
import pdb
import time
import os
import traceback

def query(query_filter, verbose = False):
    """
    arguments:
    - query_filer (str): query filter in the form that openAlex wants
    - verbose (bool):  whether to print updates

    yields:
    - results (dictionary): dictionaries 1000 papers at a time to handle as to not overload memory
    """
    cursor = '*'
    results = []
    page = 1
    total_pages = 100
    while page <= total_pages:  
        query_url = f'https://api.openalex.org/works?filter={query_filter}&select=display_name,authorships&per-page=200&cursor={cursor}'
        r = requests.get(url = query_url)
        data = r.json()
        results.extend(data['results'])
        cursor = data['meta']['next_cursor']
        if page == 1:
            total_pages = 1 + data['meta']['count']//200
        if verbose:
            print(f'on page {page} of {total_pages}: there were {len(data["results"])} results')
        if page % 5 == 0:
            yield results, False
            results = []     
        page += 1
        # openAlex wants a sleep but its slow, plus cleaning the data takes time so maybe that counts
        # time.sleep(0.1)
    yield results, True

def format_papers(results):
    """
    arguments:
    - results (dictionary): dictionary in the format openAlex spits out

    returns:
    - paper_dict (dictionary): dictionary in the style I use
    """
    paper_dict = {}
    for paper in results:
        author_ids = []
        # pdb.set_trace()
        for author in paper['authorships']:
            author_ids.append(author['author']['id'].split("/")[-1])
        if paper['display_name']:
            paper_name = paper['display_name']
        else:
            paper_name = "unknown"
        paper_dict[paper_name] = author_ids
    return paper_dict

def write_results(paper_dictionary, directory_path, file_name, start = True, end = True, verbose = False):
    """
    writes the results to the file

    arguments:
    - paper_dictionary (dictionary): the dictionary to write to the file
    - directory_path (path): path of the folder to save the files in 
    - file_name (str): the base name of the file to save in
    - start (bool): whether this is the first time writing in this file
    - end (bool): whether this is the last time writing in this file
    - verbose (bool): whether to print updates

    returns:
    - None
    """  
    assert directory_path[-1] == "/"  
    if verbose:
        print(f'writing results to {directory_path}')

    os.makedirs(directory_path, exist_ok = True)
    full_path = f'{directory_path}{file_name}'
    
    if start:
        f = open(full_path, 'w+')
        f.write('{')
    else:
        f = open(full_path, "a+")
    for key,value in paper_dictionary.items():
        if not start:
            f.write(', ')
        else:
            start = False 
        json.dump(key, f)  
        f.write(': ')
        json.dump(value, f)
    
    if end:
        f.write('}')     
    f.close()

def papers_by_author(seed_id = 'A5029009134', name = 'Kate_Meyer', directory_path = f"../data/query_tests/", max_rounds = 2, overwrite = True, verbose = False):
    '''
    starts with a seed ID and gets all papers and their co-authors, then all co-author's co-authors and so on for max_rounds times. 
    arguments:
    - seed_id (str): the openAlex ID for the seed author
    - name (str): name of the author to save files with
    - directory_path (path): the path of the folder to sasve files in
    - overwrite (bool): whether to write over a file if it has content 
    - verbose (bool): whether to print updates

    returns: 
    - dataset (dictionary): keys = paper_name, values = author_ids
    '''
    authors = [{seed_id},]
    combined_name = f"{name}.json"
    for round_number in range(max_rounds):
        if verbose:
            print(f"---- beginning round {round_number} ----")
        file_name = f"{name}_round{round_number}.json"
        start = True
        end = False
        this_round_authors = authors[-1].copy()
        next_round_authors = set()
        if overwrite == True or not os.path.isfile(f'{directory_path}{file_name}') or os.stat(f'{directory_path}{file_name}').st_size == 0:
            while this_round_authors:
                author_id = this_round_authors.pop()
                if verbose:
                    print(f'getting results for {author_id}')

                
                for results, last_flag in query(query_filter = f'author.id:{author_id}', verbose = verbose):
                    if not bool(this_round_authors) and last_flag:
                        end = True

                    # writing the current round of papers into round file
                    paper_dictionary = format_papers(results)
                    write_results(paper_dictionary=paper_dictionary, directory_path=directory_path, file_name=file_name, start = start, end = end)

                    if round_number != max_rounds - 1:
                        for author_ids in paper_dictionary.values():
                            next_round_authors = next_round_authors.union(author_ids)
                    start = False
                

            # removing authors from previous rounds that have already been checked


        else:
            # if not writing and querying this round
            f = open(f'{directory_path}{file_name}', 'r')
            last_round_data = json.load(f)
            next_round_authors = set()
            for author_ids in last_round_data.values():
                next_round_authors = next_round_authors.union(author_ids)


        f = open(f'{directory_path}{file_name}', 'r')
        last_round_data = json.load(f)
        write_results(paper_dictionary=last_round_data, directory_path=directory_path, file_name=combined_name, start = round_number == 0, end = round_number == max_rounds - 1)

        next_round_authors = next_round_authors.union(author_ids)
        for last_authors in authors:
            next_round_authors -= last_authors 
        authors.append(next_round_authors)
        start = False



def papers_by_topic(query_filter = 'title.search:Choloepus', directory_path = f"../data/query_tests/", name = 'small_sloths', overwrite = True, verbose = False):
    """
    note:
    - the basic search is 'title.search:' and just checks to see if the following string is in the title of the papers

    arguments:
    - query_filter (string): the filter by which to search the openAlex API with.
    - name (string): name of the base name to save by 
    - directory_path (path): path to the folder to save in
    - overwrite (bool): whether to write over a file if it has content 
    - verbose (bool): whether to print occasional updates 

    returns:
    - None
    """
    # build the dataset
    file_name = f'{name}.json'
    if overwrite == True or not os.path.isfile(f'{directory_path}{file_name}') or os.stat(f'{directory_path}{file_name}').st_size == 0:
        start = True
        for results, last_flag in query(query_filter):
            paper_dictionary = format_papers(results)
            write_results(paper_dictionary=paper_dictionary, directory_path=directory_path, file_name=file_name,verbose=verbose, start = start, end= last_flag)
            start = False


if __name__ == '__main__':
    """
    my random test functions
    """
    try:
        # papers_by_topic(overwrite = True)
        papers_by_author(overwrite = True, verbose = True)
    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()

    # output = query('https://api.openalex.org/works?filter=authorships.author.id:a5043805916,authorships.author.id:a5051868975')
    # # results = query('https://api.openalex.org/works?filter=default.search:two%252520toed%252520sloth%7CCholoepus')
    # for results in output:
    #     paper_dict = format_papers(results)

