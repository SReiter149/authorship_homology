import requests
import json
import pdb
import time
import os
import traceback

def query(query_filter, verbose = False):
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

            yield results
            results = []     
        page += 1
    yield results

def format_papers(results):
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

def write_results(paper_dictionary, folder_path, file_name, start = True, end = True, verbose = False):  
    assert folder_path[-1] == "/"  
    if verbose:
        print(f'writing results to {folder_path}')

    os.makedirs(folder_path, exist_ok = True)
    full_path = f'{folder_path}{file_name}'
    
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

def papers_by_author(seed_id = 'A5029009134', name = 'Kate_Meyer', folder_path = f"../data/people/", max_rounds = 2, write_rounds = 0, verbose = False):
    '''
    starts with a seed ID and gets all papers and their co-authors, then all co-author's co-aturhos.
    returns a dictionary with the paper name as key and authorIDs as values
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
        while this_round_authors:
            author_id = this_round_authors.pop()
            if verbose:
                print(f'getting results for {author_id}')
            if not bool(this_round_authors):
                end = True
            for results in query(query_filter = f'author.id:{author_id}', verbose = verbose):
                # pdb.set_trace()
                paper_dictionary = format_papers(results)
                write_results(paper_dictionary=paper_dictionary, folder_path=folder_path, file_name=file_name, start = start, end = end)

                write_results(paper_dictionary=paper_dictionary, folder_path=folder_path, file_name=file_name, start = round_number == 0, end = (round_number == max_rounds-1 and end))

                if round_number != max_rounds - 1:
                    for author_ids in paper_dictionary.values():
                        next_round_authors = next_round_authors.union(author_ids)

                # pdb.set_trace()
            start = False
            for last_authors in authors:
                next_round_authors -= last_authors 
        authors.append(next_round_authors)



def papers_by_topic(query_filter = 'title.search:Choloepus', name = 'small_sloths', folder_path = f"../data/query_tests/",verbose = False):
    # build the dataset
    file_name = f'{name}.json'
    for results in query(query_filter):
        paper_dictionary = format_papers(results)
        write_results(paper_dictionary=paper_dictionary, folder_path=folder_path, file_name=file_name,verbose=verbose)


if __name__ == '__main__':
    try:
        papers_by_topic()
        # papers_by_author(verbose = True)
    except Exception:
        print(traceback.format_exc())
        pdb.post_mortem()

    # output = query('https://api.openalex.org/works?filter=authorships.author.id:a5043805916,authorships.author.id:a5051868975')
    # # results = query('https://api.openalex.org/works?filter=default.search:two%252520toed%252520sloth%7CCholoepus')
    # for results in output:
    #     paper_dict = format_papers(results)


# Old code: 


# def getWorks(authorID):
#     '''
#     This function has no return statement but can easily be formatted to return all the papers or co-authors for a particular author
#     '''
#     works = f'https://api.openalex.org/works?filter='
#     r = requests.get(url = works)
#     data = r.json()
#     # pdb.set_trace()
#     papers = dict()
#     for paper in range(len(data['results'])):
#         authorIDs = []
#         for author in range(len(data['results'][paper]['authorships'])):
#             authorIDs.append(data['results'][paper]['authorships'][author]['author']['id'].split("/")[-1])
#         # pdb.set_trace()
#         papers[data['results'][paper]['title']] = authorIDs
#     return papers
"""
# Carleton = "https://api.openalex.org/institutions?search=Carleton%20College"
# topics = 'https://api.openalex.org/topics'
# math = 'https://api.openalex.org/topics?filter=display_name.search:math&per_page=100'
# concepts = 'https://api.openalex.org/topics?search=Mathematics'
# author = 'https://api.openalex.org/authors?search=Kate%20Meyer'

# r = requests.get(url = author)

# data = r.json()
# for i in range(len(data['results'])):
#     print(data['results'][i]['display_name'])
# pdb.set_trace()

def findPaper():
    paper = 'https://api.openalex.org/works?filter=doi:10.1086/724383'
    r = requests.get(url = paper)
    data = r.json()
    for i in range(len(data['results'][0]['authorships'])):
        authorName = data['results'][0]['authorships'][i]['author']['display_name']
        authorID = data['results'][0]['authorships'][i]['author']['id']
        print(authorName, authorID)
    return data

def findKate():
    author = 'https://api.openalex.org/authors/A5029009134'
    r = requests.get(url = author)
    data = r.json()
    
    # for i in range(len(data['results'])):
    #     string = ""
    #     string += data['results'][i]['display_name']
    #     for j in range(len(data['results'][i]['topics'])):
    #         string += ", "
    #         string += data['results'][i]['topics'][j]['display_name']
    #     print(string)
    #     print()
    return data



def getWorksFromJournal():
    papers = {}
    for page in range(1,3):
        try:
            query = f"https://api.openalex.org/works?filter=locations.source.id:S2764899347,publication_year:2020&select=id,title,authorships&per_page=200&page={page}"
            r = requests.get(url = query)
            data = r.json()
            # pdb.set_trace()
            for paper in data['results']:
                # pdb.set_trace()
                authorIDs = []
                for author in paper['authorships']:
                    authorIDs.append(author['author']['id'].split("/")[-1])
                    # print("here")
                papers[paper['title']] = authorIDs
            time.sleep(1)
        except:
            break
    return papers

def generateData(seedID = 'A5029009134'):
    '''
    starts with a seed ID and gets all papers and their co-authors, then all co-author's co-aturhos.
    returns a dictionary with the paper name as key and authorIDs as values
    '''
    papersRound1 = getWorks(seedID)
    papersRound2 = {}
    for authorIDs in papersRound1.values():
        for authorID in authorIDs:
            papersRound2 = getWorks(authorID) | papersRound2
    papersCombined = papersRound2 | papersRound1
    # pdb.set_trace()
    with open('round1.json', 'w') as f:
        json.dump(papersRound1, f)
    with open('round2.json', 'w') as f:
        json.dump(papersRound2, f)
    with open('papers.json', 'w') as f:
        json.dump(papersCombined, f)

"""