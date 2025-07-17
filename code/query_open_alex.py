import requests
import json
import pdb
import time
def query(query_url):
    cursor = '*'
    results = []
    page = 1
    while cursor:
        paged_url = f'{query_url}&per-page=200&cursor={cursor}'
        r = requests.get(url = paged_url)
        data = r.json()
        results.extend(data['results'])
        cursor = data['meta']['next_cursor']
        time.sleep(0.1)
        if page == 1:
            total_pages = 2 + data['meta']['count']//200
        print(f'on page {page} of {total_pages}')
        if page % 5 == 0:

            yield results
            results = []     
        page += 1
    yield results

def getWorks(authorID):
    '''
    This function has no return statement but can easily be formatted to return all the papers or co-authors for a particular author
    '''
    works = f'https://api.openalex.org/works?filter=author.id:{authorID}'
    r = requests.get(url = works)
    data = r.json()
    # pdb.set_trace()
    papers = dict()
    for paper in range(len(data['results'])):
        authorIDs = []
        for author in range(len(data['results'][paper]['authorships'])):
            authorIDs.append(data['results'][paper]['authorships'][author]['author']['id'].split("/")[-1])
        # pdb.set_trace()
        papers[data['results'][paper]['title']] = authorIDs
    return papers

def papers_by_author(seedID = 'A5029009134', name = 'Kate_Meyer'):
    '''
    starts with a seed ID and gets all papers and their co-authors, then all co-author's co-aturhos.
    returns a dictionary with the paper name as key and authorIDs as values
    '''
    round1_papers = getWorks(seedID)
    round1_authors = {seedID}
    for author_ids in round1_papers.values():
        round1_authors = round1_authors.union(set(author_ids))
    round2_papers = {}
    for author_id in round1_authors:
        round2_papers = getWorks(author_id) | round2_papers
    combined_papers = round2_papers | round1_papers
    # pdb.set_trace()
    with open(f'../data/people/{name}_round1.json', 'w') as f:
        json.dump(round1_papers, f)
    with open(f'../data/people/{name}_round2.json', 'w') as f:
        json.dump(round2_papers, f)
    with open(f'../data/people/{name}.json', 'w') as f:
        json.dump(combined_papers, f)

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



if __name__ == '__main__':

    output = query('https://api.openalex.org/works?filter=authorships.author.id:a5043805916,authorships.author.id:a5051868975')
    # results = query('https://api.openalex.org/works?filter=default.search:two%252520toed%252520sloth%7CCholoepus')
    for results in output:
        paper_dict = format_papers(results)


# Old code: 
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