import json
from query_open_alex import *
with open(f'../data/people/Kate_Meyer_round1.json', 'r') as f:
    round1_papers = json.load(f)

round1_authors = set()
for author_ids in round1_papers.values():
    round1_authors = round1_authors.union(set(author_ids))
# pdb.set_trace()
round2_papers = {}
for author_id in round1_authors:
    round2_papers = getWorks(author_id) | round2_papers
combined_papers = round2_papers | round1_papers

with open(f'../data/people/Kate_Meyer_round2.json', 'w') as f:
    json.dump(round2_papers, f)
with open(f'../data/people/Key_Meyer.json', 'w') as f:
    json.dump(combined_papers, f)