import bs4 as bs
import os
import codecs
from tqdm import tqdm

fold = "pbio/"
title_dict = {}

for i in tqdm(range(len(os.listdir(fold))), desc='Creating Article Title Dict'):
    path = os.listdir(fold)[i]
    file = codecs.open(fold+path, 'r', encoding="utf-8")
    file = bs.BeautifulSoup(file, "xml")
    # build dict of article title to path
    title_dict[file.find('article-title').get_text()] = path

for i in tqdm(range(len(os.listdir(fold))), desc='Finding Sources in Scope'):
    path = os.listdir(fold)[i]
    file = codecs.open(fold+path, 'r', encoding="utf-8")
    file = bs.BeautifulSoup(file, "xml")
    for ref in file.find_all("ref"):
        try:
            title_dict[ref]
            print(ref)
        except:
            continue

print(title_dict)


'''
        for ref in file.find_all("ref"):
        if ref.get('id') not in os.listdir(fold):
            print("not in plos" + ref.get('id'))
'''

