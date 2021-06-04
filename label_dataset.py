import bs4 as bs
import os
import codecs
from tqdm import tqdm
import json
import re


def main():
    articles = "articles_plos/"
    sources = "sources_plos/"

    article_title_dict = name_to_path(articles)
    source_title_dict = name_to_path(sources, article_title_dict)

    docs = []

    for i in tqdm(range(len(article_title_dict)), desc="Generating JSON"):
        article = list(article_title_dict.items())[i][0]
        path = list(article_title_dict.items())[i][1]
        srcs = []
        src_dict = dict()  # maps paper's labels to our own ids
        content = []
        file = codecs.open(path, 'r', encoding="utf-8")
        file = bs.BeautifulSoup(file, "xml")
        if not file.find('xref'):
            continue
        if "correction" in file.find('article-title').get_text().lower():
            continue
        src_index = 0
        for citation in file.find_all("ref"):
            try:
                # add to sources if in scope
                title = citation.find("article-title").get_text()
                src_path = source_title_dict[title]
                src = {"id": src_index,
                       "title": title,
                       "path": src_path,
                       "text": src_text(src_path),
                       "summary": None}
                srcs.append(src)
                src_dict[citation.find("label").get_text()] = src_index
                src_index += 1
            except:
                continue
        if len(srcs) < 1:
            continue
        cont_index = 0
        for para in file.find_all("p"):
            par = {"id": cont_index,
                   "text": para.get_text(),
                   "cited": False,
                   "citations": []}
            for ref in para.find_all("xref"):
                try:
                    ref_index = src_dict[ref.get_text()]
                    par["citations"].append(ref_index)
                except:
                    continue
            brefs = re.findall(r"\[(\d+)\]", para.get_text())
            for bref in brefs:
                try:
                    ref_index = src_dict[bref]
                    par["citations"].append(ref_index)
                except:
                    continue
            if len(par["citations"]) > 0:
                par["cited"] = True
            content.append(par)
            par["citations"] = list(dict.fromkeys(par["citations"]))
            cont_index += 1
        doc = {"title": article,
               "path": path,
               "sources": srcs,
               "content": content,
               "loc_labels": None,
               "cite_labels": None}
        docs.append(doc)

    with open("data.json", "w+") as outfile:
        json.dump(docs, outfile)


def name_to_path(root_path, title_dict={}):
    for i in tqdm(range(len(os.listdir(root_path))), desc=f'Creating Dict from {root_path}'):
        path = os.listdir(root_path)[i]
        file = codecs.open(root_path + path, 'r', encoding="utf-8")
        file = bs.BeautifulSoup(file, "xml")
        # build dict of article title to path
        title_dict[file.find('article-title').get_text()] = root_path + path

    return title_dict


def src_text(path):
    text = []
    file = codecs.open(path, 'r', encoding="utf-8")
    file = bs.BeautifulSoup(file, "xml")
    for para in file.find_all("p"):
        text.append(para.get_text())
    return text


if __name__ == "__main__":
    main()
