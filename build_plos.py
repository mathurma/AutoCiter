import bs4 as bs
import os
import codecs

plos_fold = "plos/allofplos_xml/"

for path in os.listdir(plos_fold):
    file = codecs.open(plos_fold+path, 'r', encoding="utf-8")
    file = bs.BeautifulSoup(file, "xml")
    # build dict of label to ariticle title (maybe take note of author)
    for ref in file.find_all("ref"):
        print(ref)
        if ref.get('id') not in os.listdir(plos_fold):
            print("not in plos" + ref.get('id'))
    break