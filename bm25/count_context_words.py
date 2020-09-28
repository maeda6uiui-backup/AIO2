import argparse
import collections
import gzip
import json
import logging
import os
import MeCab
from tqdm import tqdm

import hashing

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_contexts(context_filepath):
    contexts={}

    with gzip.open(context_filepath,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts[title]=text

    return contexts

def count_context_words(mecab,title,context,save_dir):
    genkeis=[]

    node=mecab.parseToNode(context)
    while node:
        features=node.feature.split(",")

        hinsi=features[0]
        if hinsi=="BOS/EOS":
            node=node.next
            continue

        genkei=features[6]
        genkeis.append(genkei)

        node=node.next

    counter=collections.Counter(genkeis)

    title_hash=hashing.get_md5_hash(title)
    save_filepath=os.path.join(save_dir,title_hash+".txt")
    with open(save_filepath,"w",encoding="utf_8",newline="") as w:
        w.write(str(len(genkeis)))
        w.write("\n")

        for tup in counter.most_common():
            w.write(tup[0])
            w.write("\t")
            w.write(str(tup[1]))
            w.write("\n")

def main(context_filepath,save_dir):
    os.makedirs(save_dir,exist_ok=True)

    mecab=MeCab.Tagger()

    logger.info("Start loading contexts.")
    contexts=load_contexts(context_filepath)

    logger.info("Start counting words.")
    for title,context in tqdm(contexts.items()):
        count_context_words(mecab,title,context,save_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--context_filepath",type=str)
    parser.add_argument("--save_dir",type=str)
    
    args=parser.parse_args()

    main(args.context_filepath,args.save_dir)
