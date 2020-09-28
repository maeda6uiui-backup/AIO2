import argparse
import collections
import glob
import json
import logging
import os
from tqdm import tqdm

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def count_nqis(input_dir,save_filepath):
    pathname=os.path.join(input_dir,"*.*")
    files=glob.glob(pathname)

    words=[]

    for file in tqdm(files):
        with open(file,"r",encoding="utf_8") as r:
            lines=r.read().splitlines()

        for i in range(1,len(lines)):
            word=lines[i].split("\t")[0]
            words.append(word)

    counter=collections.Counter(words)
    with open(save_filepath,"w",encoding="utf_8",newline="") as w:
        for tup in counter.most_common():
            w.write(tup[0])
            w.write("\t")
            w.write(str(tup[1]))
            w.write("\n")

def main(input_dir,save_filepath):
    count_nqis(input_dir,save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_dir",type=str)
    parser.add_argument("--save_filepath",type=str)
    
    args=parser.parse_args()

    main(args.input_dir,args.save_filepath)
 