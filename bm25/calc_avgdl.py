import argparse
import glob
import logging
import os

def main(input_dir):
    pathname=os.path.join(input_dir,"*.*")
    files=glob.glob(pathname)

    total_d=0
    for file in files:
        with open(file,"r") as r:
            d=int(r.readline())
            total_d+=d

    avgdl=total_d/len(files)
    print(avgdl)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_dir",type=str)
    
    args=parser.parse_args()

    main(args.input_dir)
