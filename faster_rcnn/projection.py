import argparse
import numpy as np
import os
import random
import torch
from tqdm import tqdm

#Fix the seed.
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(im_embeddings_dir,save_dir):
    os.makedirs(save_dir,exist_ok=True)

    fc=torch.nn.Linear(256,768).to(device)

    files=os.listdir(im_embeddings_dir)
    for file in tqdm(files):
        orig_filepath=os.path.join(im_embeddings_dir,file)
        im_embeddings=torch.load(orig_filepath,map_location=device).to(device)
        im_embeddings=fc(im_embeddings)

        save_filepath=os.path.join(save_dir,file)
        torch.save(im_embeddings,save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--im_embeddings_dir",type=str,default="~/FasterRCNNEmbeddings/Original")
    parser.add_argument("--save_dir",type=str,default="~/FasterRCNNEmbeddings/Projected")
    args=parser.parse_args()

    main(args.im_embeddings_dir,args.save_dir)
