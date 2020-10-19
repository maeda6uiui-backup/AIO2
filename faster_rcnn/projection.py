import argparse
import logging
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

logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(im_embed_dir,im_embed_dimension,text_embed_dimension,save_dir):
    logger.info("im_embed_dir: {}".format(im_embed_dir))
    logger.info("im_embed_dimension: {}".format(im_embed_dimension))
    logger.info("text_embed_dimension: {}".format(text_embed_dimension))
    logger.info("save_dir: {}".format(save_dir))

    os.makedirs(save_dir,exist_ok=True)

    fc=torch.nn.Linear(im_embed_dimension,text_embed_dimension).to(device)

    files=os.listdir(im_embed_dir)
    for file in tqdm(files):
        orig_filepath=os.path.join(im_embed_dir,file)
        im_embed=torch.load(orig_filepath,map_location=device).to(device)
        im_embed=fc(im_embed)

        save_filepath=os.path.join(save_dir,file)
        torch.save(im_embed,save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--im_embed_dir",type=str,default="~/FasterRCNNEmbeddings/Original")
    parser.add_argument("--im_embed_dimension",type=int,default=1024)
    parser.add_argument("--text_embed_dimension",type=int,default=768)
    parser.add_argument("--save_dir",type=str,default="~/FasterRCNNEmbeddings/Projected")
    args=parser.parse_args()

    main(args.im_embed_dir,args.im_embed_dimension,args.text_embed_dimension,args.save_dir)
