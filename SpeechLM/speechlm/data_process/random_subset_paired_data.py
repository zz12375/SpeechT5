# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import os
import argparse
from tqdm import tqdm
import numpy as np


lg_label = "__label__{}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--output-num", "-k", default=400000, type=int)
    parser.add_argument("--total-num", "-n", default=40373539, type=int)
    parser.add_argument("--seed", "-s", default=123, type=int)
    args = parser.parse_args()
    
    src, tgt = args.input.rsplit('.', 1)[-1].split('-')
    np.random.seed(args.seed)
    select_indices = np.random.choice(args.total_num, size=args.output_num, replace=False)
    select_indices = list(select_indices)
    select_indices.sort(reverse=True)

    src_f = open(f"{args.output}.{src}", "w")
    tgt_f = open(f"{args.output}.{tgt}", "w")
    with open(f"{args.input}.{src}", 'r') as f1, open(f"{args.input}.{tgt}", 'r') as f2: 
        for i, (src_line, tgt_line) in tqdm(enumerate(zip(f1, f2))):
            if len(select_indices) == 0:
                break
            if i == select_indices[-1]:
                src_f.write(src_line)
                tgt_f.write(tgt_line)
                select_indices.pop()

if __name__ == "__main__":
    main()



