import numpy as np
import os
from os.path import join as pjoin
import json
import sys
from common import *
from phone_feats import get_phone_feats


if __name__ == "__main__":
    utt2trans = {}
    annotation_path = os.path.join('audio-l2-standard', 'pinyin.txt')
    with open(annotation_path, 'r') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0]
            utt2trans[utt] = tokens[1:]

    pid2scores = {i: [] for i in range(201 + 1)}

    phone_feats = {}
    phone_feats_dir = 'l2_phone_feats'
    with os.scandir(phone_feats_dir) as it:
        for entry in it:
            if entry.is_file():
                utt = entry.name.split('.')[0]
                trans = utt2trans[utt]
                feats = np.loadtxt(pjoin(phone_feats_dir, f'{utt}.txt'))
                phone_feats[utt] = feats

                if len(trans) != feats.shape[0]:
                    print(f'WARNING: number of phones of {utt} mismatch')
                    continue

                for i, t in enumerate(trans):
                    t = t.replace('_', '')
                    idx = phone2id[t] - 1
                    pid2scores[idx + 1].append(feats[i, idx])

    for pid, scores in pid2scores.items():
        if len(scores) > 0:
            print(f'{pid}: {len(scores)} samples, {np.quantile(scores, 0.25)} {np.quantile(scores, 0.75)}')