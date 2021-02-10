import numpy as np
import os
from os.path import join as pjoin
import json
import sys
from common import *
from phone_feats import get_phone_feats


if __name__ == "__main__":
    utt2trans = get_utt2trans()
    pid2scores = {i: [] for i in range(201 + 1)}
    
    phone_feats = get_phone_feats()

    with os.scandir('aishell_phone_feats') as it:
        for entry in it:
            if entry.is_file():
                ssb = entry.name.split('.')[0]
                utt = ssb2utt.get(ssb)
                if utt is None:
                    continue

                if utt not in utt2trans:
                    print(f'WARNING: no annotation for {utt}')
                    continue

                trans = utt2trans[utt]
                feats = np.loadtxt(pjoin('aishell_phone_feats', f'{ssb}.wav.txt'))
                if len(trans) != feats.shape[0]:
                    print(f'WARNING: number of phones of {utt} mismatch')
                    continue

                print(f'{ssb}/{utt}')
                for i, t in enumerate(trans):
                    idx = phone2id[t] - 1
                    pid2scores[idx + 1].append(feats[i, idx])

    os.makedirs('feats', exist_ok=True)
    for pid, scores in pid2scores.items():
        if len(scores) > 0:
            print(f'{pid}: {len(scores)} samples')
            np.savetxt(pjoin('feats', f'{pid}.txt'), scores)