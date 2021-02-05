import numpy as np
import os
from os.path import join as pjoin
import json
import sys
from common import *


def get_utt2trans():
    # utt of filtered wavs
    filtered_wavs = set()
    with open('wav_filtered.scp') as f:
        for line in f:
            utt = line.split('\t')[0]
            filtered_wavs.add(utt)

    # utt to its phone-level transcript
    utt2trans = {}
    with open('annotations.txt') as f:
        for line in f:
            tokens = line.replace('\n', '').split()
            utt = tokens[0].split('.')[0]
            utt = ssb2utt.get(utt)
            if utt is None or utt not in filtered_wavs:
                continue
            phones = tokens[2::2]
            phones = [p.replace('5', '0') for p in phones]  # 轻声 5 -> 0
            # separate initals and finals
            utt2trans[utt] = []
            for p in phones:
                if p[:2] in INITIALS:
                    utt2trans[utt].append(p[:2])
                    final = p[2:]
                elif p[:1] in INITIALS:
                    utt2trans[utt].append(p[:1])
                    final = p[1:]
                else:
                    final = p

                # 去掉儿化
                if 'er' not in final and len(final) >= 2 and final[-2] == 'r':
                    utt2trans[utt].append(final.replace('r', ''))
                    utt2trans[utt].append('er0')
                else:
                    utt2trans[utt].append(final)

    return utt2trans


if __name__ == "__main__":
    utt2trans = get_utt2trans()
    pid2scores = {i: [] for i in range(201 + 1)}

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