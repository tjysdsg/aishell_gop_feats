import numpy as np
import os
from os.path import join as pjoin
import json
import sys


# 声母
INITIALS = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh']

# SSB utterance id to aishell2 utterance id
ssb2utt = {}
with open('ssbutt.txt') as f:
    for line in f:
        utt, ssb = line.replace('\n', '').split('|')
        ssb2utt[ssb] = utt

phone2id = {}
pid2phone = {}
with open('phones.txt') as f:
    for line in f:
        tokens = line.replace('\n', '').split()
        name, pid = tokens
        name = name.replace('_', '')
        pid = int(pid)
        phone2id[name] = pid
        pid2phone[pid] = name


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