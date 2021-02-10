import numpy as np
import os
from os.path import join as pjoin
import json
import sys
from common import *


def get_phone_feats(phone_feats_dir='aishell_phone_feats') -> dict:
    ret = {}
    with os.scandir(phone_feats_dir) as it:
        for entry in it:
            if entry.is_file():
                ssb = entry.name.split('.')[0]
                utt = ssb2utt.get(ssb)
                if utt is None:
                    continue

                feats = np.loadtxt(pjoin(phone_feats_dir, f'{ssb}.wav.txt'))
                ret[utt] = feats

    return ret
