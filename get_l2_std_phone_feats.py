from os.path import abspath, dirname
from typing import List
import librosa
import os
import sys
import numpy as np
from multiprocessing import Process


data_dir = 'audio-l2-standard'
file_dir = dirname(abspath(__file__))
call_root_dir = '/mingback/students/tjy/call'
sys.path.insert(0, call_root_dir)
from gop_server import zh_config
from gop_server.zh_gop import zh_gop
from gop_server.preprocess import float2pcm

# prepare output dir
output_dir = os.path.join(file_dir, 'l2_phone_feats')
os.makedirs(output_dir, exist_ok=True)

# init
annotation_path = os.path.join(data_dir, 'pinyin.txt')
trans_path = os.path.join(data_dir, 'trans.txt')
utt2pinyin = {}
utt2trans = {}
with open(annotation_path, 'r') as f:
    for line in f:
        tokens = line.replace('\n', '').split()
        utt = tokens[0]

        out_path = os.path.join(output_dir, f'{utt}.txt')
        if os.path.exists(out_path):
            print(f'WARNING: Skipping {wav} since it\'s already been processed')
            continue

        utt2pinyin[utt] = tokens[1:]

with open(trans_path, 'r') as f:
    for line in f:
        tokens = line.replace('\n', '').split()
        utt = tokens[0]

        out_path = os.path.join(output_dir, f'{utt}.txt')
        if os.path.exists(out_path):
            print(f'WARNING: Skipping {wav} since it\'s already been processed')
            continue

        utt2trans[utt] = tokens[-1]


def worker(data: List[str]):
    utt, trans, pinyin = data
    sys.stdout.write("\033[K")
    print('Computing GOP for {}'.format(utt), end='\r')
    file_path = os.path.join(data_dir, f'{utt}.wav')
    out_path = os.path.join(output_dir, f'{utt}.txt')
    try:
        y, _ = librosa.load(file_path, sr=16000, mono=True)
        y = float2pcm(y)
        
        from gop_server.transcript import tokenize_chinese, normalize_transcript, remove_punctuation
        trans = normalize_transcript(trans)
        segments = tokenize_chinese(trans)
        trans = ' '.join(segments)
        trans = remove_punctuation(trans)

        res = zh_gop(y, trans, pinyin)
        np.savetxt(out_path, res['phone_feats'])
    except Exception as e:
        sys.stdout.write("\033[K")
        print(f'Compute gop for {file} failed')
        print(str(e))
        return None


def main():
    nj = 16

    all_data = [[utt, trans, utt2pinyin[utt]] for utt, trans in utt2trans.items()]
    n = len(all_data)
    np.random.shuffle(all_data)

    # calculate phone feats concurrently
    i = 0
    while i < n:
        worker(all_data[i])
        # data = all_data[i: i + nj]
        # ps = [Process(target=worker, args=(data[j],)) for j in range(nj) if len(data) > j]
        # for p in ps:
        #     p.start()
 
        # for j, p in enumerate(ps):
        #     p.join()
        #     sys.stdout.write("\033[K")
        #     print(f'Batch completed: {j + 1}/{nj}', end='\r')

        sys.stdout.write("\033[K")
        print(f"=========== Progress: {i}/{n} ============", end='\r')
        # i += nj
        i += 1


if __name__ == '__main__':
    main()
