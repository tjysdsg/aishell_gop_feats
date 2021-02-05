import torch
from os.path import abspath, dirname
from typing import List
import librosa
import os
import sys
import numpy as np
from multiprocessing import Process


data_dir = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA'
file_dir = dirname(abspath(__file__))
call_root_dir = '/mingback/students/tjy/call'
sys.path.insert(0, call_root_dir)
from gop_server import zh_config
from gop_server.zh_gop import zh_gop_main
from gop_server.preprocess import float2pcm

# prepare output dir
output_dir = os.path.join(file_dir, 'aishell_phone_feats')
os.makedirs(output_dir, exist_ok=True)

# init
annotation_path = 'annotations.txt'
wavs = []
trans = []
with open(annotation_path, 'r') as f:
    for line in f:
        tokens = line.replace('\n', '').split()
        wav = tokens[0]

        out_path = os.path.join(output_dir, f'{wav}.txt')
        if os.path.exists(out_path):
            print(f'WARNING: Skipping {wav} since it\'s already been processed')
            continue

        transcript = ''
        for t in tokens[1::2]:
            transcript += t
        trans.append(transcript)
        wavs.append(wav)


def worker(data: List[str]):
    file = data[0]
    trans = data[1]
    sys.stdout.write("\033[K")
    print('Computing GOP for {}'.format(file), end='\r')
    file_path = os.path.join(data_dir, file[:7], file)  # SSBabcd/SSBabcdxxxx.wav
    out_path = os.path.join(output_dir, f'{file}.txt')
    try:
        y, _ = librosa.load(file_path, sr=16000, mono=True)
        y = float2pcm(y)
        res = zh_gop_main(y, trans, return_phone_feats=True)
        np.savetxt(out_path, res['phone_feats'])
    except Exception as e:
        sys.stdout.write("\033[K")
        print(f'Compute gop for {file} failed')
        print(str(e))
        return None


def main():
    nj = 16
    n = len(trans)
    
    all_data = list(zip(wavs, trans))
    np.random.shuffle(all_data)

    # calculate phone feats concurrently
    i = 0
    while i < n:
        data = all_data[i: i + nj]
        ps = [Process(target=worker, args=(data[j],)) for j in range(nj) if len(data) > j]
        for p in ps:
            p.start()

        for j, p in enumerate(ps):
            p.join()
            sys.stdout.write("\033[K")
            print(f'Batch completed: {j + 1}/{nj}', end='\r')

        sys.stdout.write("\033[K")
        print(f"=========== Progress: {i}/{n} ============", end='\r')
        i += nj


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
