import os, sys
FALCON_DIR= '/path/to/falcondirectory/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_mspec'
wav_dir = vox_dir + '/wav'
assure_path_exists(feats_dir)


f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:

    ctr += 1
    line = line.split('\n')[0]
    fname = line.split()[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split()[0]
    wav_fname = wav_dir + '/' + fname + '.wav'
    wav = audio.load_wav(wav_fname)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mspec_fname = feats_dir + '/' + fname + '.feats'
    np.save(mspec_fname, mel_spectrogram.T, allow_pickle=False)
