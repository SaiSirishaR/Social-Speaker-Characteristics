import os, sys
FALCON_DIR= '/home/srallaba/tools/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re

tdd_file  = sys.argv[1] #fnames
vox_dir = sys.argv[2] #main directory
feats_dir = vox_dir + '/festival/falcon_lspec'
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
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    lspec_fname = feats_dir + '/' + fname + '.feats'
    np.save(lspec_fname, spectrogram.T, allow_pickle=False)
