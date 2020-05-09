import os, sys
FALCON_DIR=os.environ['FALCONDIR']
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
import json

'''Syntax
python3.5 $FALCONDIR/dataprep_addtext.py etc/tdd .
'''

phseq_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_phones'
assure_path_exists(feats_dir)

ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['>']
ids_dict['UNK']
ids_dict['<']

f = open(phseq_file)
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split()[0]
    phones = '< ' + ' '.join(k for k in line.split()[1:]) + ' >'
    phones_ints = ' '.join(str(ids_dict[k]) for k in phones.split())

    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(phones + '\n')
    g.close()

