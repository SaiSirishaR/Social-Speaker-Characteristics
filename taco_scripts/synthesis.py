##### Mel spec extraction #######

import os, sys
FALCON_DIR= '/path/to/falcon/directory'
sys.path.append(FALCON_DIR)
import librosa
import os, numpy
import numpy as np
from dict_prep_2_1 import *
from network_1 import * 
import hparams
from utils import audio
import re
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')

 
use_cuda = torch.cuda.is_available()

rate = 16000


from hparams import *


def _istft(spec):
 ispec = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
 return ispec

def _stft(data1):
 spec = librosa.stft(data1, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
 return spec

def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
  window_length = int(16000 * min_silence_sec)
  hop_length = int(window_length / 4)
  threshold = (threshold_db)
  for x in range(hop_length, len(wav) - window_length, hop_length):
    if np.max(wav[x:x+window_length]) < threshold:
      return x + hop_length
  return len(wav)

_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(rate, n_fft, n_mels=num_mels)


max_decoder_steps = 500 # default value

def test(model, inp_text):

     if use_cuda:
        model = model.cuda()

#     model.decoder.eval()
     model.encoder.eval()
     model.postnet.eval()      

     inp_seq = Variable(torch.from_numpy(inp_text)).unsqueeze(0)
     print("inp seq is", inp_seq, numpy.shape(inp_seq))

     if use_cuda:
        inp_seq = inp_seq.cuda()

    # Greedy decoding
     mel_outputs, linear_outputs, alignments = model(inp_seq)

     linear_output = linear_outputs[0].cpu().data.numpy()
     alignment = alignments[0].cpu().data.numpy()

     print("linesr out put is", numpy.shape(linear_output), "mel output", numpy.shape(mel_outputs))
    # Predicted audio signal
#    waveform = audio.inv_spectrogram(linear_output.T)

#    return waveform, alignment, spectrogram
     return linear_output, alignment

checkpoint_path = '/home/srallaba/projects/personality_stuff/voices/arabic_data/cmu_us_arabic/scripts/checkpoints/'
checkpoint=sys.argv[1]
step = checkpoint.split('.')[0].split('_')[-1]
print("step is", step)


test_file_path ='/pth/to/testfiles/' 



path = '/path/to/database/'
text_path= path + 'ehmm/etc/new_txt.phseq.data'

######### Data Loader Module ############

dict = word_ids(text_path)


#dict = word_ids(test_file_path)
input_array = []
test_file_path= open(test_file_path).readlines()
for line in test_file_path:
  print("line is", line)
  line=line.strip()
  wordseq = line.split('\n')[0]
  words = wordseq.split(' ')
  input_array.append([dict[word] for word in words])




#tseq = word_ids(text_path)

model = Tacotron(n_vocab= int(len(dict))+1, embedding_dim=embedding_dim, mel_dim= mels, linear_dim=spec_len, r=hparams.outputs_per_step, padding_idx=hparams.padding_idx, use_memory_mask=hparams.use_memory_mask)


checkpoint = torch.load(checkpoint_path + checkpoint)
#print("loaded checkpoint", checkpoint)
model.load_state_dict(checkpoint["state_dict"])
model.decoder.max_decoder_steps = max_decoder_steps

print("input_array is", input_array)
for i in range(0,len(input_array)):
  print("processing file no:", i)
  linear_output, alignment = test(model, np.array(input_array[i]))
  waveform = audio.inv_spectrogram(linear_output.T)
  dst_wav_path = join('/path/to/resynthdierctortory/', "{}_{}.wav".format(str(i),step))
  audio.save_wav(waveform, dst_wav_path)

  dst_alignment_path = join('/path/to/resynthdierctortory/', "{}_{}_alignment.png".format(str(i), step))

  plot_alignment(alignment.T, dst_alignment_path,
                           info="tacotron, {}".format(step))
  
