##### Mel spec extraction #######

import librosa
import os, numpy
import numpy as np
from dict_prep_1 import *
from network_1 import * 
import hparams
from scipy import signal

use_cuda = torch.cuda.is_available()

rate = 16000


from hparams import *

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _istft(spec):
 ispec = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
 return ispec

def _stft(data1):
 spec = librosa.stft(data1, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
 return spec


def denormalize(S):
  return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def inv_preemphasis(x):
  return signal.lfilter([1], [1, -hparams.preemphasis], x)


def inv_spectrogram(spectrogram):
  '''Converts spectrogram to waveform using librosa'''
  S = _db_to_amp(denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
  return inv_preemphasis(_griffin_lim(S ** hparams.power))


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
  threshold = librosa.db_to_amplitude(threshold_db)
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
###     print("inp seq is", inp_seq, numpy.shape(inp_seq))

     if use_cuda:
        inp_seq = inp_seq.cuda()

    # Greedy decoding
     mel_outputs, linear_outputs, alignments = model(inp_seq)

     linear_output = linear_outputs[0].cpu().data.numpy()
###     print("linesr out put is", numpy.shape(linear_output), "mel output", numpy.shape(mel_outputs))
    # Predicted audio signal
#    waveform = audio.inv_spectrogram(linear_output.T)

#    return waveform, alignment, spectrogram
     return linear_output


checkpoint_path = '/home/srallaba/projects/personality_stuff/voices/cmu_us_slt_12Dec/scripts/checkpoints/checkpoint_step42000.pth'

test_file_path ='/home/srallaba/projects/personality_stuff/voices/cmu_us_slt_12Dec/scripts/test_file.txt' 


dict = word_ids(test_file_path)
input_array = []
test_file_path= open(test_file_path).readlines()
for line in test_file_path:
  line=line.strip()
  wordseq = line.split('\n')[0]
  words = wordseq.split(' ')
  input_array.append([dict[word] for word in words])



path = '/home/srallaba/projects/personality_stuff/voices/cmu_us_slt_12Dec/'
text_path= path + 'ehmm/etc/new_txt.phseq.data'


tseq = word_ids(text_path)

##model = Tacotron(dict_length= int(len(tseq))+1, embeds=embedding_dim, mel_dim= mels, linear_dim=spec_len)

model = Tacotron(n_vocab= int(len(tseq))+1, embedding_dim=embedding_dim, mel_dim= mels, linear_dim=spec_len, r=hparams.outputs_per_step, padding_idx=hparams.padding_idx, 
use_memory_mask=hparams.use_memory_mask)

checkpoint = torch.load(checkpoint_path)
print("loaded checkpoint")
model.load_state_dict(checkpoint["state_dict"])
model.decoder.max_decoder_steps = max_decoder_steps
print("timstmap is", checkpoint["global_step"])
print("input_array is", input_array)
for i in range(0,len(input_array)):

  print("input is", np.array(input_array[i]))
  linear_output= test(model, np.array(input_array[i]))

  print("linear out shape is",numpy.shape(linear_output))
#  linear_output= linear_output.detach().numpy()
  linear_output = denormalize(linear_output)

    # Predicted audio signal
  wav = inv_spectrogram(linear_output.T)

  #wav = _griffin_lim(numpy.transpose(waveform))
  wav = wav[:find_endpoint(wav)]
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))

  librosa.output.write_wav('/home/srallaba/projects/personality_stuff/voices/cmu_us_slt_12Dec/scripts/resynth_wav/' + str(i) + '.wav', wav, rate)

