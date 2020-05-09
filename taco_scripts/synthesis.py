
##### Mel spec extraction #######

import librosa
import os, numpy
import numpy as np

n_fft = (1025 - 1) * 2
hop_length = int(12.5 / 1000 * 16000)
win_length = int(50 / 1000 * 16000)
griffin_lim_iters = 60
num_freq = 1025
num_mels=80



def _istft(spec):
 ispec = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
 return ispec

def _stft(data1):
 spec = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
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


folder = '/home/srallaba/projects/personality_stuff/voices/cmu_us_LJspeech/Speech_Expts_Barebones/taco_25October_expts/Data/cmu_us_bdl_arctic/wav'
feats = '/home/srallaba/projects/personality_stuff/voices/cmu_us_LJspeech/Speech_Expts_Barebones/taco_25October_expts/Data/cmu_us_bdl_arctic/feats'
files = sorted(os.listdir(folder))
print("the wavefile are:", files)

for file in files:
 print("Processing file......", file)
 data, rate = librosa.load(folder+ '/' + file, sr=None) #To preserve the native sampling rate of the file, use sr=None.
 spec = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
 spectrogram = librosa.amplitude_to_db(np.abs(spec))
 np.savetxt(feats+'/' + 'spec'+'/'+file.split('.')[0]+'.spec',numpy.transpose(spectrogram))
 mel_spec = librosa.amplitude_to_db(_linear_to_mel(np.abs(spec)))
 np.savetxt(feats+'/' + 'mel_spec'+'/'+file.split('.')[0]+'.mel',numpy.transpose(mel_spec))
#wav = _griffin_lim(librosa.amplitude_to_db(np.abs(spec)))
#wav = wav[:find_endpoint(wav)]
#wav *= 32767 / max(0.01, np.max(np.abs(wav)))

