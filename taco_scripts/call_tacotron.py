############# Adding collate function to data loader ( Version 2 ) ############

import torch 
from torch.utils.data import Dataset, DataLoader
import os
import numpy
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from dict_prep_2 import *
import hparams
from network_1 import *
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
from torch import optim
import numpy
import numpy as np
from os.path import  join

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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
use_cuda = torch.cuda.is_available()
print("use cuda is", use_cuda)
if use_cuda:
    cudnn.benchmark = False


#### Creating a class for data loading and retriving it using indices #####

global_step = 0
global_epoch = 0

spec_filenames = []
mel_spec_filenames = []


def pad(text, max_len):
    return np.pad(text, (0, max_len - len(text)),
                  mode='constant', constant_values=0)


def pad_spec(spec, max_len):
    spec = np.pad(spec, [(0, max_len - len(spec)), (0, 0)],
               mode="constant", constant_values=0)
    return spec


class arctic_data(Dataset):


      def __init__(self, txt_input, spec_dir, mel_spec_dir):

       self.txt_input = txt_input
       self.spec_dir = spec_dir
       self.mel_spec_dir = mel_spec_dir


      def __len__(self,):

        return len(self.txt_input)


      def __getitem__(self,idx):

#       print("index is", idx)

       return self.txt_input[idx], self.spec_dir[idx], self.mel_spec_dir[idx] 

def collate_fn(batched_inp):

        r = hparams.outputs_per_step
        input_lengths = [len(texts[0]) for texts in batched_inp]


        max_input_length = np.max(input_lengths)

        ###### This step[ helps while decoding where we dive the number of frames by r and reshape the tensor
        max_spec_length = np.max([len(spec[1]) for spec in batched_inp]) + 1
        if max_spec_length % r != 0:
         max_spec_length += r - max_spec_length % r
         assert max_spec_length % r == 0


        a = np.array([pad(texts[0], max_input_length) for texts in batched_inp], dtype=np.int)
        text_input = torch.LongTensor(a)
        input_lengths = torch.LongTensor(input_lengths)


        b = np.array([pad_spec(spec[1], max_spec_length) for spec in batched_inp], dtype=np.float32)
        spec_input = torch.FloatTensor(b)


        c = np.array([pad_spec(mel_spec[2], max_spec_length) for mel_spec in batched_inp], dtype=np.float32)
        mspec_input = torch.FloatTensor(c)

        return text_input, spec_input, mspec_input, input_lengths


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{}.pth".format(global_step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)



def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))




def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
#    idx = min(1, len(input_lengths) - 1)
#    input_length = input_lengths[idx]
    idx=0
    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(global_step))
    # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment)



def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr

##### Paths to Dataset directories #######

path = '/path/to/scripts/'
text_path= path + 'path/to/text/'
mel_path= path + 'path/to/melspectrograms/'
spec_path = path + 'path/to/lspec/'
checkpoint_dir='/path/to/checkpointsdirectory/'
filenames = path +'fnames.train'

fnames_array = []

fg = open(filenames)
for name in fg:
  fnames_array.append(name.split('\n')[0])

######### Data Loader Module ############

dict = word_ids(text_path)
input_array = []
text_file= open(text_path).readlines()
for line in text_file:
  line=line.strip()
  wordseq = line.split('\n')[0]
  words = wordseq.split(' ')
  input_array.append([dict[word] for word in words])


spec_files = sorted(os.listdir(spec_path))
mel_files = sorted(os.listdir(mel_path))

for num, spec_file in enumerate(spec_files):
  if spec_file.split('.')[0] in fnames_array:
       spec_filenames.append(numpy.load(os.path.join(spec_path,spec_file)))
for num, mel_spec_file in enumerate(mel_files):
  if mel_spec_file.split('.')[0] in fnames_array:
       mel_spec_filenames.append(numpy.load(os.path.join(mel_path,mel_spec_file)))

exp_data = arctic_data(txt_input = input_array, spec_dir=spec_filenames, mel_spec_dir=mel_spec_filenames) #### Data paths go here
data = DataLoader(exp_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

########### Model initialization ########

print("dict length is", len(dict),dict)
model = Tacotron(n_vocab= int(len(dict))+1, embedding_dim=256, mel_dim= hparams.mels, linear_dim=hparams.spec_len, r=hparams.outputs_per_step, padding_idx=hparams.padding_idx, use_memory_mask=hparams.use_memory_mask)
#model = Tacotron(dict_length= int(len(dict))+1, embeds=embedding_dim, mel_dim= mels, linear_dim=spec_len)
if use_cuda:
  model = model.cuda()

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)



# Load checkpoint
checkpoint_path ='/path/to/specificcheckpoint/' 

print("Load checkpoint from: {}".format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["state_dict"])
optimizer.load_state_dict(checkpoint["optimizer"])
try:
  global_step = checkpoint["global_step"]
  global_epoch = checkpoint["global_epoch"]
except:
  # TODO
  pass


##### Batched and padded datasets #######
global global_step, global_epoch

with torch.autograd.set_detect_anomaly(True):


 while global_epoch < hparams.nepochs:
  running_loss = 0.

  for num, (text, spec, mel, lengths) in enumerate(data): 

#   print("text is", text, "lengths are:", lengths) #, "spec is", spec, "mel is", mel, lengths)
#   print("length of dict is", int(len(dict)), "length of data is", len(data))

#   print("batch", num)
   sorted_lengths, indices = torch.sort(
                lengths.view(-1), dim=0, descending=True)
   sorted_lengths = sorted_lengths.long().numpy()

#   print("lengths are", lengths, "sorted lemnghts are", sorted_lengths)
   text, mel, spec = text[indices], mel[indices], spec[indices]
#   print("indices are", indices)

   current_lr = _learning_rate_decay(hparams.initial_learning_rate, global_step)
   for param_group in optimizer.param_groups:
      param_group['lr'] = current_lr

   optimizer.zero_grad()


####### Feeding data to Network ######

#   print("input data is", numpy.shape(spec), spec, "text shape is", text)
   text,spec, mel = Variable(text), Variable(spec), Variable(mel)
###   print("input data", numpy.shape(text), text, numpy.shape(mel), mel)

   if use_cuda:
     text,spec, mel = text.cuda(), spec.cuda(), mel.cuda()
     mel_outputs, linear_outputs, attn = model(text, mel, input_lengths=sorted_lengths)

#     print("griffin lim on", numpy.shape(linear_outputs[0].cpu().data.numpy()))
  
#     print("mel shape is", numpy.shape(mel_outputs))
     mel_loss = criterion(mel_outputs, mel)
#     print("mel loss is", mel_loss)
     n_priority_freq = int(3000 / (hparams.sample_rate * 0.5) * hparams.num_freq)

#     print("linear out shape is", numpy.shape(spec), "with priprty", numpy.shape(spec[:, :, :n_priority_freq]))

     linear_loss = 0.5 * criterion(linear_outputs, spec) + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq], spec[:, :, :n_priority_freq])
     loss = mel_loss + linear_loss
#     print("loss is", loss)
#     loss = loss.clone()
     if global_step > 0 and global_step % hparams.checkpoint_interval == 0:


        save_states(global_step, mel_outputs, linear_outputs, attn, spec,
                    None, checkpoint_dir)

        save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)



     start_time = time.time()

            # Calculate gradients
     loss.backward()

            # clipping gradients
     grad_norm = nn.utils.clip_grad_norm(model.parameters(), 1.)

            # Update weights
     optimizer.step()
     global_step += 1


     time_per_step = time.time() - start_time
     if global_step%100 ==1:
       print("time per step is", time_per_step)
     running_loss += loss.item()


  averaged_loss = running_loss / (len(data))
  print("epoch", global_epoch, "Loss: {}".format(running_loss / (len(data))))
  global_epoch += 1


