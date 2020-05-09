import torch
import torch.nn as nn
import numpy



class Prenet(nn.Module):

  def __init__(self, embed_dim, sizes=[256,128]):

    super(Prenet, self).__init__()

    self.net = nn.Sequential(nn.Linear(sizes[:1][0], sizes[:1][0]),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(sizes[:1][0], sizes[1:][0]),
                             nn.ReLU(),
                             nn.Dropout(0.2))

  def forward(self, text_seq):
    print("in prenet")     
    prenet_out=self.net(text_seq)
    print("prenet outputs are:", numpy.shape(prenet_out))
    return prenet_out
    

class Encoder(nn.Module):

  def __init__(self, embed_dim):

   super(Encoder, self).__init__()
   self.prenet = Prenet(embed_dim, sizes=[256, 128])   

  def forward(self, input, input_lengths=None):
    
    inputs=self.prenet(input)







class Tacotron(nn.Module):

  def __init__(self, dict_length, embeds, mel_dim, linear_dim):
   super(Tacotron, self).__init__()
   self.dict_length = dict_length
   self.embeds = embeds
   self.mel_dim = mel_dim
   self.linear_dim = linear_dim
   self.embedding = nn.Embedding(dict_length, embeds)
   self.encoder = Encoder(embeds)

  def forward(self, text, spec=None, input_lengths=None):
    embedded = self.embedding(text)
    print("embedding dim are:", numpy.shape(embedded))
    encoded_vec = self.encoder(embedded, input_lengths)

