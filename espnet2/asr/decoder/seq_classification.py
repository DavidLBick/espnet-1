import torch
import torch.nn.functional as F
from typeguard import check_argument_types 

from espnet2.asr.decoder.abs_decoder import AbsDecoder 
from espnet2.utils.get_default_kwargs import get_default_kwargs 
from espnet.nets.pytorch_backend.nets_utils import to_device

class SeqClassifier(AbsDecoder):
  def __init__(
    self, 
    vocab_size, 
    encoder_output_size, 
    pool_type,
    attention_heads,
    embed_dim
  ):
    super().__init__()
    self.pool_type = pool_type
    if pool_type == 'att':
      self.query = nn.Embedding(encoder_output_size, embed_dim) 
      self.value = nn.Linear(encoder_output_size, embed_dim) 
      self.key = nn.Linear(encoder_output_size, embed_dim)
      self.mha = nn.MultiheadAttention(embed_dim, attention_heads)

    self.logistic = nn.Linear(encoder_output_size, vocab_size)
    self.softmax = nn.Softmax(dim=-1)
  
  def pool(hs_pad, ys_in_pad):
    if self.pool_type == 'att':
      key, query, value = self.key(hs_pad), self.query(ys_in_pad), self.value(hs_in_pad)
      pooled, att = self.multihead_att(key, query, value) 

    elif self.pool_type == 'mean':
      pooled, att = hs_pad.mean(dim=1), None

    elif self.pool_type == 'max':
      pooled, att = hs_pad.max(dim=1), None

    return pooled, att

  def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
    print(hs_pad)
    print(hlens)
    print(ys_in_pad)
    print(ys_in_lens)
    pooled, att = self.pool(hs_pad, ys_in_pad)
    logits = self.softmax(self.logistic(pooled)) 
    return logits


    
