import math
import torch
from typing import Optional
from torch import Tensor
from torch import exp, sin, cos
from torch import zeros, ones
from torch import rsqrt
from torch.nn.functional import softmax
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import init
from torch.nn import Sequential, Dropout
from torch.nn import Linear, ReLU


class PositionalEncoding(Module):
    def __init__(self, sequence_lenght_limit: int, model_dimension: int, scaling_factor: int = 10000, device=None, dtype=None):
        super().__init__()
        self.embeddings = Parameter(data=torch.zeros(sequence_lenght_limit, model_dimension, device=device, dtype=dtype), requires_grad=False)
        for dimension in range(model_dimension):
            self.embeddings[:,dimension] = dimension // 2 + 1
            self.embeddings[:,dimension] = exp(-2*self.embeddings[:,dimension] * math.log(scaling_factor) / model_dimension)
            for sequence in range(sequence_lenght_limit):
                if dimension % 2 == 0:
                    self.embeddings[sequence,dimension] = sin(sequence * self.embeddings[sequence,dimension])
                else:
                    self.embeddings[sequence,dimension] = cos(sequence * self.embeddings[sequence,dimension])

    def forward(self, input: Tensor) -> Tensor:
        input = input + self.embeddings[:,:input.size(1)]
        return input

def attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:        
    scale = 1 / math.sqrt(key.size(-1))
    score = query @ key.transpose(-2, -1) * scale
    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))
    return softmax(score, dim=-1) @ value

def split(sequence: Tensor, number_of_heads: int) -> Tensor:
    batch_size, sequence_length, model_dimension = sequence.size()
    sequence = sequence.view(batch_size, sequence_length, number_of_heads, model_dimension // number_of_heads)
    sequence = sequence.transpose(1, 2)
    return sequence

def concat(sequence: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.size()
    sequence = sequence.transpose(1, 2).contiguous()
    sequence = sequence.view(batch_size, sequence_lenght, heads_dimension* number_of_heads)
    return sequence

class MultiheadAttention(Module):
    def __init__(self, model_dimension: int, key_dimension: int, value_dimension: int, number_of_heads):
        super().__init__()
        self.number_of_heads = number_of_heads
        self.query_projector = Linear(model_dimension, model_dimension, bias=False)
        self.key_projector = Linear(key_dimension, model_dimension, bias=False)
        self.value_projector = Linear(value_dimension, model_dimension, bias=False)
        self.output_projector = Linear(model_dimension, model_dimension, bias=False)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query, key, value = self.query_projector(query), self.key_projector(key), self.value_projector(value)
        query, key, value = split(query, self.number_of_heads), split(key, self.number_of_heads), split(value, self.number_of_heads)
        heads = attention(query, key, value, mask)
        heads = concat(heads)
        return self.output_projector(heads)
    

class LayerNormalization(Module):
    def __init__(self, model_dimension: int, epsilon: float = 0.00001, bias: bool = True):
        super().__init__()
        self.bias = bias
        self.gamma = Parameter(data=ones(model_dimension))
        self.beta = Parameter(data=zeros(model_dimension)) if bias else None
        self.epsilon = epsilon

    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, unbiased=True, keepdim=True)
        output = self.gamma * ((input - mean) *rsqrt(variance + self.epsilon)) 
        return output + self.beta if self.bias else output
    

class FeedForward(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, dropout_rate: float = 0.2, bias: bool = False):
        super().__init__()
        self.layers = Sequential(
            Linear(model_dimension, hidden_dimension, bias),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dimension, model_dimension, bias)
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    

class Encoder(Module):
    def __init__(self, model_dimension: int, ffn_dimension: int, number_of_heads: int, dropout: float = 0.2, bias: bool = False):
        super().__init__()
        self.multihead_attention = MultiheadAttention(model_dimension, model_dimension, model_dimension, number_of_heads)
        self.first_layer_normalization = LayerNormalization(model_dimension, bias=bias)

        self.feed_forward_network = FeedForward(model_dimension, ffn_dimension, dropout, bias=bias)
        self.second_layer_normalization = LayerNormalization(model_dimension, bias=bias)

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sequence = sequence + self.multihead_attention(sequence, sequence, sequence, mask)
        sequence = self.first_layer_normalization(sequence)        
        sequence = sequence + self.feed_forward_network(sequence)
        sequence = self.second_layer_normalization(sequence)
        return sequence
    

class Decoder(Module):
    def __init__(self, model_dimension: int, ffn_dimension: int, number_of_heads: int, dropout: float = 0.2):
        super().__init__()
        self.first_multihead_attention = MultiheadAttention(model_dimension, model_dimension, model_dimension, number_of_heads)
        self.first_layer_normalization = LayerNormalization(model_dimension)

        self.second_multihead_attention = MultiheadAttention(model_dimension, model_dimension, model_dimension, number_of_heads)
        self.second_layer_normalization = LayerNormalization(model_dimension)
        
        self.feed_forward_network = FeedForward(model_dimension, ffn_dimension, dropout)
        self.third_layer_normalization = LayerNormalization(model_dimension)
        
    def forward(self, input: Tensor, output: Tensor, input_mask: Optional[Tensor] = None, output_mask: Optional[Tensor] = None) -> Tensor:
        output = output + self.first_multihead_attention(output, output, output, output_mask)
        output = self.first_layer_normalization(output)

        output = output + self.second_multihead_attention(input, input, output, input_mask)
        output = self.second_layer_normalization(output)

        output = output + self.feed_forward_network(output)
        output = self.third_layer_normalization(output)
        return output