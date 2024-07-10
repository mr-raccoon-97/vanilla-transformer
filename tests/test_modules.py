import torch
import pytest
from torch.nn import Parameter
from torch.nn import LayerNorm as PytorchLayerNorm
from torch.nn import MultiheadAttention as PytorchMultiheadAttention
from torch.nn.functional import scaled_dot_product_attention

from model.transformers import attention
from model.transformers import MultiheadAttention
from model.transformers import FeedForward
from model.transformers import LayerNormalization

def test_layer_normalization():
    layer_norm = LayerNormalization(512)
    pytorch_layer_norm = PytorchLayerNorm(512)

    pytorch_layer_norm.weight = Parameter(layer_norm.gamma.detach().clone())
    pytorch_layer_norm.bias = Parameter(layer_norm.beta.detach().clone())

    input = torch.randn(8, 32, 512)

    #TODO: The values are not close enough, need to investigate.
    assert torch.allclose(layer_norm(input), pytorch_layer_norm(input), atol=1e-2)


def test_attention():
    query = torch.randn(8, 4, 512)
    key = torch.randn(8, 4, 512)
    value = torch.randn(8, 4, 512)
    assert torch.allclose(attention(query, key, value), scaled_dot_product_attention(query, key, value), atol=1e-5)


def test_multihead_attention():
    query = torch.ones(8, 32, 512)
    key = torch.ones(8, 32, 256)
    value = torch.ones(8, 32, 256)

    multihead = MultiheadAttention(512, 256, 256, 8)
    pytorch_multihead = PytorchMultiheadAttention(512, 8, kdim=256, vdim=256, bias=False)
    pytorch_multihead.q_proj_weight = Parameter(multihead.query_projector_weight.detach().clone())
    pytorch_multihead.k_proj_weight = Parameter(multihead.key_projector_weight.detach().clone())
    pytorch_multihead.v_proj_weight = Parameter(multihead.value_projector_weight.detach().clone())
    pytorch_multihead.out_proj.weight = Parameter(multihead.output_projector_weight.detach().clone())

    assert torch.allclose(multihead(query, key, value), pytorch_multihead(query, key, value)[0], atol=1e-5)

    query = torch.ones(8, 32, 512)
    key = torch.ones(8, 32, 512)
    value = torch.ones(8, 32, 512)

    multihead = MultiheadAttention(512, 512, 512, 8)
    pytorch_multihead = PytorchMultiheadAttention(512, 8)
    with torch.no_grad():
        pytorch_multihead.in_proj_weight[:512,:].copy_(multihead.query_projector_weight.detach().clone())
        pytorch_multihead.in_proj_weight[512:512+512,:].copy_(multihead.key_projector_weight.detach().clone())
        pytorch_multihead.in_proj_weight[512+512:512+512+512,:].copy_(multihead.value_projector_weight.detach().clone())
        pytorch_multihead.out_proj.weight = Parameter(multihead.output_projector_weight.detach().clone())

    assert torch.allclose(multihead(query, key, value), pytorch_multihead(query, key, value)[0], atol=1e-5)