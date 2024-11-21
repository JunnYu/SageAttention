"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import paddle
from typing import Any, List, Literal, Optional, Tuple, Union

from qattn_fused import quant_per_block_int8_cuda, quant_per_block_int8_fuse_sub_mean_cuda, quant_per_warp_int8_cuda
from qattn_fused import sub_mean_cuda
from qattn_fused import transpose_pad_permute_cuda
from qattn_fused import scale_fuse_quant_cuda, mean_scale_fuse_quant_cuda

def per_block_int8(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    km: Optional[paddle.Tensor] = None,
    BLKQ: int =128, 
    BLKK: int =64, 
    sm_scale: Optional[float] = None, 
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` and the key tensor `k` with per block quantization.

    Parameters
    ----------
    q : paddle.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : paddle.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[paddle.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    sm_scale : Optional[float]
        The scale factor for the softmax operation. Default is ``head_dim**-0.5``. 
        It will be multiplied by ``1.44269504`` to work together with the triton attention kernel.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``paddle.float16`` or ``paddle.bfloat16``
    """

    q_int8 = paddle.empty(q.shape, dtype=paddle.int8)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = paddle.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), dtype=paddle.float32)
    k_scale = paddle.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), dtype=paddle.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    
    sm_scale *= 1.44269504

    quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale

def per_warp_int8(
    q: paddle.Tensor, 
    k: paddle.Tensor, 
    km: Optional[paddle.Tensor] = None, 
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` with per warp quantization and the key tensor `k` with per block quantization.
    Warp size of quantizing `q` is 32, with a block size of 128.
    Block size of quantizing `k` is 64.

    Parameters
    ----------
    q : paddle.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : paddle.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[paddle.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // 128 * 4]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // 64]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``paddle.float16`` or ``paddle.bfloat16``
    """

    q_int8 = paddle.empty(q.shape, dtype=paddle.int8)
    k_int8 = paddle.empty(k.shape, dtype=paddle.int8)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = paddle.empty((b, h_qo, ((qo_len + 127) // 128) * (128 // 32)), dtype=paddle.float32)
    k_scale = paddle.empty((b, h_kv, (kv_len + 63) // 64), dtype=paddle.float32)

    quant_per_warp_int8_cuda(q, q_int8, q_scale, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, 64, _tensor_layout)
    else:
        quant_per_block_int8_cuda(k, k_int8, k_scale, 64, _tensor_layout)
    
    return q_int8, q_scale, k_int8, k_scale

def sub_mean(
    v: paddle.Tensor, 
    tensor_layout: str ="HND"
):
    """
    Calculate the mean of the tensor `v` along the sequence length dimension and subtract it from `v`. Result is stored as fp16.

    Parameters
    ----------
    v : paddle.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[paddle.Tensor, paddle.Tensor]
        A tuple containing:
        - The tensor `v_smoothed` with the mean subtracted and stored as fp16. Shape: Same as `v` with `float16` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with dtype same as `v`.

    Note
    ----
    - The tensors `v` must have the dtype ``paddle.float16`` or ``paddle.bfloat16``
    - The returned tensor `v_smoothed` will have dtype ``paddle.float16`` regardless of the input dtype.
    - The returned mean tensor will have the same dtype as the input tensor.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(axis=1 if _tensor_layout == 0 else 2)

    v_smoothed = paddle.empty(v.shape, dtype=paddle.float16)
    
    # subtract mean and store the result as fp16
    sub_mean_cuda(v, vm, v_smoothed, _tensor_layout)

    return v_smoothed, vm

def per_channel_fp8(
    v: paddle.Tensor,
    tensor_layout: str ="HND",
    scale_max: float = 448.0,
    smooth_v: bool = True
):
    """
    Transpose, pad and permute the tensor `v` and quantize it to fp8 with per channel quantization.
    `v` is first transposed along the head dimension and the sequence length dimension, then padded to a multiple of 64.
    After that, the tensor is permuted along the sequence length dimension by ``[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]``.
    The quantization is done per channel, with the scale value and smooth factor calculated per channel.

    Parameters
    ----------
    v : paddle.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    scale_max : float
        The maximum scale value for the quantization. Default is 448.0 (upper bound of E4M3 data format).

    smooth_v : bool
        Whether to smooth the quantized tensor. Default is True.

    Returns
    -------
    Tuple[paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]
        A tuple containing:
        - The quantized tensor `v_fp8`. Shape:
            - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, head_dim, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
            - If `tensor_layout` is "NHD": ``[batch_size, head_dim, num_kv_heads, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
        - The scale tensor of `v`. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.

    Note
    ----
    - The tensors `v` must have the dtype ``paddle.float16`` or ``paddle.bfloat16``
    - The returned mean tensor will be None if `smooth_v` is False. Otherwise it will have dtype ``paddle.float32``.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype)

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = paddle.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype)
    
    transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)

    v_fp8 = paddle.empty(v_transposed_permutted.shape, dtype="float8_e4m3fn")

    v_scale = paddle.empty((b, h_kv, head_dim), dtype=paddle.float32)
    vm = paddle.empty((b, h_kv, head_dim), dtype=paddle.float32)

    if smooth_v:
        mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None



    
