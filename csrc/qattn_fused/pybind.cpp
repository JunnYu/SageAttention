/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <paddle/extension.h>
#include "attn_cuda.h"
#include <cuda_fp16.h>
#include "fused.h"


PYBIND11_MODULE(qattn_fused, m)
{
  // qattn
  m.def("qk_int8_sv_f16_accum_f32_attn_per_warp", &qk_int8_sv_f16_accum_f32_attn_per_warp, "QK int8 sv f16 accum f32 attn per warp");
  m.def("qk_int8_sv_f16_accum_f16_attn_per_warp", &qk_int8_sv_f16_accum_f16_attn_per_warp, "QK int8 sv f16 accum f16 attn per warp");
  m.def("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_per_warp", &qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_per_warp, "QK int8 sv f16 accum f16 attn per warp fuse v mean");

  m.def("qk_int8_sv_f8_accum_f32_attn_per_warp", &qk_int8_sv_f8_accum_f32_attn_per_warp, "QK int8 sv f8 accum f32 attn");
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp", &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp, "QK int8 sv f8 accum f32 fuse v scale attn");
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_per_warp", &qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_per_warp, "QK int8 sv f8 accum f32 fuse v scale fuse v mean attn");

  m.def("qk_int8_sv_f16_accum_f16_attn_per_warp_buf", &qk_int8_sv_f16_accum_f16_attn_per_warp_buf, "QK int8 sv f16 accum f16 attn per warp buf");
  m.def("qk_int8_sv_f8_accum_f32_attn_per_warp_buf", &qk_int8_sv_f8_accum_f32_attn_per_warp_buf, "QK int8 sv f8 accum f32 attn per warp buf");
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp_buf", &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp_buf, "QK int8 sv f8 accum f32 fuse v scale attn per warp buf");
  // fused
  m.def("quant_per_block_int8_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, float, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, int, int>(&quant_per_block_int8_cuda), "quant_per_block_int8_cuda");
  m.def("quant_per_block_int8_fuse_sub_mean_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, paddle::Tensor, int, int>(&quant_per_block_int8_fuse_sub_mean_cuda), "quant_per_block_int8_fuse_sub_mean_cuda");
  m.def("quant_per_warp_int8_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, int>(&quant_per_warp_int8_cuda), "quant_per_warp_int8_cuda");

  m.def("sub_mean_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, int>(&sub_mean_cuda), "sub_mean_cuda");

  m.def("transpose_pad_permute_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, int>(&transpose_pad_permute_cuda), "transpose_pad_permute_cuda");
  m.def("scale_fuse_quant_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, int, float, int>(&scale_fuse_quant_cuda), "scale_fuse_quant_cuda");
  m.def("mean_scale_fuse_quant_cuda", py::overload_cast<paddle::Tensor, paddle::Tensor, paddle::Tensor, paddle::Tensor, int, float, int>(&mean_scale_fuse_quant_cuda), "mean_scale_fuse_quant_cuda");
}