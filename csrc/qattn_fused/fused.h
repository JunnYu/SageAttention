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

#include <paddle/extension.h>

void quant_per_block_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                paddle::Tensor input,
                paddle::Tensor mean,
                paddle::Tensor output,
                paddle::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int tensor_layout);

void sub_mean_cuda(
                paddle::Tensor input,
                paddle::Tensor mean,
                paddle::Tensor output,
                int tensor_layout);

void transpose_pad_permute_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                int tensor_layout);

void scale_fuse_quant_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda(
                paddle::Tensor input,
                paddle::Tensor output,
                paddle::Tensor mean,
                paddle::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);