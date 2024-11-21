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

paddle::Tensor qk_int8_sv_f16_accum_f32_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f16_accum_f16_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f16_accum_f16_attn_per_warp_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f16_accum_f16_fuse_v_mean_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_mean,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f8_accum_f32_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn_per_warp(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    paddle::Tensor value_mean,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f8_accum_f32_attn_per_warp_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);

paddle::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_per_warp_buf(paddle::Tensor query,
                    paddle::Tensor key,
                    paddle::Tensor value,
                    paddle::Tensor output,
                    paddle::Tensor query_scale,
                    paddle::Tensor key_scale,
                    paddle::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    float sm_scale,
                    int return_lse);