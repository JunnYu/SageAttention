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

#pragma once
#include <paddle/extension.h>

#define CHECK_CUDA(x) \
  PD_CHECK(x.is_gpu(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)     \
  PD_CHECK(x.dtype() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)    \
  PD_CHECK(x.dims().size() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_NUMEL(x, minimum)     \
  PD_CHECK(x.numel() >= minimum, \
              "Tensor " #x " must have at last " #minimum " elements")
#define CHECK_SHAPE(x, ...)                                   \
  PD_CHECK(x.dims() == common::make_ddim({__VA_ARGS__}), \
              "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  PD_CHECK(true, "Tensor " #x " must be contiguous") // TODO: implement x.is_contiguous()
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  PD_CHECK(x.strides()[x.strides().size() - 1] == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")