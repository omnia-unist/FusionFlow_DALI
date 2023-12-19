// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/core/static_switch.h"
#include "dali/operators/math/expressions/arithmetic_meta.h"
#include "dali/operators/math/expressions/expression_impl_gpu_binary.cuh"
#include "dali/operators/math/expressions/expression_factory_instances/expression_impl_factory.h"

namespace dali {
namespace expr {

IMPLEMENT_OP_FACTORY_GPU_BINARY(sub);

}  // namespace expr
}  // namespace dali
