// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_MM_WITH_UPSTREAM_H_
#define DALI_CORE_MM_WITH_UPSTREAM_H_

#include "dali/core/mm/memory_resource.h"

namespace dali {
namespace mm {

template <typename MemoryKind>
class with_upstream {
 public:
  virtual memory_resource<MemoryKind> *upstream() const = 0;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_WITH_UPSTREAM_H_
