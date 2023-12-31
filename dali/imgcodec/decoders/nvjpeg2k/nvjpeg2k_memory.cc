// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k_memory.h"
#include "dali/imgcodec/decoders/memory_pool.h"

namespace dali {
namespace imgcodec {
namespace nvjpeg_memory {

nvjpeg2kDeviceAllocator_t GetDeviceAllocatorNvJpeg2k() {
  nvjpeg2kDeviceAllocator_t allocator;
  allocator.device_malloc = &DeviceNew;
  allocator.device_free = &ReturnBufferToPool;
  return allocator;
}

nvjpeg2kPinnedAllocator_t GetPinnedAllocatorNvJpeg2k() {
  nvjpeg2kPinnedAllocator_t allocator;
  allocator.pinned_malloc = &HostNew;
  allocator.pinned_free = &ReturnBufferToPool;
  return allocator;
}

}  // namespace nvjpeg_memory
}  // namespace imgcodec
}  // namespace dali
