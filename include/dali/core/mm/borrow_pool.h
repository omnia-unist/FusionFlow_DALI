// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef DALI_CORE_MM_BORROW_POOL_H_
#define DALI_CORE_MM_BORROW_POOL_H_

#include <sys/shm.h>
#include <stdlib.h>
#include <malloc.h>
#include <cuda_runtime.h>
#include <mutex>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/cuda_error.h"
#include "dali/core/mm/detail/align.h"
#include "dali/core/device_guard.h"

namespace dali {
namespace mm {

#define  SHML_SIZE  30
#define  KEY_REFERENCE 1024

struct shared_block {
  cudaIpcMemHandle_t my_handle;
  void* ptr;
  size_t size;
  int shml_index;
  bool sharable;
};

class BorrowPool {
 public:
  explicit BorrowPool(int device_id);

  void init();

  void* allocate(size_t bytes, size_t alignment);

  bool deallocate(void* ptr, size_t bytes, size_t alignment);

 private:
  struct borrow_block {
    void* ptr = nullptr;
    size_t size = 0;
    bool is_free = 0;
  };

  std::mutex lock_;
  int balance = 0;
  int shmid = -1;
  int device_id = 0;
  shared_block shared_list[SHML_SIZE];
  borrow_block borrow_list[SHML_SIZE];
};

extern BorrowPool borrow_pool_;

}  // namespace mm
}  // namespace dali
#endif  // DALI_CORE_MM_BORROW_POOL_H_
