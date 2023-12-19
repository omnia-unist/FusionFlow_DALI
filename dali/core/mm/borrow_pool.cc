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

#include "dali/core/mm/borrow_pool.h"

namespace dali {
namespace mm {

static int SharedMemoryInit(int& shmid, int device_id) {
  shmid = shmget(reinterpret_cast<key_t>(device_id * KEY_REFERENCE + KEY_REFERENCE), 0, 0);
  if (shmid == -1) {
      fprintf(stdout, "[SharedMemoryInit-DALI] Failed\n");
      return -1;
  }
  fprintf(stdout, "[SharedMemoryInit-DALI] Success\n");
  return 0;
}

static int SharedMemoryRead(int& shmid, shared_block* shared_list) {
  void* shmaddr = shmat(shmid, NULL, 0);
  if (shmaddr == reinterpret_cast<void*>(-1)) {
      fprintf(stdout, "[SharedMemoryRead-DALI] Failed\n");
      return -1;
  }
  fprintf(stdout, "[SharedMemoryRead-DALI] Success\n");
  memset(shared_list, 0, sizeof(shared_block) * SHML_SIZE);
  mempcpy(shared_list, reinterpret_cast<char *>(shmaddr), sizeof(shared_block) * SHML_SIZE);
  fprintf(stdout, "[SharedMemoryRead-DALI] Printing Info on Memory for recheck!\n");
  // for (int i = 0; i < SHML_SIZE; i++) {
  //     fprintf(stdout, "Pointer %p with size %lu\n", shared_list[i].ptr, shared_list[i].size);
  // }
  fflush(stdout);
  return 0;
}

BorrowPool::BorrowPool(int device_id = 0) : device_id(device_id) {
  fprintf(stdout, "Initializtion of Borrow Pool");
  fflush(stdout);
}

void BorrowPool::init() {
  std::lock_guard<std::mutex> guard(lock_);
  fprintf(stdout, "Initializing Borrow Pool\n");
  if (shmid == -1) {
    SharedMemoryInit(shmid, 0);
    fflush(stdout);
  }
  SharedMemoryRead(shmid, shared_list);
  for (int i = 0; i < SHML_SIZE; i++) {
    borrow_list[i] = borrow_block {shared_list[i].ptr, shared_list[i].size, 1};
  }
}

void* BorrowPool::allocate(size_t bytes, size_t alignment) {
  std::lock_guard<std::mutex> guard(lock_);
  fprintf(stdout, "Trying to allocate %lu bytes:\n", bytes);
  fflush(stdout);

  borrow_block* target_block = nullptr;
  for (int i = 0; i < SHML_SIZE; i++) {
    // fprintf(stdout,
    // "\tTrying to allocate %lu\n, current candidate %p with %lu memory, Free State: %d\n",
    // bytes, borrow_list[i].ptr, borrow_list[i].size, borrow_list[i].is_free);
    // fflush(stdout);

    if (!borrow_list[i].is_free || bytes > borrow_list[i].size)
      continue;
    if (target_block == nullptr || target_block->size > borrow_list[i].size)
      target_block = borrow_list + i;
  }

  if (target_block) {
    fprintf(stdout,
    "Allocate block %p with %lu memory for %lu\n",
    target_block->ptr, target_block->size, bytes);
    fprintf(stdout, "[Allocation] Balance: %d\n", ++balance);
    fflush(stdout);

    target_block->is_free = 0;
    return target_block->ptr;
  }
  return nullptr;
}

bool BorrowPool::deallocate(void* ptr, size_t bytes, size_t alignment) {
  if (!ptr)
    return false;
  std::lock_guard<std::mutex> guard(lock_);
  for (int i = 0; i < SHML_SIZE; i++) {
    if (borrow_list[i].ptr == ptr) {
      fprintf(stdout, "Deallocate Block with ptr %p\n", ptr);
      fprintf(stdout, "[Deallocation] Balance: %d\n", --balance);
      fflush(stdout);

      borrow_list[i].is_free = 1;
      return true;
    }
  }
  return false;
}

}  // namespace mm
}  // namespace dali
