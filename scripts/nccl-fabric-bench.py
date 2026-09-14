#!/usr/bin/env python3
"""nccl-fabric-bench.py — 2-node NCCL all_reduce microbenchmark (fabric-only).

Run one rank per node inside the dspark vLLM container (it ships torch + NCCL,
no MPI needed). Pure fabric test: vLLM stays up and untouched.

Head (rank 0):
  docker exec deepseek-v4-flash-vllm-dspark-1 python3 -u /tmp/nccl-fabric-bench.py
Worker (rank 1, same env):
  ssh andrejs@spark2 "docker exec deepseek-v4-flash-vllm-dspark-1 python3 -u /tmp/nccl-fabric-bench.py"

Env needed in-container: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE (passed via
docker exec -e). NCCL_* come from the container's compose env; override with
-e NCCL_IB_HCA=... to A/B single vs dual PF.
"""
import os
import time

import torch
import torch.distributed as dist

SIZES_BYTES = [256 * 1024, 8 * 1024 * 1024, 64 * 1024 * 1024]
WARMUP = 10
ITERS = 20


def main():
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29510')}",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    rank = dist.get_rank()
    if rank == 0:
        print(f"world_size={dist.get_world_size()}", flush=True)
    for nbytes in SIZES_BYTES:
        tensor = torch.empty(nbytes // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(WARMUP):
            dist.all_reduce(tensor)
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if rank == 0:
            # nccl-tests convention: algbw = size/time; busbw = algbw*2*(n-1)/n.
            # For allreduce n=2 those are identical, so busbw here == size/time.
            n = dist.get_world_size()
            algbw = nbytes * ITERS / dt
            busbw = algbw * 2 * (n - 1) / n
            print(f"{nbytes / 1e6:.1f} MB: algbw={algbw / 1e9:.2f} GB/s "
                  f"busbw={busbw / 1e9:.2f} GB/s ({dt / ITERS * 1e3:.2f} ms/op)",
                  flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
