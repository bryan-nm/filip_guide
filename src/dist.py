"""Distributed bootstrap for Aurora (Intel XPU + oneCCL).

Ported from mini-embed-filip (encoder-agnostic). Turns an `mpiexec`-launched
process into a torch.distributed rank pinned to one GPU tile. Aurora exposes 6
Max-1550 GPUs x 2 tiles = 12 tiles/node; we run one rank per tile (16 nodes x 12
= 192 ranks).

Everything degrades gracefully: with no launcher env (a laptop/MPS smoke test)
`init_distributed` returns a single-rank world on the best local device and never
touches the process group, so `train_retrieval.py` runs unchanged there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


# Intel extension registers the `xpu` backend and the oneCCL process-group
# backend. Both imports are optional so the module still loads on CPU/CUDA/Mac.
try:  # noqa: SIM105
    import intel_extension_for_pytorch as ipex  # noqa: F401
except Exception:
    ipex = None
try:  # registers the "ccl" backend for torch.distributed
    import oneccl_bindings_for_pytorch  # noqa: F401
except Exception:
    pass


def _resolve_xpu_backend() -> str:
    """Pick the distributed backend for Intel XPU: prefer native `xccl` (torch
    >=2.6 on the Aurora `frameworks` module), fall back to `ccl` (torch-ccl).
    Override with FILIPG_DIST_BACKEND if the autopick is wrong on a given build.
    """
    override = os.environ.get("FILIPG_DIST_BACKEND")
    if override:
        if override == "ccl":
            try:
                import oneccl_bindings_for_pytorch  # noqa: F401
            except Exception:
                pass
        return override
    try:
        if getattr(torch.distributed, "is_xccl_available", lambda: False)():
            return "xccl"
    except Exception:
        pass
    try:
        import oneccl_bindings_for_pytorch  # noqa: F401
        return "ccl"
    except Exception:
        return "xccl"


def _first_env(*names: str, default: int = 0) -> int:
    for n in names:
        v = os.environ.get(n)
        if v is not None and v != "":
            return int(v)
    return default


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str
    group_size: int = 1          # contrastive subgroup size (for grouped all-gather)
    group = None                 # torch.distributed.ProcessGroup for this rank's subgroup
    group_rank: int = 0          # this rank's index within its subgroup

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _detect_topology() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank). Ask MPI directly first (ground truth
    under mpiexec; the env-var path silently returned world=1 on Aurora MPICH)."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world = comm.Get_size()
        if world > 1:
            rank = comm.Get_rank()
            local = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
            return rank, world, local
    except Exception:
        pass
    world = _first_env("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", default=1)
    rank = _first_env("PALS_RANKID", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK", default=0)
    local = _first_env("PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "PMI_LOCAL_RANK",
                       "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", default=0)
    return rank, world, local


def _pick_local_device(local_rank: int, device_name: str) -> torch.device:
    if device_name not in ("auto", "xpu") and not device_name.startswith("xpu"):
        return torch.device(device_name)     # honour explicit cpu/cuda/mps (smoke tests)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        n = torch.xpu.device_count()
        idx = local_rank % max(n, 1)
        torch.xpu.set_device(idx)
        return torch.device(f"xpu:{idx}")
    if torch.cuda.is_available():
        idx = local_rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warn_if_device_selector_doubled(device, rank: int) -> None:
    """Catch the Aurora OpenCL+Level-Zero double-enumeration footgun early: with
    ONEAPI_DEVICE_SELECTOR exposing both backends every tile enumerates twice,
    ranks mis-pin across backends, and the GPU aborts with a 'NotPresent' page
    fault. The launch script must set ONEAPI_DEVICE_SELECTOR=level_zero:gpu."""
    if device.type != "xpu" or rank != 0:
        return
    sel = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    dc = torch.xpu.device_count()
    if "opencl" in sel.lower():
        print(f"[dist] WARNING: ONEAPI_DEVICE_SELECTOR={sel!r} enumerates OpenCL+Level-Zero, "
              f"doubling the device list (torch.xpu.device_count()={dc}). Set "
              f"ONEAPI_DEVICE_SELECTOR=level_zero:gpu in the launch script.", flush=True)


def init_distributed(device_name: str = "auto", group_size: int = 1, init_pg: bool = True) -> DistEnv:
    """Initialise the process group (if launched under MPI) and pin the tile.

    `group_size` > 1 builds contiguous rank subgroups of that size for the bounded
    FILIP negative pool (grouped_all_gather).
    """
    rank, world, local = _detect_topology()
    device = _pick_local_device(local, device_name)
    _warn_if_device_selector_doubled(device, rank)

    if world <= 1:
        return DistEnv(rank=0, world_size=1, local_rank=0, device=device,
                       backend="none", group_size=1, group_rank=0)
    if not init_pg:
        return DistEnv(rank=rank, world_size=world, local_rank=local, device=device,
                       backend="mpi", group_size=1, group_rank=0)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local)

    if device.type == "xpu":
        backend = _resolve_xpu_backend()
    elif device.type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(
                backend=backend, init_method="env://", world_size=world, rank=rank,
                device_id=device if device.type in ("xpu", "cuda") else None)
        except TypeError:
            torch.distributed.init_process_group(
                backend=backend, init_method="env://", world_size=world, rank=rank)
    if rank == 0:
        print(f"[dist] process group up: backend={backend} world={world} "
              f"group_size={group_size} rank0 device={device}", flush=True)
    env = DistEnv(rank=rank, world_size=world, local_rank=local, device=device,
                  backend=backend, group_size=max(group_size, 1))

    if group_size > 1:
        if world % group_size != 0 and rank == 0:
            print(f"[dist] WARNING: world_size={world} not divisible by "
                  f"group_size={group_size}; last group will be ragged.")
        for g0 in range(0, world, group_size):
            members = list(range(g0, min(g0 + group_size, world)))
            grp = torch.distributed.new_group(ranks=members, backend=backend)
            if rank in members:
                env.group = grp
                env.group_rank = members.index(rank)
                env.group_size = len(members)
    return env


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        return
    try:
        from mpi4py import MPI
        if MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Barrier()
    except Exception:
        pass


def cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce across the full world (no-op if not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t


# ---------------------------------------------------------------------------
# Manual data-parallel (DDP's reducer can't trace the contrastive graph)
# ---------------------------------------------------------------------------
def broadcast_parameters(model, src: int = 0) -> None:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    for p in model.parameters():
        torch.distributed.broadcast(p.data, src=src)


def average_gradients(model) -> None:
    """Average gradients across the full world after backward, before step.

    Used instead of DDP: the contrastive loss consumes logit_scale and the
    subgroup-gathered embeddings outside the module forward, which DDP mishandles.
    Grads are flattened into one buffer for a single all-reduce (the model is ~40
    small tensors; per-tensor collectives are latency-bound at high rank counts).
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    world = torch.distributed.get_world_size()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
    flat /= world
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


# ---------------------------------------------------------------------------
# Grouped all-gather of variable-length per-token tensors (grad on local slice)
# ---------------------------------------------------------------------------
def grouped_all_gather(local: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """All-gather `local` ([B,L,D] or [B,L]) within this rank's subgroup, padding
    L to the subgroup-max. The local slice keeps its autograd graph (gathered
    copies are detached), so gradients flow to the local rank's embeddings only."""
    if env.group is None or env.group_size <= 1:
        return local
    L = torch.tensor([local.size(1)], device=local.device)
    torch.distributed.all_reduce(L, op=torch.distributed.ReduceOp.MAX, group=env.group)
    L_max = int(L.item())
    if local.size(1) < L_max:
        pad = list(local.shape)
        pad[1] = L_max - local.size(1)
        local = torch.cat([local, local.new_zeros(*pad)], dim=1)
    gathered = [torch.empty_like(local) for _ in range(env.group_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=env.group)
    gathered[env.group_rank] = local
    return torch.cat(gathered, dim=0)


def grouped_all_gather_ids(local: torch.Tensor, env: DistEnv) -> torch.Tensor:
    """All-gather a 1-D int vector ([B]) within this rank's subgroup -> [G]
    (per-rank B constant via drop_last). Column order matches grouped_all_gather."""
    if env.group is None or env.group_size <= 1:
        return local
    gathered = [torch.empty_like(local) for _ in range(env.group_size)]
    torch.distributed.all_gather(gathered, local.contiguous(), group=env.group)
    return torch.cat(gathered, dim=0)
