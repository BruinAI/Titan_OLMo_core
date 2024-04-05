"""
Train a mock FSDP transformer model. Launch this script via `torchrun`:
    torchrun --nproc-per-node=8 -m benchmarks.fsdp.train
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)

from .common import TransformerConfig, build_components, print_rank0

log = logging.getLogger(__name__)


def main(
    config: TransformerConfig,
    batch_size: int,
    num_batches: int = 100,
    fsdp_wrapper: Literal["torch", "olmo_core"] = "olmo_core",
    dry_run: bool = False,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
):
    model, optim, dataloader = build_components(
        config, batch_size, num_batches=num_batches, fsdp_wrapper=fsdp_wrapper
    )

    if load_path is not None:
        print_rank0(f"Loading checkpoint from {load_path}...")
        load_model_and_optim_state(load_path, model, optim)

    if dry_run:
        print_rank0("Dry run complete")
        return

    if save_path is not None:
        checkpoint_dir = Path(save_path) / "pretrain"
        print_rank0(f"Saving checkpoint to {checkpoint_dir}...")
        save_model_and_optim_state(checkpoint_dir, model, optim)

    print_rank0("Starting training...")
    for i, batch in enumerate(iter(dataloader)):
        log.debug("Batch: %s", batch)
        batch_start = time.monotonic()

        # Zero-gradients.
        optim.zero_grad()

        # Run forward pass.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)

            # Compute loss.
            logits_for_loss = logits[..., :-1, :].contiguous()
            logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
            labels = batch[..., 1:].contiguous()
            labels = labels.view(-1)
            loss = F.cross_entropy(logits_for_loss, labels)

        # Trigger backward pass.
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError(
                f"NaN loss encountered.\nInputs: {batch}\nLogits: {logits_for_loss}\nLabels: {labels}"
            )

        # Clip gradient norms.
        model.clip_grad_norm_(1.0)

        # Take optimizer step.
        optim.step()

        batch_end = time.monotonic()
        print_rank0(
            f"Batch [{i+1}/{num_batches}]:\n"
            f"  loss={loss.item():.3f}\n"
            f"  throughput/seconds_per_batch={batch_end-batch_start:.1f}",
        )

    if save_path is not None:
        checkpoint_dir = Path(save_path) / "final"
        print_rank0(f"Saving checkpoint to {checkpoint_dir}...")
        save_model_and_optim_state(checkpoint_dir, model, optim)

    print_rank0("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Train an FSDP model")
    parser.add_argument(
        "--fsdp",
        choices=["torch", "olmo_core"],
        default="olmo_core",
        help="""The FSDP implementation.""",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="""The model size.""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="""The per-device batch size.""",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="""The number of batches to train for.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--save-path",
        type=str,
    )
    parser.add_argument(
        "--load-path",
        type=str,
    )
    args = parser.parse_args()

    config: TransformerConfig
    if args.model_size == "tiny":
        config = TransformerConfig.tiny()
    elif args.model_size == "small":
        config = TransformerConfig.small()
    elif args.model_size == "medium":
        config = TransformerConfig.medium()
    else:
        raise NotImplementedError(args.model_size)

    if args.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    if args.debug and dist.get_rank() == 0:
        logging.basicConfig(level=logging.DEBUG)

    main(
        config,
        args.batch_size,
        num_batches=args.num_batches,
        fsdp_wrapper=args.fsdp,
        dry_run=args.dry_run,
        save_path=args.save_path,
        load_path=args.load_path,
    )
