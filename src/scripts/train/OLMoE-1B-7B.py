"""
Train a 1B-7B OLMoE model (mixture of experts).
Run this script without any arguments to see usage info.
"""

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.internal.experiment import CommonComponents, ExperimentConfig, main
from olmo_core.launch.beaker import OLMoCoreBeakerImage
from olmo_core.nn.moe import MoEActivationFn, MoEConfig, MoEMLPImplementation, MoEType
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.train import TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    MoEHandlerCallback,
    WandBCallback,
)


def finalize_config(config: ExperimentConfig):
    # need dev image for megablocks and grouped-gemm
    config.launch.beaker_image = OLMoCoreBeakerImage.dev


def build_model_config(common: CommonComponents) -> TransformerConfig:
    model_config = TransformerConfig.olmo_1B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        n_layers=16,
        n_heads=16,
        compile=False,
        fused_ops=False,
        block_name=TransformerBlockType.moe_reordered_norm,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )
    model_config.block.feed_forward = None
    model_config.block.feed_forward_moe = MoEConfig(
        name=MoEType.dropless,
        hidden_size=int(0.5 * model_config.d_model),
        activation_fn=MoEActivationFn.swiglu,
        mlp_implementation=MoEMLPImplementation.grouped,
        num_experts=64,
        top_k=8,
        num_layers=model_config.n_layers,
        zloss_weight=0.001,
        loss_weight=0.01,
        bias=False,
        dtype=model_config.dtype,
    )
    return model_config


def build_optim_config(common: CommonComponents) -> AdamWConfig:
    del common
    return AdamWConfig(
        lr=4e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
        fused=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            rank_microbatch_size=2 * 4096,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            z_loss_multiplier=1e-5,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "moe",
            MoEHandlerCallback(),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-1B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-1B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    main(
        global_batch_size=1024 * 4096,
        model_config_builder=build_model_config,
        optim_config_builder=build_optim_config,
        trainer_config_builder=build_trainer_config,
        finalize_config=finalize_config,
    )
