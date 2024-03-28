import json
import logging
import struct
import sys
import tempfile
from dataclasses import replace
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import safetensors as sft
import safetensors.torch as sft_torch
import torch
import torch.nn as nn
from cached_path import cached_path
from pydantic import BaseModel

from olmo_core.exceptions import OLMoUserError
from olmo_core.io import (
    PathOrStr,
    clear_directory,
    deserialize_from_tensor,
    dir_is_empty,
    file_exists,
    get_bytes_range,
    is_url,
    serialize_to_tensor,
    upload,
)
from olmo_core.utils import TORCH_DTYPE_TO_STR, TORCH_DTYPES, wait_for

from .sharded_flat_parameter import ShardedFlatParameter
from .utils import barrier, get_rank, scatter_object

log = logging.getLogger(__name__)


class TensorStorageMetadata(BaseModel):
    flattened_offsets_per_file: Dict[str, Tuple[int, int]]
    """
    Maps file name to the offsets within the full flattened tensor that the shard in the file
    corresponds to.
    """

    shape: Tuple[int, ...]
    """
    The shape of the full (unflattened) tensor.
    """

    dtype: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return TORCH_DTYPES[self.dtype]


class StorageMetadata(BaseModel):
    tensors: Dict[str, TensorStorageMetadata]


class TensorSavePlan(BaseModel):
    flattened_offsets_per_rank: Dict[int, Tuple[int, int]]
    """
    Maps global process rank to the offsets within the full flattened tensor that the shard for the
    rank corresponds to. Some ranks may be omitted.
    """


class SavePlan(BaseModel):
    tensors: Dict[str, TensorSavePlan]


class SafeTensorsLoader:
    """
    A wrapper around ``safetensors`` loading functionality for PyTorch that works with remote
    files as well without having to download the whole file.

    This should be used a context manager.
    """

    def __init__(self, path: PathOrStr):
        self.path = path
        self.safe_open: Optional[sft.safe_open] = None

    @cached_property
    def header_length(self) -> int:
        return struct.unpack("<Q", get_bytes_range(self.path, 0, 8))[0]

    @cached_property
    def header(self) -> Dict[str, Any]:
        return json.loads(get_bytes_range(self.path, 8, self.header_length))

    def get_shape(self, key: str) -> Tuple[int, ...]:
        return self.header[key]["shape"]

    def get_dtype(self, key: str) -> torch.dtype:
        return sft_torch._getdtype(self.header[key]["dtype"])

    def get_numel(self, key: str) -> int:
        return reduce(lambda x, y: x * y, self.get_shape(key), 1)

    def get_flat_slice(self, key: str, start_idx: int = 0, end_idx: Optional[int] = None) -> torch.Tensor:
        if self.safe_open is not None:
            return self.safe_open.get_slice(key)[start_idx:end_idx]  # type: ignore
        elif is_url(self.path):
            # Validate indices. Can only work with positive indices.
            if start_idx < 0:
                start_idx = self.get_numel(key) + start_idx
            elif start_idx > self.get_numel(key):
                raise IndexError(f"slice start index ({start_idx}) out of range")

            if end_idx is None:
                end_idx = self.get_numel(key)
            elif end_idx < 0:
                end_idx = self.get_numel(key) + end_idx
            elif end_idx > self.get_numel(key):
                raise IndexError(f"slice end index ({end_idx}) out of range")

            dtype = self.get_dtype(key)
            bytes_per_item = sft_torch._SIZE[dtype]
            num_bytes = bytes_per_item * (end_idx - start_idx)

            # Transform `start_idx` into a byte offset.
            offset_start = self.header[key]["data_offsets"][0]
            offset_start += bytes_per_item * start_idx
            # At this point `offset_start` is an offset into the byte-buffer part
            # of the file, not the file itself. We have to offset further by the header size byte
            # and the number of bytes in the header itself.
            offset_start += 8 + self.header_length

            # Load the tensor.
            array_bytes = get_bytes_range(self.path, offset_start, num_bytes)
            tensor = torch.frombuffer(bytearray(array_bytes), dtype=dtype)
            if sys.byteorder == "big":
                tensor = torch.from_numpy(tensor.numpy().byteswap(inplace=False))
            return tensor
        else:
            raise OLMoUserError(
                f"{self.__class__.__name__} is meant to be used as a context manager, did you forget to call __enter__?"
            )

    def __enter__(self):
        if not is_url(self.path):
            self.safe_open = sft.safe_open(self.path, framework="pt", device="cpu")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.safe_open is not None:
            self.safe_open.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
            self.safe_open = None


class SafeTensorsMultiFileLoader:
    """
    A wrapper around :class:`SafeTensorsLoader` that should be used when working with multiple ``safetensors``
    files at once to avoid unnecessary IO.
    """

    def __init__(self):
        self.loaders: Dict[str, SafeTensorsLoader] = {}

    def open(self, path: PathOrStr) -> SafeTensorsLoader:
        if (loader := self.loaders.get(str(path))) is not None:
            return loader
        loader = SafeTensorsLoader(path)
        self.loaders[str(path)] = loader
        return loader


class Checkpointer:
    """
    A checkpointer for saving and loading *non-nestaed* state dictionaries, i.e. where keys are strings and values
    are either regular :class:`torch.Tensor`s, :class:`torch.nn.Parameter`s, or :class:`ShardedFlatParameter`s.

    For saving and loading model and optimizer states together, use :func:`save_model_and_optim_state()`
    and :func:`load_model_and_optim_state()` instead.
    """

    METADATA_FILENAME = "metadata.json"

    @torch.no_grad()
    def save(self, dir: PathOrStr, state_dict: Dict[str, torch.Tensor], save_overwrite: bool = False):
        """
        Save a state dict. The state dict can contain regular Tensors, Parameters, or :class:`ShardedFlatParameter`s.

        When calling this from a distributed context, all ranks must call this at the same time and the
        state dict must have the same keys and tensor types across each rank.
        """
        dir = self._normalize_dir(dir)

        local_rank = get_rank()

        local_dir: Path
        remote_dir: Optional[str] = None
        clean_up_local_dir = False
        if not is_url(dir):
            local_dir = Path(dir)
            if local_rank == 0:
                if save_overwrite and not dir_is_empty(local_dir):
                    clear_directory(local_dir)
                local_dir.mkdir(parents=True, exist_ok=True)

            barrier()

            # All ranks wait for rank 0 to create the directory. On NFS the directory might
            # not be available immediately. This also ensures all ranks share the filesystem.
            description = f"Waiting for '{local_dir}' to be created"
            try:
                wait_for(local_dir.exists, description)
            except TimeoutError as e:
                raise RuntimeError(
                    f"{description} timed out, please ensure each rank is saving to the same directory on a shared filesystem."
                ) from e
        else:
            local_dir = Path(tempfile.mkdtemp())
            remote_dir = str(dir).rstrip("/")
            clean_up_local_dir = True
            # NOTE: we do have the ability to clear bucket storage "folders" via `clear_directory`,
            # but that's super dangerous. All it takes is one person passing in the wrong folder
            # name and they could wipe out a ton of very important checkpoints.
            if local_rank == 0:
                if file_exists(f"{remote_dir}/{self.METADATA_FILENAME}"):
                    raise FileExistsError(
                        f"Remote checkpoint directory '{remote_dir}' already contains a checkpoint!"
                    )

        try:
            if not dir_is_empty(local_dir):
                raise FileExistsError(f"Checkpoint directory '{local_dir}' is not empty!")

            barrier()

            flat_views, global_save_plan, metadata = self._get_global_save_plan_and_metadata(state_dict)

            # Construct local flat tensors state dict to save.
            local_state_dict: Dict[str, torch.Tensor] = {}
            for key in state_dict.keys():
                tensor_save_plan = global_save_plan.tensors[key]
                local_flat_tensor = flat_views[key]

                if (local_offsets := tensor_save_plan.flattened_offsets_per_rank.get(local_rank)) is not None:
                    assert local_offsets[1] - local_offsets[0] == local_flat_tensor.numel()
                    local_state_dict[key] = local_flat_tensor

            # Save safetensors file.
            local_sft_path = local_dir / self._filename_for_rank(local_rank)
            sft_torch.save_file(local_state_dict, local_sft_path)
            if remote_dir is not None:
                upload(
                    local_sft_path,
                    f"{remote_dir}/{self._filename_for_rank(local_rank)}",
                    save_overwrite=save_overwrite,
                )

            # Save metadata.
            if local_rank == 0:
                metadata_path = local_dir / self.METADATA_FILENAME
                with open(metadata_path, "w") as f:
                    json.dump(metadata.model_dump(), f)

                if remote_dir is not None:
                    upload(metadata_path, f"{remote_dir}/{self.METADATA_FILENAME}", save_overwrite=save_overwrite)

            barrier()
        finally:
            if clean_up_local_dir and local_dir.exists():
                clear_directory(local_dir)

    @torch.no_grad()
    def load(
        self,
        dir: PathOrStr,
        state_dict: Dict[str, torch.Tensor],
        no_dist: bool = False,
        _safetensors_mfl: Optional[SafeTensorsMultiFileLoader] = None,
        _metadata: Optional[StorageMetadata] = None,
    ):
        """
        Load a state dict in-place.

        :param dir: The state dict directory.
        :param state_dict: The state dict to load into.
        :param no_dist: Disable distributed communication even if within a distributed context.
        """
        dir = self._normalize_dir(dir)
        metadata = _metadata or self._collect_metadata(dir, no_dist=no_dist)
        safetensors_mfl = _safetensors_mfl or SafeTensorsMultiFileLoader()

        # Load each tensor from the slices in each file.
        for key in state_dict.keys():
            tensor_storage_metadata = metadata.tensors[key]
            tensor = state_dict[key]

            flat_tensor, full_shape, offsets_per_rank = self._get_flat_view_and_full_shape_and_flattened_offsets(
                tensor
            )
            # Rank 0 will always be present, other ranks will not be for regular unsharded tensors.
            offsets = offsets_per_rank.get(get_rank(), offsets_per_rank[0])
            if full_shape != tensor_storage_metadata.shape:
                raise ValueError(
                    f"Shape mismatched for '{key}', expected {full_shape}, found {tensor_storage_metadata.shape}"
                )

            for filename, offsets_in_file in tensor_storage_metadata.flattened_offsets_per_file.items():
                # Check for overlap in offsets, and if there is overlap, load the slice from disk.
                if (
                    offsets_in_file[0] <= offsets[0] < offsets_in_file[1]
                    or offsets_in_file[0] < offsets[1] <= offsets_in_file[1]
                ):
                    with safetensors_mfl.open(f"{dir}/{filename}") as loader:
                        if len((shape_in_file := loader.get_shape(key))) != 1:
                            raise ValueError(
                                f"Expected a 1D tensor at {key} in {filename}, found shape {shape_in_file}"
                            )

                        if (dtype := loader.get_dtype(key)) != flat_tensor.dtype:
                            raise ValueError(
                                f"Data type mismatch between tensor to load ({dtype}) and to load into ({flat_tensor.dtype})"
                            )

                        numel_in_file = loader.get_numel(key)

                        # Start and end index of the slice within `flat_tensor` that we're going to load
                        # from a slice of `flat_tensor_to_load`.
                        flat_tensor_start, flat_tensor_end = 0, flat_tensor.numel()
                        # Start and end index of the slice within `flat_tensor_to_load` that we're going
                        # to load into the slice of `flat_tensor`.
                        flat_tensor_to_load_start, flat_tensor_to_load_end = 0, numel_in_file
                        # There are 5 scenarios to consider in terms of where the tensors overlap.
                        # Suppose the original flat tensor has 6 elements: 'x x x x x x'
                        # -------------------------------------------
                        # (A) flat_tensor_slice_to_load: [x x x]x x x  (0, 3)
                        #     flat_tensor:                x x[x x x]x  (2, 5)
                        # -------------------------------------------
                        # (B) flat_tensor_slice_to_load:  x x[x x x]x  (2, 5)
                        #     flat_tensor:               [x x x]x x x  (0, 3)
                        # -------------------------------------------
                        # (C) flat_tensor_slice_to_load:  x[x x x x]x  (1, 5)
                        #     flat_tensor:                x x[x x]x x  (2, 4)
                        # -------------------------------------------
                        # (D) flat_tensor_slice_to_load:  x x[x x]x x  (2, 4)
                        #     flat_tensor:                x[x x x x]x  (1, 5)
                        # -------------------------------------------
                        # (E) flat_tensor_slice_to_load:  x x[x x]x x  (2, 4)
                        #     flat_tensor:                x x[x x]x x  (2, 4)
                        # -------------------------------------------
                        if offsets[0] <= offsets_in_file[0]:
                            # Scenarios (B), (D), (E)
                            flat_tensor_start = offsets_in_file[0] - offsets[0]
                        else:
                            # Scenarios (A), (C)
                            flat_tensor_to_load_start = offsets[0] - offsets_in_file[0]

                        if offsets[1] <= offsets_in_file[1]:
                            # Scenarios (B), (C), (E)
                            flat_tensor_to_load_end -= offsets_in_file[1] - offsets[1]
                        else:
                            # Scenarios (A), (D)
                            flat_tensor_end -= offsets[1] - offsets_in_file[1]

                        log.debug(
                            "Loading '%s'\n  offsets: %s\n  offsets in file: %s\n  load into: (%s, %s)\n  load from: (%s, %s)",
                            key,
                            offsets,
                            offsets_in_file,
                            flat_tensor_start,
                            flat_tensor_end,
                            flat_tensor_to_load_start,
                            flat_tensor_to_load_end,
                        )

                        # Load the slice.
                        flat_tensor_to_load = loader.get_flat_slice(
                            key, flat_tensor_to_load_start, flat_tensor_to_load_end
                        )
                        flat_tensor[flat_tensor_start:flat_tensor_end].copy_(flat_tensor_to_load)

                        del flat_tensor_to_load

            state_dict[key] = self._copy_into(tensor, flat_tensor)
            del flat_tensor

    @torch.no_grad()
    def unshard(
        self,
        dir: PathOrStr,
        device: Optional[torch.device] = None,
        rank0_only: bool = False,
        no_dist: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Unshard a checkpoint, returning the full state dict. This can be used in both distributed
        and non-distributed contexts. If you only want to load a single copy to rank 0 in a distributed
        context, set ``rank0_only=True``, in which case other ranks will receive an empty state dict.

        Alternatively, setting ``no_dist=True`` will return a full state dict from whatever process
        calls this.
        """
        dir = self._normalize_dir(dir)

        if rank0_only and no_dist and get_rank() != 0:
            raise ValueError(
                f"calling `unshard()` with `rank0_only=True` and `no_dist=True` is undefined for rank `{get_rank()} != 0`"
            )
        elif rank0_only and get_rank() != 0:
            return {}

        # Load metadata.
        metadata = self._collect_metadata(dir, no_dist=no_dist or rank0_only)

        # Initialize state dict.
        state_dict = {}
        for key, tensor_metadata in metadata.tensors.items():
            tensor = torch.empty(tensor_metadata.shape, dtype=tensor_metadata.torch_dtype, device=device)
            state_dict[key] = tensor

        # Load the state dict in place.
        self.load(dir, state_dict, _metadata=metadata, no_dist=no_dist or rank0_only)

        return state_dict

    def _filename_for_rank(self, rank: int) -> str:
        return f"rank_{rank}.safetensors"

    def _copy_into(self, target: torch.Tensor, source: torch.Tensor):
        target_data = get_local_tensor_data(target)
        source_data = get_local_tensor_data(source)
        target_data.copy_(source_data.view(target_data.shape))
        return target

    def _get_flat_view_and_full_shape_and_flattened_offsets(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, ...], Dict[int, Tuple[int, int]]]:
        from torch.distributed._shard.sharded_tensor import (
            ShardedTensor as TorchShardedTensor,
        )

        flat_view: torch.Tensor
        full_shape: Tuple[int, ...]
        flattened_offsets_per_rank: Dict[int, Tuple[int, int]] = {}
        if isinstance(tensor, ShardedFlatParameter):
            flat_view = tensor.data
            full_shape = tensor.unsharded_shape
            for rank, offset in enumerate(tensor.sharding_spec.unsharded_flattened_offsets):
                flattened_offsets_per_rank[rank] = offset
        elif isinstance(tensor, TorchShardedTensor):
            metadata = tensor.metadata()
            # For example, the metadata for a (16, 8) tensor could look like this:
            #  ShardedTensorMetadata(
            #      shards_metadata=[
            #          ShardMetadata(shard_offsets=[0, 0], shard_sizes=[8, 8], placement="rank:0/cpu"),
            #          ShardMetadata(shard_offsets=[8, 0], shard_sizes=[8, 8], placement="rank:1/cpu"),
            #      ],
            #      size=torch.Size([16, 8]),
            #      tensor_properties=TensorProperties(
            #          dtype=torch.float32,
            #          layout=torch.strided,
            #          requires_grad=False,
            #          memory_format=torch.contiguous_format,
            #          pin_memory=False,
            #      ),
            #  )
            flat_view = tensor.local_tensor().view(-1)
            full_shape = tuple(metadata.size)
            numel_per_flat_row = reduce(lambda a, b: a * b, full_shape[1:], 1)
            for shard_metadata in metadata.shards_metadata:
                assert shard_metadata.placement is not None
                rank = shard_metadata.placement.rank()
                assert rank is not None
                # We make the simplifying assumption that the tensor is sharded in the 1st dimension only (dim=0).
                for dim_start in shard_metadata.shard_offsets[1:]:
                    if dim_start != 0:
                        raise NotImplementedError(
                            f"this is only implemented for tensors sharded in the 1st dimension"
                        )
                # This logic is only valid under the above assumption.
                offset_start = shard_metadata.shard_offsets[0] * numel_per_flat_row
                offset_end = offset_start + shard_metadata.shard_sizes[0] * numel_per_flat_row
                flattened_offsets_per_rank[rank] = (offset_start, offset_end)
        else:
            flat_view = tensor.view(-1)
            flattened_offsets_per_rank = {0: (0, tensor.numel())}
            full_shape = tuple(tensor.shape)
        return flat_view, full_shape, flattened_offsets_per_rank

    def _get_global_save_plan_and_metadata(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], SavePlan, StorageMetadata]:
        tensors_flat_view = {}
        tensors_save_plan = {}
        tensors_metadata = {}
        for key, tensor in state_dict.items():
            (
                flat_view,
                full_shape,
                flattened_offsets_per_rank,
            ) = self._get_flat_view_and_full_shape_and_flattened_offsets(tensor)
            tensors_flat_view[key] = flat_view
            tensors_save_plan[key] = TensorSavePlan(flattened_offsets_per_rank=flattened_offsets_per_rank)
            tensors_metadata[key] = TensorStorageMetadata(
                flattened_offsets_per_file={
                    self._filename_for_rank(rank): offsets for rank, offsets in flattened_offsets_per_rank.items()
                },
                shape=full_shape,
                dtype=TORCH_DTYPE_TO_STR[tensor.dtype],
            )

        tensors_save_plan = scatter_object(tensors_save_plan)
        tensors_metadata = scatter_object(tensors_metadata)
        return tensors_flat_view, SavePlan(tensors=tensors_save_plan), StorageMetadata(tensors=tensors_metadata)

    def _normalize_dir(self, dir: PathOrStr) -> str:
        dir = str(dir).strip("/")
        if dir.startswith("file://"):
            dir = dir.replace("file://", "", 1)
        return dir

    def _collect_metadata(self, dir: str, no_dist: bool = False) -> StorageMetadata:
        metadata: Optional[StorageMetadata] = None
        if no_dist or get_rank() == 0:
            with open(cached_path(f"{dir}/{self.METADATA_FILENAME}")) as f:
                metadata = StorageMetadata(**json.load(f))
        if not no_dist:
            metadata = scatter_object(metadata)
        assert metadata is not None
        return metadata


class ParamGroup(TypedDict):
    params: List[int]
    """
    Parameter IDs.
    """


class OptimStateDict(TypedDict):
    state: Dict[int, Dict[str, torch.Tensor]]
    """
    Maps parameter IDs to the optimizer-specific state of each parameter.
    """

    param_groups: List[ParamGroup]
    """
    Parameter groups.
    """


def get_local_tensor_data(tensor: torch.Tensor) -> torch.Tensor:
    from torch.distributed._shard.sharded_tensor import (
        ShardedTensor as TorchShardedTensor,
    )

    if isinstance(tensor, TorchShardedTensor):
        return tensor.local_tensor()
    else:
        return tensor.data


def wrap_tensor_for_sharded_parameter(tensor: torch.Tensor, param: Optional[torch.Tensor]) -> torch.Tensor:
    from torch.distributed._shard.sharded_tensor import (
        ShardedTensor as TorchShardedTensor,
    )

    if isinstance(param, ShardedFlatParameter):
        return param.wrap(tensor, requires_grad=False)
    elif isinstance(param, TorchShardedTensor):
        shards = param.local_shards()
        assert len(shards) == 1
        new_shards = [replace(shards[0], tensor=tensor)]
        return TorchShardedTensor._init_from_local_shards_and_global_metadata(
            new_shards,
            param.metadata(),
        )
    else:
        return tensor


def flatten_optimizer_state(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    model_state: Optional[Dict[str, torch.Tensor]] = None,
    optim_state: Optional[OptimStateDict] = None,
) -> Dict[str, torch.Tensor]:
    model_state = model_state or model.state_dict()
    optim_state = optim_state or optim.state_dict()  # type: ignore
    assert optim_state

    param_ids_are_names: bool = False

    # Collect mapping of parameter IDs from the optimizer to the FQN of the corresponding parameter.
    name_to_param: Dict[str, nn.Parameter] = {k: v for k, v in model.named_parameters()}
    param_to_name: Dict[nn.Parameter, str] = {v: k for k, v in model.named_parameters()}
    param_id_to_name: Dict[int, str] = {}
    for param_group, param_group_state in zip(optim.param_groups, optim_state["param_groups"]):
        for param, param_id in zip(param_group["params"], param_group_state["params"]):
            if isinstance(param_id, str):  # PyTorch FSDP does this.
                param_id = len(param_id_to_name)
                param_ids_are_names = True
            param_id_to_name[param_id] = param_to_name[param]
    del param_to_name

    flat_optim_state: Dict[str, torch.Tensor] = {}

    # Serialize param groups to tensors.
    flat_optim_state["num_param_groups"] = torch.tensor(len(optim_state["param_groups"]))
    for i, param_group in enumerate(optim_state["param_groups"]):
        # make copy.
        param_group = {k: v for k, v in param_group.items()}
        if param_ids_are_names:
            param_group["param_names"] = param_group["params"]
        else:
            param_group["param_names"] = [param_id_to_name[param_id] for param_id in param_group["params"]]  # type: ignore
        flat_optim_state[f"param_group{i}"] = serialize_to_tensor(param_group)

    # Flatten state tensors and wrap any tensor with the right sharded class if the corresponding
    # parameter is sharded.
    state_keys: Set[str] = set()
    for param_id, state in optim_state["state"].items():
        if isinstance(param_id, str):
            param_name = param_id
        else:
            param_name = param_id_to_name[param_id]
        param = name_to_param.get(param_name)
        for key, tensor in state.items():
            state_keys.add(key)
            if key == "step":
                # step tensors might be shared between params, which safetensors doesn't like
                tensor = tensor.clone()
            else:
                tensor = wrap_tensor_for_sharded_parameter(tensor, param)
            flat_optim_state[f"state.{key}.{param_name}"] = tensor
    flat_optim_state["state_keys"] = serialize_to_tensor(list(state_keys))

    return flat_optim_state


def unflatten_optimizer_state(flat_optim_state: Dict[str, torch.Tensor]) -> OptimStateDict:
    num_param_groups = int(flat_optim_state["num_param_groups"].item())
    optim_state: OptimStateDict = {
        "state": {},
        "param_groups": [],
    }

    param_name_to_id: Dict[str, int] = {}

    # Deserialize param group data while collecting the mapping of param names to IDs.
    for i in range(num_param_groups):
        param_group = deserialize_from_tensor(flat_optim_state[f"param_group{i}"])
        param_names = param_group.pop("param_names")
        for param_name, param_id in zip(param_names, param_group["params"]):
            param_name_to_id[param_name] = param_id
        optim_state["param_groups"].append(param_group)

    # Unflatten the state tensors.
    state_keys = deserialize_from_tensor(flat_optim_state["state_keys"])
    for param_name, param_id in param_name_to_id.items():
        param_state: Dict[str, torch.Tensor] = {}
        for key in state_keys:
            state_tensor = flat_optim_state.get(f"state.{key}.{param_name}")
            if state_tensor is not None:
                # Ensure we have a regular tensor here, not some sharded wrapper.
                param_state[key] = get_local_tensor_data(state_tensor)

        optim_state["state"][param_id] = param_state

    return optim_state


def save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    model_state: Optional[Dict[str, torch.Tensor]] = None,
    optim_state: Optional[OptimStateDict] = None,
    save_overwrite: bool = False,
):
    """
    Save model and optimizer state dictionaries. The model state can be a sharded model, in which
    case this method will correctly handle the optimizer state to ensure it can be loaded again with
    a different distributed topology through :func:`load_model_and_optim_state()`.
    """
    dir = str(dir).rstrip("/")

    model_state = model_state or model.state_dict()
    flat_optim_state = flatten_optimizer_state(model, optim, model_state=model_state, optim_state=optim_state)

    checkpointer = Checkpointer()
    checkpointer.save(f"{dir}/model", model_state, save_overwrite=save_overwrite)
    checkpointer.save(f"{dir}/optim", flat_optim_state, save_overwrite=save_overwrite)


def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    model_state: Optional[Dict[str, torch.Tensor]] = None,
    optim_state: Optional[OptimStateDict] = None,
):
    """
    Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.
    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology.

    Note: You need to make sure your optimizer state is initialized before calling this.
    To do so, it suffices to call :func:`init_optimizer_state` before this.
    """
    dir = str(dir).rstrip("/")

    checkpointer = Checkpointer()

    # Load model state in-place.
    model_state = model_state or model.state_dict()
    checkpointer.load(f"{dir}/model", model_state)
    model.load_state_dict(model_state)

    # Load flattened optimizer state in-place.
    flat_optim_state = flatten_optimizer_state(model, optim, model_state=model_state, optim_state=optim_state)
    checkpointer.load(f"{dir}/optim", flat_optim_state)

    # Unflatten optimizer state.
    optim_state = unflatten_optimizer_state(flat_optim_state)
    optim.load_state_dict(optim_state)  # type: ignore


def init_optimizer_state(optim: torch.optim.Optimizer):
    """
    Ensure optimizer state is initialized.
    """
    for group in optim.param_groups:
        for p in group["params"]:
            p.grad = p.data.new(p.size()).zero_()
            p.grad.requires_grad_(False)
    optim.step()
    optim.zero_grad()
