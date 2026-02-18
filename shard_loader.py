"""
Direct partial model loader for memory-efficient shard loading.

This module implements selective weight loading that only loads the layers
needed for a specific shard, never allocating memory for unused layers.
This is critical for memory-constrained distributed inference.

Approach:
1. Create empty model scaffold with init_empty_weights() - no memory allocated
2. Identify which weights are needed for the shard
3. Load only those weights from safetensors using selective loading
4. Use set_module_tensor_to_device() to populate weights
5. Prune model structure to only keep loaded layers
"""

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_files(
    model_path: str, cache_dir: Optional[str] = None
) -> Tuple[Path, Optional[Dict]]:
    """
    Get the safetensors file(s) and weight map for a model.

    Returns:
        Tuple of (model_dir, weight_map)
        - weight_map is None for single-file models
        - weight_map is dict mapping weight_name -> filename for sharded models
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    # Check if it's a local path
    if os.path.isdir(model_path):
        model_dir = Path(model_path)
    else:
        # Download from HuggingFace Hub
        # First try to get index file for sharded models
        try:
            index_file = hf_hub_download(
                model_path,
                filename="model.safetensors.index.json",
                cache_dir=cache_dir,
            )
            model_dir = Path(index_file).parent

            with open(index_file, "r") as f:
                index_data = json.load(f)

            return model_dir, index_data.get("weight_map", {})

        except EntryNotFoundError:
            # Single file model
            pass

        # Download single safetensors file
        try:
            safetensors_file = hf_hub_download(
                model_path,
                filename="model.safetensors",
                cache_dir=cache_dir,
            )
            model_dir = Path(safetensors_file).parent
            return model_dir, None

        except EntryNotFoundError:
            raise ValueError(
                f"Could not find model.safetensors or model.safetensors.index.json for {model_path}"
            )

    # Local path - check for index file
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index_data = json.load(f)
        return model_dir, index_data.get("weight_map", {})

    # Single file
    if (model_dir / "model.safetensors").exists():
        return model_dir, None

    raise ValueError(f"Could not find safetensors files in {model_dir}")


def _get_layer_prefix(config) -> str:
    """Get the layer prefix based on model type."""
    model_type = config.model_type.lower()

    # Most models use "model.layers"
    if model_type in ["llama", "mistral", "qwen2", "qwen", "mixtral", "phi"]:
        return "model.layers"
    elif model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        return "transformer.h"
    else:
        # Default fallback
        return "model.layers"


def _get_needed_weight_keys(shard, config) -> Set[str]:
    """
    Determine which state_dict keys are needed for this shard.

    Args:
        shard: Shard specification
        config: Model configuration

    Returns:
        Set of weight key prefixes to load
    """
    needed = set()
    layer_prefix = _get_layer_prefix(config)
    model_type = config.model_type.lower()

    # Determine the model prefix (varies by architecture)
    if model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        model_prefix = "transformer"
        embed_key = "transformer.wte"
        norm_key = "transformer.ln_f"
        head_key = "lm_head"
    else:
        # Llama-style models (most common)
        model_prefix = "model"
        embed_key = "model.embed_tokens"
        norm_key = "model.norm"
        head_key = "lm_head"

    # Embeddings (only first shard)
    if shard.is_first_layer():
        needed.add(embed_key)
        logger.info(f"   Shard needs embeddings: {embed_key}")

    # Layers in our shard range
    for layer_idx in range(shard.start_layer, shard.end_layer + 1):
        layer_key = f"{layer_prefix}.{layer_idx}"
        needed.add(layer_key)
    logger.info(f"   Shard needs layers: {shard.start_layer} to {shard.end_layer}")

    # Rotary embeddings (if present, needed by all shards for position encoding)
    if hasattr(config, "rope_scaling") or model_type in [
        "llama",
        "mistral",
        "qwen2",
        "qwen",
    ]:
        needed.add(f"{model_prefix}.rotary_emb")

    # Final norm and lm_head (only last shard)
    if shard.is_last_layer():
        needed.add(norm_key)
        needed.add(head_key)
        logger.info(f"   Shard needs final norm: {norm_key}")
        logger.info(f"   Shard needs lm_head: {head_key}")

    return needed


def _should_load_key(key: str, needed_prefixes: Set[str]) -> bool:
    """Check if a weight key should be loaded based on needed prefixes."""
    for prefix in needed_prefixes:
        if key.startswith(prefix):
            return True
    return False


def _prune_model_structure(model, shard, config):
    """
    Prune the model structure to only contain the loaded layers.
    This removes references to unloaded layers from the ModuleList.
    """
    model_type = config.model_type.lower()

    # Get the layers module
    if model_type in ["gpt2", "gpt_neo", "gpt_neox"]:
        inner_model = model.transformer
        layers_attr = "h"
    else:
        inner_model = model.model
        layers_attr = "layers"

    all_layers = getattr(inner_model, layers_attr)

    # Extract only the layers we need
    needed_layers = [
        all_layers[i] for i in range(shard.start_layer, shard.end_layer + 1)
    ]

    # Re-index layer_idx on attention modules for KV cache
    for new_idx, layer in enumerate(needed_layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            old_idx = layer.self_attn.layer_idx
            layer.self_attn.layer_idx = new_idx
            if new_idx == 0 or new_idx == len(needed_layers) - 1:
                logger.debug(f"   Re-indexed layer {old_idx} → cache index {new_idx}")

    # Replace the layers ModuleList with only the needed layers
    setattr(inner_model, layers_attr, nn.ModuleList(needed_layers))

    # Handle embeddings for non-first shards
    if not shard.is_first_layer():
        if hasattr(inner_model, "embed_tokens"):
            # Set to None but keep attribute for compatibility
            inner_model.embed_tokens = None
        elif hasattr(inner_model, "wte"):
            inner_model.wte = None

    # Handle lm_head for non-last shards
    if not shard.is_last_layer():
        if hasattr(model, "lm_head"):
            model.lm_head = None
        if hasattr(inner_model, "norm"):
            inner_model.norm = None
        elif hasattr(inner_model, "ln_f"):
            inner_model.ln_f = None

    logger.info(f"   Pruned model: {len(needed_layers)} layers kept")


def load_shard_direct(
    model_id: str,
    shard,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any]:
    """
    Load ONLY the layers needed for a shard directly from disk.
    Never allocates memory for unused layers.

    Args:
        model_id: HuggingFace model ID or local path
        shard: Shard specification defining which layers to load
        cache_dir: Directory to cache downloaded models
        device: Target device for loaded weights
        dtype: Target dtype for loaded weights

    Returns:
        Tuple of (model, config)
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info(f"🚀 Direct partial loading for shard {shard}")
    logger.info(f"   Model: {model_id}")
    logger.info(f"   Device: {device}, dtype: {dtype}")

    # 1. Load configuration
    config = AutoConfig.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True
    )

    logger.info("📋 Model Config:")
    logger.info(f"   Model type: {config.model_type}")
    logger.info(f"   Hidden size: {config.hidden_size}")
    logger.info(f"   Num layers: {config.num_hidden_layers}")
    logger.info(f"   Vocab size: {config.vocab_size}")

    # 2. Get model files and weight map
    model_dir, weight_map = _get_model_files(model_id, cache_dir)
    logger.info(f"   Model directory: {model_dir}")
    logger.info(f"   Sharded: {weight_map is not None}")

    # 3. Determine which weights we need
    needed_prefixes = _get_needed_weight_keys(shard, config)

    # 4. Create empty model scaffold (no memory allocated)
    logger.info("📝 Creating empty model scaffold...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # 5. Load weights selectively
    logger.info("📦 Loading weights selectively...")
    loaded_count = 0
    skipped_count = 0

    if weight_map is None:
        # Single safetensors file
        safetensors_path = model_dir / "model.safetensors"
        logger.info(f"   Loading from single file: {safetensors_path.name}")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _should_load_key(key, needed_prefixes):
                    tensor = f.get_tensor(key)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    set_module_tensor_to_device(model, key, device, value=tensor)
                    loaded_count += 1
                else:
                    skipped_count += 1
    else:
        # Sharded safetensors files
        # Group keys by file to minimize file opens
        files_to_load: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            if _should_load_key(key, needed_prefixes):
                files_to_load.setdefault(filename, []).append(key)
            else:
                skipped_count += 1

        logger.info(f"   Loading from {len(files_to_load)} shard files")

        for filename, keys in files_to_load.items():
            shard_path = model_dir / filename
            logger.debug(f"   Loading {len(keys)} keys from {filename}")

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    set_module_tensor_to_device(model, key, device, value=tensor)
                    loaded_count += 1

    logger.info(f"   Loaded {loaded_count} tensors, skipped {skipped_count}")

    # 6. Prune model structure
    logger.info("✂️ Pruning model structure...")
    _prune_model_structure(model, shard, config)

    # 7. Patch config for this shard (for DynamicCache compatibility)
    # CRITICAL: Patch BOTH the standalone config AND the model's internal config
    # The model uses model.config internally for attention mask generation
    shard_layer_count = shard.end_layer - shard.start_layer + 1
    original_layers = config.num_hidden_layers
    config.num_hidden_layers = shard_layer_count
    model.config.num_hidden_layers = shard_layer_count
    logger.info(
        f"   Patched config: num_hidden_layers={shard_layer_count} (was {original_layers})"
    )

    # 8. Set to eval mode and optionally tie weights
    model.eval()
    # Only tie weights if we have BOTH embed_tokens and lm_head (full model or single-shard case)
    if shard.is_first_layer() and shard.is_last_layer():
        model.tie_weights()

    # 9. Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("✅ Shard loaded successfully!")

    return model, config


def estimate_shard_memory(shard, config, dtype: torch.dtype = torch.float16) -> int:
    """
    Estimate memory required for a shard in bytes.

    This is useful for checking if a shard will fit in available memory
    before attempting to load it.
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)

    hidden_size = config.hidden_size
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    vocab_size = config.vocab_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    # Per-layer parameters (approximate)
    # Self attention: Q, K, V, O projections
    attn_params = hidden_size * (
        num_heads * head_dim + 2 * num_kv_heads * head_dim + hidden_size
    )
    # MLP: gate, up, down projections
    mlp_params = hidden_size * intermediate_size * 3
    # Layer norms
    norm_params = hidden_size * 2

    layer_params = attn_params + mlp_params + norm_params

    # Shard-specific params
    shard_layers = shard.end_layer - shard.start_layer + 1
    total_params = shard_layers * layer_params

    # Add embeddings for first shard
    if shard.is_first_layer():
        total_params += vocab_size * hidden_size

    # Add lm_head for last shard (often tied to embeddings)
    if shard.is_last_layer():
        if not getattr(config, "tie_word_embeddings", True):
            total_params += vocab_size * hidden_size
        total_params += hidden_size  # final norm

    return total_params * bytes_per_param
