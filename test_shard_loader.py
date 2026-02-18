"""
Test script for direct partial shard loading.
"""

import torch
from shard import Shard
from shard_loader import estimate_shard_memory, load_shard_direct


def test_first_shard():
    """Test loading the first half of a model (with embeddings)."""
    print("=" * 60)
    print("TEST: First shard (layers 0-7 of 16)")
    print("=" * 60)

    shard = Shard(
        "meta-llama/Llama-3.2-1B-Instruct", start_layer=0, end_layer=7, n_layers=16
    )
    print(f"Shard: {shard}")

    # Estimate memory
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", cache_dir="./model_cache"
    )
    mem = estimate_shard_memory(shard, config)
    print(f"Estimated memory: {mem / 1024**2:.1f} MB")

    # Load shard
    print("Loading shard...")
    model, config = load_shard_direct(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        shard=shard,
        cache_dir="./model_cache",
        device="cpu",
        dtype=None,  # float32 for testing
    )

    # Check model structure
    inner = model.model
    print(f"Layers in model: {len(inner.layers)}")
    print(f"Has embed_tokens: {inner.embed_tokens is not None}")
    print(f"Has norm: {inner.norm is not None}")
    print(f"Has lm_head: {model.lm_head is not None}")

    # Count actual parameters (exclude meta device)
    meta_device = torch.device("meta")
    total_params = sum(p.numel() for p in model.parameters() if p.device != meta_device)
    print(f"Total loaded parameters: {total_params:,}")
    print(f"Loaded memory: {total_params * 4 / 1024**2:.1f} MB (float32)")

    # Verify first shard has embeddings
    assert inner.embed_tokens is not None, "First shard should have embeddings"
    assert inner.norm is None, "First shard should NOT have norm"
    assert model.lm_head is None, "First shard should NOT have lm_head"
    assert len(inner.layers) == 8, f"Expected 8 layers, got {len(inner.layers)}"

    print("FIRST SHARD TEST PASSED!")
    return model


def test_last_shard():
    """Test loading the last half of a model (with lm_head)."""
    print("\n" + "=" * 60)
    print("TEST: Last shard (layers 8-15 of 16)")
    print("=" * 60)

    shard = Shard(
        "meta-llama/Llama-3.2-1B-Instruct", start_layer=8, end_layer=15, n_layers=16
    )
    print(f"Shard: {shard}")

    # Estimate memory
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", cache_dir="./model_cache"
    )
    mem = estimate_shard_memory(shard, config)
    print(f"Estimated memory: {mem / 1024**2:.1f} MB")

    # Load shard
    print("Loading shard...")
    model, config = load_shard_direct(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        shard=shard,
        cache_dir="./model_cache",
        device="cpu",
        dtype=None,
    )

    # Check model structure
    inner = model.model
    print(f"Layers in model: {len(inner.layers)}")
    print(f"Has embed_tokens: {inner.embed_tokens is not None}")
    print(f"Has norm: {inner.norm is not None}")
    print(f"Has lm_head: {model.lm_head is not None}")

    # Count actual parameters
    meta_device = torch.device("meta")
    total_params = sum(p.numel() for p in model.parameters() if p.device != meta_device)
    print(f"Total loaded parameters: {total_params:,}")
    print(f"Loaded memory: {total_params * 4 / 1024**2:.1f} MB (float32)")

    # Verify last shard has lm_head
    assert inner.embed_tokens is None, "Last shard should NOT have embeddings"
    assert inner.norm is not None, "Last shard should have norm"
    assert model.lm_head is not None, "Last shard should have lm_head"
    assert len(inner.layers) == 8, f"Expected 8 layers, got {len(inner.layers)}"

    print("LAST SHARD TEST PASSED!")
    return model


def test_wrapper():
    """Test that TransformersShard wrapper works with pre-pruned model."""
    print("\n" + "=" * 60)
    print("TEST: TransformersShard wrapper with pre-pruned model")
    print("=" * 60)

    from shard_loader import load_shard_direct
    from sharded_model import TransformersShard

    shard = Shard(
        "meta-llama/Llama-3.2-1B-Instruct", start_layer=0, end_layer=7, n_layers=16
    )

    model, config = load_shard_direct(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        shard=shard,
        cache_dir="./model_cache",
        device="cpu",
        dtype=None,
    )

    # Create wrapper
    wrapped = TransformersShard(model, shard, pre_pruned=True)

    print(f"Wrapper shard: {wrapped.shard}")
    print(f"Wrapper layers: {len(wrapped.layers)}")
    print(f"Wrapper has embed_tokens: {wrapped.embed_tokens is not None}")
    print(f"Wrapper has norm: {wrapped.norm is not None}")
    print(f"Wrapper has lm_head: {wrapped.lm_head is not None}")
    print(f"Wrapper config num_hidden_layers: {wrapped.config.num_hidden_layers}")

    # Verify wrapper config is patched
    assert wrapped.config.num_hidden_layers == 8, "Config should be patched to 8 layers"

    print("WRAPPER TEST PASSED!")


def test_forward_pass():
    """Test that we can actually do a forward pass through a shard."""
    print("\n" + "=" * 60)
    print("TEST: Forward pass through first shard")
    print("=" * 60)

    from shard_loader import load_shard_direct
    from sharded_model import TransformersShard

    shard = Shard(
        "meta-llama/Llama-3.2-1B-Instruct", start_layer=0, end_layer=7, n_layers=16
    )

    model, config = load_shard_direct(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        shard=shard,
        cache_dir="./model_cache",
        device="cpu",
        dtype=torch.float32,  # Use float32 for CPU
    )

    # Create wrapper
    wrapped = TransformersShard(model, shard, pre_pruned=True)

    # Verify model.config is patched
    print(f"model.config.num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"wrapped.config.num_hidden_layers: {wrapped.config.num_hidden_layers}")
    assert model.config.num_hidden_layers == 8, (
        "Model config should be patched to 8 layers"
    )

    # Create test input - process prefix first
    batch_size = 1
    prefix_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, prefix_len))

    print(f"Prefix input shape: {input_ids.shape}")

    # Initial forward pass (prefix)
    with torch.no_grad():
        cache_position = torch.arange(prefix_len, dtype=torch.long)
        output = wrapped(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )

    print(f"After prefix - Output type: {type(output)}")
    pkv = output.past_key_values
    print(f"After prefix - KV cache layers: {len(pkv.key_cache)}")
    print(f"After prefix - KV cache seq_len: {pkv.get_seq_length()}")

    # Now do incremental decoding (single token)
    print("\nTesting incremental decoding...")
    next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
    cache_position = torch.tensor([prefix_len], dtype=torch.long)

    with torch.no_grad():
        output2 = wrapped(
            input_ids=next_token,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )

    pkv2 = output2.past_key_values
    print(f"After 1 token - KV cache seq_len: {pkv2.get_seq_length()}")

    # One more token
    cache_position = torch.tensor([prefix_len + 1], dtype=torch.long)
    next_token = torch.randint(0, config.vocab_size, (batch_size, 1))

    with torch.no_grad():
        output3 = wrapped(
            input_ids=next_token,
            past_key_values=pkv2,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )

    pkv3 = output3.past_key_values
    print(f"After 2 tokens - KV cache seq_len: {pkv3.get_seq_length()}")

    assert pkv3.get_seq_length() == prefix_len + 2, (
        f"Expected {prefix_len + 2} cached tokens"
    )

    print("FORWARD PASS TEST PASSED!")


if __name__ == "__main__":
    test_first_shard()
    test_last_shard()
    test_wrapper()
    test_forward_pass()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
