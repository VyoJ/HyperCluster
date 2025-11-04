# Chat Template Fix - Preventing Infinite Generation

## Problem

The model was generating repetitive text and not stopping until hitting the max token limit (256 tokens). The generated text showed the model repeating itself without producing a proper EOS (End of Sequence) token.

### Symptoms
- Answer keeps repeating the same pattern
- Generation continues until `max_tokens` limit
- No EOS token detection in logs
- Text output shows only `?` characters or gibberish

## Root Cause

According to [HuggingFace Chat Templating Documentation](https://huggingface.co/docs/transformers/en/chat_templating), chat models require proper formatting with:

1. **Chat Templates**: Messages must be formatted using `apply_chat_template()` with proper role structures
2. **Generation Prompt**: `add_generation_prompt=True` adds tokens indicating where the assistant should respond
3. **Special Tokens**: Models like Qwen use control tokens like `<|im_start|>`, `<|im_end|>`, etc.

**Our Issue**: The code was using raw `tokenizer.encode()` instead of `apply_chat_template()`, so:
- No proper chat structure
- No indication to the model where to start responding
- No proper EOS token generation
- Model doesn't know when to stop

## The Fix

### 1. Modified `transformers_inference.py` - Encode Method

**Before:**
```python
def _encode():
    # Use simple encoding - chat template with add_generation_prompt=True
    # adds trailing newlines that cause the model to generate only newlines
    tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
    return np.array(tokens, dtype=np.int64)
```

**After:**
```python
def _encode():
    # Use chat template for proper formatting with control tokens
    # This ensures the model receives proper start/end tokens and knows when to stop
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template with add_generation_prompt=True
    # This adds the proper assistant response prompt tokens
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Now tokenize the formatted prompt
    tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
    
    logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
    return np.array(tokens, dtype=np.int64)
```

### 2. Enhanced EOS Token Detection in `ring_pipeline.py`

**Before:**
```python
eos_token_id = self.inference_engine.tokenizer.eos_token_id
if token_id == eos_token_id:
    logger.info(f"   🛑 EOS token ({eos_token_id}) detected, stopping generation")
    break
```

**After:**
```python
# Check for EOS - support multiple EOS tokens
eos_token_id = self.inference_engine.tokenizer.eos_token_id

# Some models have multiple EOS tokens (e.g., Qwen has both eos_token_id and special tokens)
eos_tokens = {eos_token_id}
if hasattr(self.inference_engine.tokenizer, 'eos_token_ids'):
    eos_tokens.update(self.inference_engine.tokenizer.eos_token_ids)

if token_id in eos_tokens:
    logger.info(f"   🛑 EOS token ({token_id}) detected, stopping generation")
    break
```

## What Changed

1. **Proper Message Formatting**: Queries are now wrapped in a chat message structure with `role: "user"`
2. **Chat Template Application**: Using `apply_chat_template()` with `add_generation_prompt=True`
3. **Special Tokens**: The formatted prompt now includes all necessary control tokens for the model
4. **Multiple EOS Support**: Enhanced detection for models that might have multiple EOS tokens

## Expected Behavior After Fix

When you run the same query now:

1. **Proper Encoding**: The prompt will be encoded with chat template tokens
   ```
   Input: "Is the Earth flat? Answer with yes or no"
   Formatted: "<|im_start|>user\nIs the Earth flat? Answer with yes or no<|im_end|>\n<|im_start|>assistant\n"
   ```

2. **Correct Generation**: Model will generate a proper response and emit an EOS token
   ```
   Generated: "No"
   EOS token detected: 151645
   ```

3. **Clean Termination**: Generation stops when EOS is detected, not just at max_tokens

## References

- [HuggingFace Chat Templating Guide](https://huggingface.co/docs/transformers/en/chat_templating)
- Key Points from Documentation:
  - Always use `apply_chat_template()` for chat models
  - Set `add_generation_prompt=True` to start assistant response
  - For training: use `add_generation_prompt=False`
  - Don't use `add_special_tokens=True` when tokenizing after `apply_chat_template(tokenize=False)`

## Testing

To verify the fix works:

```bash
# Start first node (head)
python main.py --ring

# In another terminal, start second node
python main.py --ring --bootstrap-ticket <ticket>

# Send a query
llm query Is the Earth flat? Answer with yes or no
```

Expected output:
- Proper chat-formatted prompt in logs
- Clean, concise answer (e.g., "No")
- EOS token detection message
- Generation stops well before 256 tokens

## Additional Notes

- This fix applies to all chat-instruct models (Qwen, Llama, Mistral, etc.)
- Base models (non-chat) should still work as they don't have chat templates
- The `add_generation_prompt=True` is critical for inference but should be `False` for training
