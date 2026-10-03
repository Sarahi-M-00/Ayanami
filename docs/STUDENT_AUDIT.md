# STUDENT_AUDIT.md — Qwen3-1.7B-heretic snapshot audit
Audited: 2026-10-03T00:25:55+00:00
Snapshot: `student_base/` (read-only copy of /home/ling/Ayanami-AI/)

## 1. File list (size + SHA256)

| file | size_bytes | sha256 |
| --- | --- | --- |
| .gitattributes | 1570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |
| README.md | 2213 | `943f58b22cbb80d0f7f77f441f398475bf11150b5bc2655959c28be4a523b72f` |
| chat_template.jinja | 4168 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| config.json | 1417 | `db21e5529322523f48341a67b677e772ab7818e06226e145e8cec40b22197269` |
| generation_config.json | 214 | `893b0dccf83626cdfdc498e5b2a255a935b4c3c693350aed18f7003c1a3f3de9` |
| model.safetensors | 3441185608 | `e4600a43e0655ee8af35d30af78538f47c97b2936951cdb0d00fc9c1f9389737` |
| tokenizer.json | 11422817 | `79cb3c783570f1b8fe73b9ed530ae50cae9ce4b6344c0b5edefc50478847eaa4` |
| tokenizer_config.json | 694 | `04b1682c59acbd057f4c9072297faa73d56fc9de053094c659cdb4c464f58f86` |

Extra files in snapshot: none.
Safetensors shards: single file `model.safetensors`, no `.index.json` — nothing to cross-check.

## 2. Safetensors header

Tensors: 310. Total parameters: 1,720,574,976 (1.7206 B).
Dtype distribution: {'BF16': 310}. Header `__metadata__`: {'format': 'pt'}.
Layer indices present: 0..27 consecutive: True (28 layers).
`lm_head.weight` in file: False (tied embeddings expected: True).

## 3. NaN/Inf scan (one tensor at a time, float32)

Tensors with NaN: none. Tensors with Inf: none.

## 4. config.json (real values)

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 1000000,
    "rope_type": "default"
  },
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "5.17.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

Summary: architecture `Qwen3ForCausalLM`, model_type `qwen3`, 28 layers (all `full_attention`), hidden_size 2048, 16 attention heads / 8 KV heads, head_dim 128, intermediate_size 6144, vocab_size 151936, tie_word_embeddings true, max_position_embeddings 40960, rope_theta 1000000 (rope_type default), rms_norm_eps 1e-06, hidden_act silu, attention_bias false, dtype bfloat16, bos_token_id 151643, eos_token_id 151645, pad_token_id null, written by transformers 5.17.0.
VERIFY long context (32k class): CONFIRMED-ish — max_position_embeddings=40960 (40k, above 32k); tokenizer model_max_length=131072. Usable context is 40960 per config.
VERIFY transformers>=4.51: CONFIRMED — snapshot targets transformers 5.x; audit runs 5.18.0.

## 5. generation_config.json (real values)

```json
{
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95,
  "transformers_version": "5.17.0"
}
```

Summary: sampling on (temperature 0.6, top_k 20, top_p 0.95), eos_token_id [151645, 151643], bos/pad 151643.

## 6. Tokenizer

tokenizer_class: Qwen2Tokenizer (fast). Base vocab entries: 151643. Added tokens: 26 (ids 151643..151668).
Special token map: {151643: '<|endoftext|>', 151644: '<|im_start|>', 151645: '<|im_end|>', 151646: '<|object_ref_start|>', 151647: '<|object_ref_end|>', 151648: '<|box_start|>', 151649: '<|box_end|>', 151650: '<|quad_start|>', 151651: '<|quad_end|>', 151652: '<|vision_start|>', 151653: '<|vision_end|>', 151654: '<|vision_pad|>', 151655: '<|image_pad|>', 151656: '<|video_pad|>', 151657: '<tool_call>', 151658: '</tool_call>', 151659: '<|fim_prefix|>', 151660: '<|fim_middle|>', 151661: '<|fim_suffix|>', 151662: '<|fim_pad|>', 151663: '<|repo_name|>', 151664: '<|file_sep|>', 151665: '<tool_response>', 151666: '</tool_response>', 151667: '<think>', 151668: '</think>'}.
tokenizer_config: eos_token `<|im_end|>` (id 151645), pad_token `<|endoftext|>` (id 151643), bos_token null, model_max_length 131072, split_special_tokens false.

Tokenizer fingerprint (SHA256 of sorted vocab + added tokens): `563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa`
This fingerprint is the Phase 4 key for teacher/student logit compatibility.

## 7. Chat template renders

### (a) plain chat
```
<|im_start|>user
Explain what DNS is.<|im_end|>
<|im_start|>assistant

```
### (b) chat with system message
```
<|im_start|>system
You are a concise technical assistant.<|im_end|>
<|im_start|>user
Explain what DNS is.<|im_end|>
<|im_start|>assistant

```
### (c) chat with tools + tool response
```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_status", "description": "Read service status.", "parameters": {"type": "object", "properties": {}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Is the web service up?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_status", "arguments": {}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
active (running)
</tool_response><|im_end|>
<|im_start|>assistant

```
Template is standard Qwen3 ChatML (`<|im_start|>`/`<|im_end|>`), Hermes-style `<tool_call>`/` <tool_response>`, thinking via `<think>` blocks or `reasoning_content`.
Thinking on (default) ends with: `'DNS is.<|im_end|>\n<|im_start|>assistant\n'`
Thinking off (`enable_thinking=False`) appends an empty think block: True.

## 8. Model card claims (from README.md) vs snapshot

- Lineage Qwen3-1.7B-Base -> Qwen3-1.7B -> heretic abliteration (Heretic tool, 25 Optuna trials, trial 19 selected): from card, not verifiable in weight files. TAKEN AS CARD CLAIM.
- Refusals 3/100 (was 92/100), KL 0.0566 vs original: from card table. NOT verifiable offline.
- Languages en+es: from card front-matter. NOT verifiable in files (no language id in config).
- BF16 safetensors ~2B params: CONFIRMED — dtype bfloat16, 1,720,574,976 params counted in header, 3441185608 bytes on disk.
- License apache-2.0, base_model Qwen/Qwen3-1.7B: from card front-matter.

Audit wall time: 18.7s.

## 9. Smoke test and baseline-0

Load: 1.5s BF16 on CPU (6 threads), peak RSS 509.2 MB after load, 3765.7 MB at end.
Generation: 20 prompts (10 en + 10 es), greedy, max 64 new tokens, thinking disabled. Total 566 tokens in 149.7s -> mean 3.78 tok/s.
Suspect template behavior (model emitting turn markers or tool tags unprompted): none observed in these 20 short prompts.
Full prompts/outputs: `runs/baseline0/prompts.jsonl`, `runs/baseline0/outputs.jsonl`, `runs/baseline0/metrics.json`. This is the regression reference for all later phases.
