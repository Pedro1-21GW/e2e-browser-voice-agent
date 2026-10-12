"""Reference outputs for the M1 language model, from Hugging Face Transformers (an independent implementation).

Loads the same GGUF file the browser engine reads (Transformers dequantizes it to fp32), runs greedy decoding on a
few chat prompts, and writes token ids, the greedy continuation and the top-5 logits of the first step to
reference.json. The engine's check compares against this file.

Usage:  python h2-own-inference-engine/m1-llm/reference.py
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO, FILE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_0.gguf"
OUT = os.path.join(os.path.dirname(__file__), "reference.json")
STEPS = 32

SYSTEM = "You are a helpful voice assistant. Answer in one or two short sentences."
CHATS = [
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the capital of France?"}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Give me one tip to sleep better."}],
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "How many legs does an octopus have?"}],
]
# Strings that exercise the tokenizer: spaces, digits, contractions, accents, emoji, code, newlines.
TOKENIZER_TESTS = [
    "Hello, world!", "  two  spaces and\ttab", "I'm sure it's 12345 or 3.14159...", "São Paulo é ótimo, não é?",
    "emoji 🙂👍 and 日本語", "def f(x):\n    return x**2\n\n\nprint(f(3))", "<|im_start|>user\nhi<|im_end|>\n",
    "UPPER lower MiXeD 1st 2nd 3rd", "trailing spaces   ", "line\r\nbreaks\n\n",
]

torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(REPO, gguf_file=FILE)
model = AutoModelForCausalLM.from_pretrained(REPO, gguf_file=FILE, torch_dtype=torch.float32).eval()

out = {"model": f"{REPO}/{FILE}", "steps": STEPS, "chats": [], "tokenizer": []}
for text in TOKENIZER_TESTS:
    out["tokenizer"].append({"text": text, "ids": tok.encode(text, add_special_tokens=False)})

for chat in CHATS:
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    t = time.time()
    first = model(torch.tensor([ids])).logits[0, -1]
    top = torch.topk(first, 5)
    gen = model.generate(torch.tensor([ids]), max_new_tokens=STEPS, do_sample=False, attention_mask=torch.ones(1, len(ids), dtype=torch.long))
    new = gen[0, len(ids):].tolist()
    out["chats"].append({
        "messages": chat, "prompt": prompt, "prompt_ids": ids,
        "top5_ids": top.indices.tolist(), "top5_logits": [round(v, 4) for v in top.values.tolist()],
        "greedy_ids": new, "greedy_text": tok.decode(new, skip_special_tokens=False),
    })
    print(f"{len(ids)} prompt tokens -> {tok.decode(new, skip_special_tokens=True)!r}  ({time.time() - t:.1f} s)")

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT)
