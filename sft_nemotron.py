#!/usr/bin/env python3
"""Taifoon Nemotron-3-Nano-4B QLoRA fine-tuning — v2 protocol dataset

Final approach (v7): load in FULL bf16 (no bitsandbytes quantization).
Ollama is evicted from GPU before training and restored after.

Why bf16 instead of 4-bit:
  NemotronH is 90% Mamba2 SSM blocks. With bitsandbytes 4-bit, the Mamba2
  forward kernel produces logits with grad_fn=None (confirmed via diagnostic:
  loss.grad_fn=None, logits.grad_fn=None) — the entire computation graph is
  detached. This affects both the triton kernel path (use_mamba_kernels=True)
  AND the pure Python path (use_mamba_kernels=False). Root cause: Mamba2's
  selective scan operations go through non-standard paths that interact badly
  with bitsandbytes' quantized weight storage.

  Solution: load in bf16 (full precision for all weights), which lets autograd
  track the full computation graph normally. LoRA adapters are added on top.
  Requires ~8GB base model + 3-4GB activations = ~12GB. Works on 21GB GPU
  once Ollama's 9.5GB is freed.

Run with: python3 sft_nemotron.py
(no LD_PRELOAD needed — bf16 doesn't require mamba-ssm triton extensions)
"""
import os, sys, json, torch, warnings, logging, subprocess, time
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID  = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_PATH = "/root/taifoon-training/taifoon_intel_v2.jsonl"
CKPT_DIR  = "/root/taifoon-training/nemotron-checkpoints"
LORA_R=32; LORA_ALPHA=64; LORA_DROPOUT=0.05
EPOCHS=3; BS=1; GRAD=16; LR=2e-4; WARMUP=50; MAX_SEQ=1024

log.info("=== Taifoon Nemotron-3-Nano-4B LoRA SFT v7 (bf16, no bnb) ===")
log.info(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
free0, total = torch.cuda.mem_get_info(0)
log.info(f"GPU free at start: {free0/1e9:.1f}GB / {total/1e9:.1f}GB")

# === STEP 1: Evict Ollama from GPU ===
log.info("Evicting Ollama model from GPU (keep_alive=0)...")
try:
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": "nemotron-3-nano:4b", "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        log.info(f"Ollama evict response: {resp.status}")
except Exception as e:
    log.warning(f"Ollama evict request failed (may already be unloaded): {e}")

# Wait for VRAM to free up
time.sleep(5)
free1, _ = torch.cuda.mem_get_info(0)
log.info(f"GPU free after evict: {free1/1e9:.1f}GB")

# === STEP 2: Load dataset ===
records = [json.loads(l) for l in open(DATA_PATH)]
TMPL = "<|system|>\n{s}\n<|user|>\n{u}\n<|assistant|>\n{a}"
def fmt(r):
    msgs = r["messages"]
    s = next((m["content"] for m in msgs if m["role"]=="system"), "")
    u = next((m["content"] for m in msgs if m["role"]=="user"), "")
    a = next((m["content"] for m in msgs if m["role"]=="assistant"), "")
    return {"text": TMPL.format(s=s, u=u, a=a)}
data  = Dataset.from_list([fmt(r) for r in records])
split = data.train_test_split(test_size=0.05, seed=42)
log.info(f"Dataset: {len(split['train'])} train / {len(split['test'])} val")

# === STEP 3: Load model in bf16 (no quantization) ===
log.info("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

log.info("Loading Nemotron-3-Nano-4B in bf16 (full precision, no bitsandbytes)...")
# NOTE: LD_PRELOAD=/root/taifoon-training/libtorch_shim.so is required even
# in bf16 mode because NemotronH imports causal_conv1d_cuda.so at module load
# time (for use_mamba_kernels=True), and that .so has the same ABI mismatch:
# undefined symbol: torchCheckFail(...std::__cxx11::basic_string...)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
model.config.use_cache = False
log.info(f"Loaded: {model.__class__.__name__}")

free2, _ = torch.cuda.mem_get_info(0)
log.info(f"GPU free after model load: {free2/1e9:.1f}GB")

# === STEP 4: LoRA config (attention layers only) ===
# Skip Mamba-specific in_proj/out_proj — target only transformer attention/MLP
lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    task_type=TaskType.CAUSAL_LM, bias="none")

# === STEP 5: Verify grad flow before training ===
from peft import get_peft_model
log.info("Wrapping with LoRA...")
model = get_peft_model(model, lora_cfg)
model.enable_input_require_grads()
model.print_trainable_parameters()

# Quick grad check
model.train()
_inp = torch.randint(0, tok.vocab_size, (1, 16), device=model.device)
with torch.enable_grad():
    _out = model(_inp, labels=_inp)
if _out.loss.grad_fn is None:
    log.error(f"FATAL: loss.grad_fn=None even in bf16 mode. logits.grad_fn={_out.logits.grad_fn}")
    sys.exit(1)
log.info(f"Grad check PASSED: loss={_out.loss.item():.4f}, grad_fn={_out.loss.grad_fn}")
del _inp, _out

# === STEP 6: Train ===
os.makedirs(CKPT_DIR, exist_ok=True)
args = SFTConfig(
    output_dir=CKPT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GRAD,
    learning_rate=LR, warmup_steps=WARMUP, optim="paged_adamw_8bit", bf16=True,
    logging_steps=25, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, report_to="none", dataloader_num_workers=0,
    max_length=MAX_SEQ, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})

# Pass pre-wrapped PEFT model (enable_input_require_grads already called)
trainer = SFTTrainer(
    model=model, processing_class=tok,
    train_dataset=split["train"], eval_dataset=split["test"],
    args=args)

log.info("Starting SFT (3 epochs)...")
result = trainer.train()
log.info(f"Done. Loss: {result.training_loss:.4f}")
trainer.save_model(CKPT_DIR + "/final")
tok.save_pretrained(CKPT_DIR + "/final")
log.info(f"Saved to {CKPT_DIR}/final")

# === STEP 7: Restore Ollama ===
log.info("Restoring nemotron-3-nano:4b in Ollama GPU memory (keep_alive=-1)...")
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": "nemotron-3-nano:4b", "keep_alive": -1,
                         "prompt": "ping", "stream": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        log.info(f"Ollama restore response: {resp.status}")
except Exception as e:
    log.warning(f"Ollama restore failed (restore manually): {e}")

log.info("=== TRAINING COMPLETE ===")
