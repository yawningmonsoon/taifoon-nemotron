#!/usr/bin/env python3
"""Taifoon Nemotron-3-Nano-4B QLoRA fine-tuning — v2 protocol dataset
Run with: LD_PRELOAD=/root/taifoon-training/libtorch_shim.so python3 sft_nemotron.py
"""
import os, sys, json, torch, warnings, logging
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID  = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_PATH = "/root/taifoon-training/taifoon_intel_v2.jsonl"
CKPT_DIR  = "/root/taifoon-training/nemotron-checkpoints"
LORA_R=32; LORA_ALPHA=64; LORA_DROPOUT=0.05
EPOCHS=3; BS=2; GRAD=8; LR=2e-4; WARMUP=50; MAX_SEQ=2048

log.info("=== Taifoon Nemotron-3-Nano-4B QLoRA SFT ===")
log.info(f"GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

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

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj","in_proj","out_proj"],
    task_type=TaskType.CAUSAL_LM, bias="none")

log.info("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

log.info("Loading Nemotron-3-Nano-4B in NF4 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb,
    device_map="auto", trust_remote_code=True)
model.config.use_cache = False
log.info(f"Loaded: {model.__class__.__name__}")

os.makedirs(CKPT_DIR, exist_ok=True)
args = SFTConfig(
    output_dir=CKPT_DIR, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BS, gradient_accumulation_steps=GRAD,
    learning_rate=LR, warmup_steps=WARMUP, optim="paged_adamw_8bit", bf16=True,
    logging_steps=25, eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, report_to="none", dataloader_num_workers=0,
    max_length=MAX_SEQ)

trainer = SFTTrainer(
    model=model, processing_class=tok,
    train_dataset=split["train"], eval_dataset=split["test"],
    args=args, peft_config=lora)

log.info("Starting SFT (3 epochs)...")
result = trainer.train()
log.info(f"Done. Loss: {result.training_loss:.4f}")
trainer.save_model(CKPT_DIR + "/final")
tok.save_pretrained(CKPT_DIR + "/final")
log.info(f"Saved to {CKPT_DIR}/final")
log.info("=== TRAINING COMPLETE ===")
