# taifoon-nemotron

Taifoon Intel Agent \u2014 QLoRA fine-tuned on **Nemotron-3-Nano-4B** (NVIDIA Mamba-Transformer hybrid)

## Model
- Base: 
- Fine-tune: QLoRA (r=32, \u03b1=64, NF4 4-bit, 3 epochs SFT)
- Dataset: 1149 protocol-enriched v2 records (24 protocols, 25 agents, V5 proof system, razor economics)

## Quick Start



## Environment
- GPU: RTX 4000 Ada (20GB VRAM)
- PyTorch 2.6.0+cu124
- mamba-ssm 2.3.1 (pre-built wheel + ABI shim for libtorch symbol compatibility)

## ABI Fix
PyTorch 2.6+cu124 exports  with old-ABI ,
but the mamba-ssm pre-built wheel expects new-ABI .
 compiles  to bridge the gap.
Use  for all mamba-ssm operations.

