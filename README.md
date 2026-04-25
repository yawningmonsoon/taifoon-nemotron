> ## ⚠️ This is the **code** repo — for docs, conventions, and the agent client, see  
> ### → [yawningmonsoon/taifoon-intel-platform](https://github.com/yawningmonsoon/taifoon-intel-platform)
>
> If you are an autonomous agent (trader, polymarket helper, scanner, etc.), read [CONVENTIONS.md](https://github.com/yawningmonsoon/taifoon-intel-platform/blob/main/CONVENTIONS.md) before touching this stack.

# taifoon-nemotron — Unified Intel API

Three QLoRA adapters served on ONE CUDA device with ONE base model (NVIDIA Nemotron-3-Nano-4B-BF16):

| Slug         | Adapter path                                             | Persona                                  |
|--------------|----------------------------------------------------------|------------------------------------------|
| `taifoon`    | `/root/taifoon-nemotron/adapters/final`                  | protocol intel, V5 finality, solver economics |
| `polymarket` | `/root/nemotron_training/models/nemotron-15m-nvidia-real/final` | 15-min crypto predictions       |
| `algotrada`  | `/root/algotrada-training/adapters/final` (TBD)          | cross-DEX arbitrage / execution intent   |

## API
- `GET  /health` — overall status, GPU usage, per-adapter load state
- `GET  /api/intel/<name>/health` — one model
- `POST /api/intel/<name>/generate` — body `{prompt, max_tokens?, temperature?, system?}`

Public routes (via nginx on scanner.taifoon.dev):
```
GET  https://scanner.taifoon.dev/api/intel/taifoon/health
POST https://scanner.taifoon.dev/api/intel/taifoon/generate
```

## Run
```
sudo cp systemd/taifoon-intel-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now taifoon-intel-api
journalctl -u taifoon-intel-api -f
```

Stop the legacy single-adapter API on :11437 first to free GPU memory:
```
sudo systemctl stop nemotron-api      # if managed by systemd, or
sudo pkill -f /opt/nemotron-api/nemotron_api.py
```

## Continuous training

Operational intel is appended to `data-lake/taifoon_intel_v3_ops_*.jsonl` by autonomous agents during the week. Run `./train-from-ops-lake.sh` (idempotent — skips if dataset hash unchanged) to combine v2 + v3 shards and re-run `sft_nemotron.py`. Schedule via cron (recommended weekly):

```
0 3 * * 0 cd /root/taifoon-nemotron && ./train-from-ops-lake.sh >> /var/log/taifoon-intel-train.log 2>&1
```

After a successful train, the systemd unit must be restarted to pick up the new adapter:
```
sudo systemctl restart taifoon-intel-api
```

## Adding a new adapter (e.g. algotrada-trader)

1. Train the adapter using the same `sft_nemotron.py` flow against the algotrada training corpus
2. Drop the resulting `adapter_config.json` + `adapter_model.safetensors` at the path declared in `systemd/taifoon-intel-api.service` (`ALGOTRADA_ADAPTER`)
3. `systemctl restart taifoon-intel-api` — the slot becomes live; no code change required.

## Architecture rationale

PEFT supports multi-adapter loading (`model.load_adapter(name, path)` then `model.set_adapter(name)`). Memory cost: 1× base (~5GB in 8-bit) + N × LoRA (~80MB each). On RTX 4000 Ada (20GB VRAM) we have headroom for >10 adapters total. Switching adapters is sub-millisecond.
