#!/usr/bin/env python3
"""
serve_intel.py
─────────────────────────────────────────────────────────────────
Unified Taifoon Intel inference server.

Serves three QLoRA adapters from ONE Nemotron-3-Nano-4B base model
on a single CUDA device. Adapters are pre-loaded at startup; per-request
routing uses peft `set_adapter()` for sub-millisecond switching.

Routes:
  GET  /health                           overall liveness + per-model status
  GET  /api/intel/<name>/health          one model's status (loaded? params?)
  POST /api/intel/<name>/generate        {prompt, max_tokens?, temperature?, system?}

Where <name> ∈ {taifoon, polymarket, algotrada}. Algotrada slot returns
503 until its adapter directory is populated.

PERFORMANCE NOTES:
- BF16 (not 8-bit) for faster matmul on RTX 4000 Ada (~5GB VRAM, fits comfortably)
- Startup pre-warm with dummy generate eliminates 60s cold-CUDA on first user request
- use_cache=True explicitly for KV cache reuse across tokens

Owner: yawningmonsoon · Updated: 2026-04-25
"""
import os, json, logging, threading, time
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [INTEL-API] %(message)s'
)
log = logging.getLogger(__name__)

# ── adapter registry ───────────────────────────────────────────
ADAPTERS = {
    'taifoon':   {
        'path':    os.environ.get('TAIFOON_ADAPTER',   '/root/taifoon-nemotron/adapters/final'),
        'persona': 'You are Taifoon Intel — protocol routing, V5 finality, solver economics.',
    },
    'polymarket': {
        'path':    os.environ.get('POLYMARKET_ADAPTER', '/root/nemotron_training/models/nemotron-15m-nvidia-real/final'),
        'persona': 'You are Nemotron-Enhanced — 15-minute crypto price predictions on Polymarket.',
    },
    'algotrada': {
        'path':    os.environ.get('ALGOTRADA_ADAPTER',  '/root/algotrada-training/adapters/final'),
        'persona': 'You are Algotrada Trader — cross-DEX arbitrage and execution intent.',
    },
}
BASE_MODEL = os.environ.get('BASE_MODEL', 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16')
PORT       = int(os.environ.get('PORT', '11500'))
USE_8BIT   = os.environ.get('USE_8BIT', '0') == '1'  # default OFF — BF16 is faster on Ada
PREWARM    = os.environ.get('PREWARM', '1') == '1'

app = Flask(__name__)
LOCK = threading.Lock()

_state = {
    'base_loaded':  False,
    'adapters':     {},
    'tokenizer':    None,
    'model':        None,
    'started_at':   None,
    'base_model':   BASE_MODEL,
    'precision':    '8-bit' if USE_8BIT else 'bf16',
}


def _load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    log.info('=' * 70)
    log.info(f'Loading base {BASE_MODEL}  precision={_state["precision"]}')
    log.info('=' * 70)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if USE_8BIT:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map='auto',
            torch_dtype=torch.float16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )

    model = None
    for slug, cfg in ADAPTERS.items():
        path = cfg['path']
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'adapter_config.json')):
            log.warning(f'  [{slug}] NOT FOUND at {path} — slot reserved')
            _state['adapters'][slug] = {'loaded': False, 'path': path, 'error': 'adapter directory missing'}
            continue
        try:
            if model is None:
                log.info(f'  [{slug}] PeftModel.from_pretrained ← {path}')
                model = PeftModel.from_pretrained(base, path, adapter_name=slug)
            else:
                log.info(f'  [{slug}] model.load_adapter ← {path}')
                model.load_adapter(path, adapter_name=slug)
            _state['adapters'][slug] = {'loaded': True, 'path': path}
        except Exception as e:
            log.error(f'  [{slug}] LOAD FAILED: {e}')
            _state['adapters'][slug] = {'loaded': False, 'path': path, 'error': str(e)}

    if model is None:
        log.warning('No adapters loaded; serving base model only')
        model = base

    model.eval()
    _state['base_loaded'] = True
    _state['tokenizer']   = tok
    _state['model']       = model
    _state['started_at']  = int(time.time())

    if PREWARM:
        loaded = [n for n, info in _state['adapters'].items() if info.get('loaded')]
        if loaded:
            log.info(f'Pre-warming CUDA kernels with adapter={loaded[0]} (1 token)...')
            t0 = time.time()
            try:
                with torch.inference_mode():
                    if hasattr(model, 'set_adapter'):
                        model.set_adapter(loaded[0])
                    inputs = tok('Hi', return_tensors='pt').to(model.device)
                    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                       pad_token_id=tok.eos_token_id, use_cache=True)
                log.info(f'  pre-warm complete in {time.time()-t0:.1f}s')
            except Exception as e:
                log.warning(f'  pre-warm failed (non-fatal): {e}')

    log.info(f'Startup complete — listening on 0.0.0.0:{PORT}')


@app.route('/health')
def health():
    return jsonify({
        'status':       'ok' if _state['base_loaded'] else 'starting',
        'base_model':   BASE_MODEL,
        'precision':    _state['precision'],
        'started_at':   _state['started_at'],
        'adapters':     _state['adapters'],
        'gpu':          _gpu_info(),
    })


@app.route('/api/intel/<name>/health')
def adapter_health(name):
    if name not in ADAPTERS:
        return jsonify({'status': 'unknown_model', 'available': list(ADAPTERS.keys())}), 404
    info = _state['adapters'].get(name, {'loaded': False, 'error': 'not initialized'})
    code = 200 if info.get('loaded') else 503
    return jsonify({'name': name, **info}), code


@app.route('/api/intel/<name>/generate', methods=['POST'])
def adapter_generate(name):
    if name not in ADAPTERS:
        return jsonify({'error': 'unknown_model', 'available': list(ADAPTERS.keys())}), 404
    info = _state['adapters'].get(name, {})
    if not info.get('loaded'):
        return jsonify({'error': 'adapter_not_loaded', 'reason': info.get('error', 'unknown')}), 503

    body = request.get_json(force=True, silent=True) or {}
    prompt   = body.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400
    max_new  = int(body.get('max_tokens', 128))
    temp     = float(body.get('temperature', 0.6))
    system   = body.get('system', ADAPTERS[name]['persona'])

    text = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

    import torch
    with LOCK:
        try:
            if hasattr(_state['model'], 'set_adapter'):
                _state['model'].set_adapter(name)
        except Exception:
            pass
        tok = _state['tokenizer']
        inputs = tok(text, return_tensors='pt').to(_state['model'].device)
        t0 = time.time()
        with torch.inference_mode():
            out = _state['model'].generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temp,
                do_sample=temp > 0.0,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        dur = (time.time() - t0)
        gen = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    n_new = int(out.shape[1] - inputs['input_ids'].shape[1])
    return jsonify({
        'model':      name,
        'response':   gen.strip(),
        'duration':   round(dur, 3),
        'tokens':     n_new,
        'tokens_per_second': round(n_new / max(dur, 1e-3), 2),
    })


def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'name':      torch.cuda.get_device_name(0),
                'allocated': round(torch.cuda.memory_allocated() / 1e9, 2),
                'reserved':  round(torch.cuda.memory_reserved() / 1e9, 2),
            }
    except Exception:
        pass
    return None


if __name__ == '__main__':
    _load()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
