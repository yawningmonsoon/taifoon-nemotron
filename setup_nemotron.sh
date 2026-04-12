#!/usr/bin/env bash
# taifoon-nemotron — repeatable installation on RTX 4000 Ada (20GB)
# Installs PyTorch 2.6+cu124, mamba-ssm (pre-built + ABI shim), and training deps
set -euo pipefail
LOG=/root/taifoon-training/setup.log
mkdir -p /root/taifoon-training/wheels
echo "[$(date)] taifoon-nemotron setup" | tee $LOG

# 1. PyTorch 2.6.0+cu124 (required for mamba-ssm 2.3.1 pre-built wheels)
echo "[1/5] Installing PyTorch 2.6+cu124..." | tee -a $LOG
pip install torch==2.6.0+cu124 torchvision torchaudio ninja \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --break-system-packages -q 2>&1 | tee -a $LOG

# 2. Training stack
echo "[2/5] Installing training stack..." | tee -a $LOG
pip install transformers==4.57.6 peft==0.17.1 bitsandbytes==0.49.2 \
    trl==0.17.1 datasets accelerate packaging \
    --break-system-packages -q 2>&1 | tee -a $LOG

# 3. mamba-ssm pre-built wheel (cxx11abiTRUE variant for torch 2.6)
echo "[3/5] Installing mamba-ssm pre-built wheels..." | tee -a $LOG
cd /root/taifoon-training/wheels
CAUSAL_WHL="causal_conv1d-1.6.1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
MAMBA_WHL="mamba_ssm-2.3.1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
[ -f "$CAUSAL_WHL" ] || curl -L -o "$CAUSAL_WHL" \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/${CAUSAL_WHL}"
[ -f "$MAMBA_WHL" ] || curl -L -o "$MAMBA_WHL" \
    "https://github.com/state-spaces/mamba/releases/download/v2.3.1/${MAMBA_WHL}"
pip install "$CAUSAL_WHL" "$MAMBA_WHL" --break-system-packages --no-deps --force-reinstall 2>&1 | tee -a $LOG
cd -

# 4. ABI shim — bridges torchCheckFail symbol mismatch (torch 2.6 old-ABI vs wheel new-ABI)
echo "[4/5] Building ABI compatibility shim..." | tee -a $LOG
cat > /root/taifoon-training/torch_shim.cpp << 'CPPEOF'
#include <string>
#include <cstdint>
namespace c10 { namespace detail {
    [[noreturn]] void torchCheckFail(const char* func, const char* file, uint32_t line, const std::string& msg);
    [[noreturn]] void torchCheckFail(
        const char* func, const char* file, uint32_t line,
        const std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>& msg)
    {
        std::string s(msg.data(), msg.size());
        torchCheckFail(func, file, line, s);
    }
}}
CPPEOF
g++ -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
    /root/taifoon-training/torch_shim.cpp \
    -o /root/taifoon-training/libtorch_shim.so \
    -Wl,--allow-shlib-undefined 2>&1 | tee -a $LOG

# 5. Patch mamba_ssm __init__.py to make selective_scan import soft (only triton ops needed)
echo "[5/5] Patching mamba_ssm for soft import..." | tee -a $LOG
INIT=$(python3 -c "import mamba_ssm; import os; print(os.path.join(os.path.dirname(mamba_ssm.__file__), '__init__.py'))")
cat > "$INIT" << 'PYEOF'
__version__ = "2.3.1"
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:
    selective_scan_fn = None; mamba_inner_fn = None
    Mamba = None; Mamba2 = None; MambaLMHeadModel = None
PYEOF

# Verify
LD_PRELOAD=/root/taifoon-training/libtorch_shim.so python3 -c "
import mamba_ssm
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
print('mamba-ssm OK:', mamba_ssm.__version__)
" && echo "[OK] mamba-ssm verified" | tee -a $LOG

echo "[$(date)] Setup complete." | tee -a $LOG
echo "Run training with:"
echo "  LD_PRELOAD=/root/taifoon-training/libtorch_shim.so \\"
echo "  nohup python3 /root/taifoon-nemotron/sft_nemotron.py > /root/taifoon-training/train_nemotron.log 2>&1 &"
