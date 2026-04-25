#!/usr/bin/env bash
# train-from-ops-lake.sh
# ──────────────────────────────────────────────────────────────────
# Continuously refresh the taifoon-intel adapter with operational
# intelligence appended by autonomous agents during the week.
#
# Combines:
#   data-lake/taifoon_intel_v2.jsonl              (1149 base records)
#   data-lake/taifoon_intel_v3_ops_*.jsonl        (operational deltas)
# Writes:
#   data-lake/taifoon_intel_combined.jsonl        (training input)
# Then invokes:
#   sft_nemotron.py                               (QLoRA r=32, α=64)
#
# Idempotent: skips training if combined dataset hash hasn't changed
# vs the last successful run (.last_train_hash).
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
LAKE_DIR="${LAKE_DIR:-./data-lake}"
HASH_FILE=".last_train_hash"

mkdir -p "$LAKE_DIR"

# 1. Combine all jsonl shards into a single training input
COMBINED="$LAKE_DIR/taifoon_intel_combined.jsonl"
cat "$LAKE_DIR"/taifoon_intel_v2.jsonl 2>/dev/null > "$COMBINED" || true
for shard in "$LAKE_DIR"/taifoon_intel_v3_ops_*.jsonl; do
    [ -e "$shard" ] && cat "$shard" >> "$COMBINED"
done
RECORDS=$(wc -l < "$COMBINED")
HASH=$(sha256sum "$COMBINED" | awk '{print $1}')
LAST=$([ -f "$HASH_FILE" ] && cat "$HASH_FILE" || echo "")

echo "[$(date -u +%FT%TZ)] combined_records=$RECORDS hash=$HASH"
if [ "$HASH" = "$LAST" ]; then
    echo "  no change since last successful train — skipping"
    exit 0
fi

echo "  invoking sft_nemotron.py with combined dataset..."
DATASET="$COMBINED" python3 sft_nemotron.py \
    || { echo "training failed; not updating hash"; exit 1; }

echo "$HASH" > "$HASH_FILE"
echo "[$(date -u +%FT%TZ)] training cycle complete; adapter at adapters/final/"
