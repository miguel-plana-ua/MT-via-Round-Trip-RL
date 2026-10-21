#!/bin/bash

# Plain Backtranslation Baseline Training Script
# This script trains the backtranslation baseline on multiple language pairs.

# Language pairs to train (source-target)
LANG_PAIRS=(
    "eng_Latn-bho_Deva"
    "eng_Latn-dyu_Latn"
)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output log file
LOG_FILE="${SCRIPT_DIR}/backtranslation_output_$(date +%Y%m%d_%H%M%S).log"

echo "Plain Backtranslation Baseline Training"
echo "Log file: ${LOG_FILE}"
echo ""

# Loop through each language pair
for LANG_PAIR in "${LANG_PAIRS[@]}"; do
    {
        echo "=========================================="
        echo "Starting backtranslation training: ${LANG_PAIR}"
        echo "Time: $(date)"
        echo "=========================================="

        # Extract source and target languages
        SRC_LANG="${LANG_PAIR%%-*}"
        TGT_LANG="${LANG_PAIR##*-}"

        python "${SCRIPT_DIR}/backtranslation.py" \
            --dataset_config "${LANG_PAIR}" \
            --src_lang_nllb "${SRC_LANG}" \
            --tgt_lang_nllb "${TGT_LANG}" \
            --num_epochs 2 \
            --batch_size 2 \
            --bt_batch_size 2 \
            --learning_rate 3e-5 \
            --regenerate_bt_every 1

        # Check if training succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for ${LANG_PAIR}"
            exit 1
        fi

        echo "Completed: ${LANG_PAIR} at $(date)"
        echo ""
    } >> "$LOG_FILE" 2>&1
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "All backtranslation training runs completed!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

