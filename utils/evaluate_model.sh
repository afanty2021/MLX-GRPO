#!/bin/bash
# Evaluate model on test questions

MODEL_PATH="${1:-outputs/ultra-light/ultra-light-test/checkpoint-50}"

echo "============================================================"
echo "Evaluating Model: $MODEL_PATH"
echo "============================================================"
echo ""

# Test questions (easy to hard)
questions=(
    "Sarah has 12 apples and buys 8 more. How many apples does she have in total?"
    "A store has 48 shirts. They sell half of them on Monday and 12 more on Tuesday. How many shirts are left?"
    "Emily earns \$15 per hour and works 6 hours a day. If she works 5 days, how much does she earn in total?"
    "A farmer has 30 chickens. Each chicken lays 2 eggs per day. If the farmer sells eggs for \$3 per dozen, how much money does he make in one day?"
    "Tom has \$500. He spends 40% on rent, \$80 on food, and saves the rest. How much does he save?"
)

answers=(20 12 450 15 220)

for i in "${!questions[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Question $((i+1)): ${questions[$i]}"
    echo "Expected Answer: ${answers[$i]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    uv run python utils/inference.py \
        --model "$MODEL_PATH" \
        --prompt "${questions[$i]}" \
        --max-tokens 150 \
        --temperature 0.0
    
    echo ""
done

echo "============================================================"
echo "Evaluation Complete!"
echo "============================================================"

