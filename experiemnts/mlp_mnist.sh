#!/bin/bash

seed=1

for i in 0 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.7
do
    python eval/mlp/train.py --config eval/mlp/config.py \
        --config.model.dropout.p=$i \
        --config.model.dropout.variant="vanilla" \
        --config.seed=$seed

    python eval/mlp/train.py --config eval/mlp/config.py \
        --config.model.dropout.p=$i \
        --config.model.dropout.variant="blockwise[cuda]" \
        --config.seed=$seed
done
