#!/bin/bash

for model_penalty in $(seq 0.4 0.1 0.6)
do
    python3 speed_up_test.py --model_penalty $model_penalty
done