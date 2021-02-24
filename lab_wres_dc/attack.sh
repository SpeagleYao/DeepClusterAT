#!/usr/bin/env bash

for model in `ls ../cp_dc_adv`;
do
    echo "model:" ${model} >> dc_res18_adv.txt 
    python main_verify.py --model-checkpoint ../cp_dc_adv/${model} >> dc_res18_adv.txt 
done 