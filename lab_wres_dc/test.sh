#!/usr/bin/env bash
#2, 3
for model in `ls ../cp_dc_ft`;
do
    echo "model:" ${model} >> dc_res18_ft_search.txt 
    python main_verify.py --model-checkpoint ../cp_dc_ft/${model} >> dc_res18_ft_search.txt 
done 