#!/usr/bin/env bash
#2, 3
for k in `echo 10 50`
do
    python train_pt.py --nmb_cluster ${k}
    python train_ft.py --nmb_cluster ${k}
done