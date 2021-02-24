#!/usr/bin/env bash
#0, 1
for k in `echo 100 200`
do
    python train_pt2.py --nmb_cluster ${k}
    python train_ft2.py --nmb_cluster ${k}
done