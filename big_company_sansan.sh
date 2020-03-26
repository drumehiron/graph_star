#!/bin/sh

python big_company_sansan_lp.py --dataset=big_company_sansan --dropout=0 --hidden=512 --l2=5e-4 --num_layers=3 --cross_layer=False --patience=500 --residual=True --residual_star=True --modelname=big_company_sansan