#!/bin/bash

set -e

mkdir -p log

target_all=('nh' 'nf' 'eh' 'ef')

for the_target in ${target_all[@]}
do
    for i in {0..9}
    do
        python etr5.py -t ${the_target} -su ${i} | tee log/log_etr5_${the_target}_su${i}.txt &
    done
    wait
done

python pred_etr5_consensus.py

