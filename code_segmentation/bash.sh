#!/bin/bash

set -e

#the_seed=$1
the_seed=0
echo ${the_seed}

target_all=('eh' 'ef' 'nh' 'nf')

for target in ${target_all[@]}
do
    echo $target
    num=10
    dir=epoch${num}
    echo $dir
    mkdir -p $dir
    python train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
    cp weights_${target}_seed${the_seed}.h5 $dir

    for num in {20..50..10}
    do
        dir=epoch${num}
        echo $dir
        mkdir -p $dir
        sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g; s/the_lr=1e-3/the_lr=1e-4/g; s/model.summary()/#model.summary()/g' train.py > continue_train.py
        python continue_train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
        cp weights_${target}_seed${the_seed}.h5 $dir
    done
    python pred.py -t $target -s $the_seed -e $num | tee $dir/result_${target}_$the_seed.txt
done






