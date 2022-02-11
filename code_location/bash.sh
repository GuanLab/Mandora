#!/bin/bash

set -e

#the_seed=$1
the_seed=0

sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g; s/the_lr=1e-3/the_lr=1e-4/g; s/model.summary()/#model.summary()/g; s/nb_epoch=50/nb_epoch=10/g' train.py > continue_train.py

## 1.narrowing hand
type_target='nh'
for i in {0..14}
do
    target=${type_target}${i}
    echo $target

    num=50
    dir=epoch${num}
    echo $dir
    mkdir -p $dir
    python train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
    cp weights_${target}_seed${the_seed}.h5 $dir

    for num in {60..100..10}
    do
        dir=epoch${num}
        echo $dir
        mkdir -p $dir
        python continue_train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
        cp weights_${target}_seed${the_seed}.h5 $dir
    done
    python pred.py -t $target -s $the_seed -e $num | tee $dir/result_${target}_$the_seed.txt
done

## 2.erosion hand
type_target='eh'
for i in {0..5}
do
    target=${type_target}${i}
    echo $target

    num=50
    dir=epoch${num}
    echo $dir
    mkdir -p $dir
    python train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
    cp weights_${target}_seed${the_seed}.h5 $dir

    for num in {60..100..10}
    do
        dir=epoch${num}
        echo $dir
        mkdir -p $dir
        python continue_train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
        cp weights_${target}_seed${the_seed}.h5 $dir
    done
    python pred.py -t $target -s $the_seed -e $num | tee $dir/result_${target}_$the_seed.txt
done

## 3.narrowing foot
type_target='nf'
for i in {0..5}
do
    target=${type_target}${i}
    echo $target

    num=50
    dir=epoch${num}
    echo $dir
    mkdir -p $dir
    python train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
    cp weights_${target}_seed${the_seed}.h5 $dir

    for num in {60..100..10}
    do
        dir=epoch${num}
        echo $dir
        mkdir -p $dir
        python continue_train.py -t $target -s $the_seed | tee -a log_${target}_${the_seed}.txt
        cp weights_${target}_seed${the_seed}.h5 $dir
    done
    python pred.py -t $target -s $the_seed -e $num | tee $dir/result_${target}_$the_seed.txt
done

## 4.consensus prediction
num_seed=1

# nh
for k in {0..14}
do
    target=nh$k
    echo $target
    python pred_consensus.py -t ${target} -ns ${num_seed} -e 100 | tee log_${target}.txt &
done
wait

# nf
for k in {0..5}
do
    target=nf$k
    echo $target
    python pred_consensus.py -t ${target} -ns ${num_seed} -e 100 | tee log_${target}.txt &
done
wait

# eh
for k in {0..5}
do
    target=eh$k
    echo $target
    python pred_consensus.py -t ${target} -ns ${num_seed} -e 100 | tee log_${target}.txt &
done
wait





