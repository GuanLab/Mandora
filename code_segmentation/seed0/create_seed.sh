#!/bin/bash

set -e


python partition_data.py 0

for i in {1..9}
do
    mkdir -p ../seed${i} 
    cp individual_all.txt ../seed${i}
    cp partition_data.py ../seed${i}
    cd ../seed${i}
    python partition_data.py ${i}
    cd ../seed0
done




