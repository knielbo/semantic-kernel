#!/usr/bin/env bash
# test graphvis installation

# test environment or make
source sickern/bin/activate


#SEED=res/seedlist.txt
# python src/train_vectors.py
#python src/build_kernel.py --model mdl/vectors_expr3.pcl --seed $SEED --norm True
#python src/build_graph.py --seed $SEED



SEEDS=res/*
for seedls in $SEEDS
do
    echo "[fINFO] Processing $seedls file..."
    python src/build_kernel.py --model mdl/vectors_expr3.pcl --seed $seedls --norm True
    python src/build_graph.py --seed $seedls
    echo "-----"
done