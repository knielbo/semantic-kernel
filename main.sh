#!/usr/bin/env bash

# input vars
VENVNAME=sickern
TRAIN=false
SEEDS=res/*

# test graphvis installation
if ! command -v dot -V &> /dev/null
then
    echo "[INFO] installing graphviz and graphviz-dev..."
    sudo apt-get -y install graphviz graphviz-dev
else
    echo "[INFO] graphviz and graphviz-dev identified..."
fi

# test environment
if [ -d $VENVNAME ] 
then
    echo "[INFO] activate: $VENVNAME"
    source sickern/bin/activate
else
    echo "[INFO] build: $VENVNAME"
    bash create_venv.sh
    source sickern/bin/activate
    python -m spacy download da_core_news_sm
    python -m nltk.downloader punkt
    echo "[INFO] activate: $VENVNAME"
fi

# train vector representations
if [ $TRAIN = true ]
then
    echo "[INFO] training vectors..."
    python train_vectors.py
else
    echo "[INFO] using existing vectors"
fi

# build kernel and graph
for seedls in $SEEDS
do
    echo "[INFO] Processing $seedls file..."
    python src/build_kernel.py --model mdl/vectors_expr3.pcl --seed $seedls --norm True
    python src/build_graph.py --seed $seedls
    echo "-----"
done