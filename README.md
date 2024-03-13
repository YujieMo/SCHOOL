# Revisiting Self-Supervised Heterogeneous Graph Learning from Spectral Clustering Perspective 

This repository contains the reference code for the manuscript ``Revisiting Self-Supervised Heterogeneous Graph Learning from Spectral Clustering Perspective" 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)


## Installation
* pip install -r requirements.txt 
* Unzip the datasets (datasets can be found in ./dataset/)

## Preparation
Important args:
* `--use_pretrain` Test checkpoints to reproduce the results 
* `--dataset` Heterogeneous graph: ACM, Yelp, DBLP, Aminer
* `--custom_key` Node: node classification

## Training
python main.py


