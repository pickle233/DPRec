# A Dual-Path Preference Evolution Framework with Semantic Alignment for Knowledge Graph Recommendation


## Introduction
This repository provides the official implementation of DPRec, a knowledge graph-based recommendation model.

## Requirement
The code has been tested running under Python 3.9.21. The required packages are as follows:
- pytorch == 2.1.1
- numpy == 1.26.4
- sklearn == 1.6.1

## Usage

Run main.py to train the model.


## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-20M, and Last.FM.

We follow the papers 
"[RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://github.com/hwwang55/RippleNet)" 
and "[User Preference Interaction Fusion and Swap Attention Graph Neural Network for Recommender System](https://github.com/mwmytu/PIFSA-GNN)" 
for data processing.


|                       |               | Book-Crossing | MovieLens-20M | Last.FM |
| :-------------------: | :------------ |--------------:|--------------:| ---------: |
| User-Item Interaction | #Users        |        17,860 |        98,741 |      1,872 |
|                       | #Items        |        14,967 |        16,954 |      3,846 |
|                       | #Interactions |       139,746 |       674,424 |      42,346|
|    Knowledge Graph    | #Entities     |        77,903 |       102,569 |      9,366 |
|                       | #Relations    |            25 |            32 |         60 |
|                       | #Triplets     |       151,500 |       499,474 |     15,518 |


