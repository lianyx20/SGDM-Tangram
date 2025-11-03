# SGDM-Tangram
Official implementation of **SGDM-Tangram** by *Yongxiang Lian*, *Yueyang Cang*, *Pingge Hu*, *Yuchen He*.

---

## Overview
**SGDM-Tangram** is a cognitive image reconstruction research project.
This repository provides a minimal pipeline and core implementation for review purposes.
Complete implementation details (params, configs, preprocessing, visualization) will be available upon the official publication of the paper.

## Requirements
Run setup.sh to quickly create a conda environment that contains the packages necessary to run our scripts; activate the environment with conda activate tangram.

## Datasets
Annotations, stimulus, EEG(raw and preprocessed) of EEG-Kilogram: https://osf.io/7qm35/

## Pipeline
1. Finetune: CLIP_finetune_tangram.py/CLIP_finetune_things.py
   
2. Train SGDM(stage 1): Train_vae_cogcode.py
   
3. Train SGDM stage 2: Available upon the official publication

4. Generation: CognitiveGen.ipynb

4. Eval: Eval_Metrics.ipynb

## Acknowledge
1.THING-EEG dataset cited in the paper:
"A large and rich EEG dataset for modeling human visual object recognition".
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy.

2.Base model from the paper:
"Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion".
Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, Haoyang Qin, Quanying Liu.
