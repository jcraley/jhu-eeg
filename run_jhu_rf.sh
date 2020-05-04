#!/bin/bash 

for ((ii=1; ii<35; ii++))
do
    features='["bandpass"]'
    python run_experiment.py \
        --experiment_name b_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["sampen","power","linelength","lle"]'
    python run_experiment.py \
        --experiment_name spll_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["bandpass","sampen","power","linelength","lle"]'
    python run_experiment.py \
        --experiment_name bspll_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

done
