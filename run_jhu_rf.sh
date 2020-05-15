#!/bin/bash 

for ((ii=1; ii<35; ii++))
do
    features='["bandpass_normalized"]'
    python run_experiment.py \
        --experiment_name b_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal --features $features \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["sampen_normalized","power_normalized","linelength_normalized","lle_normalized"]'
    python run_experiment.py \
        --experiment_name spll_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal --features $features \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["bandpass_normalized","sampen_normalized","power_normalized","linelength_normalized","lle_normalized"]'
    python run_experiment.py \
        --experiment_name bspll_rf --trial_name pt"$ii" \
        --dataset jhu_clipped_longitudinal --features $features \
        --train_manifest Manifests/jhu/pt"$ii"-train.csv \
        --val_manifest Manifests/jhu/pt"$ii"-test.csv \
        --model_type RandomForest

done
