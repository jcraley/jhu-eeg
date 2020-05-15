#!/bin/bash 

for ((ii=1; ii<25; ii++))
do
    features='["bandpass_normalized"]'
    python run_experiment.py \
        --experiment_name chb_b_rf --trial_name pt"$ii" \
        --dataset chb --features $features \
        --train_manifest Manifests/chb/pt"$ii"-train.csv \
        --val_manifest Manifests/chb/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["sampen_normalized","power_normalized","linelength_normalized","lle_normalized"]'
    python run_experiment.py \
        --experiment_name chb_spll_rf --trial_name pt"$ii" \
        --dataset chb --features $features \
        --train_manifest Manifests/chb/pt"$ii"-train.csv \
        --val_manifest Manifests/chb/pt"$ii"-test.csv \
        --model_type RandomForest

    features='["bandpass_normalized","sampen_normalized","power_normalized","linelength_normalized","lle_normalized"]'
    python run_experiment.py \
        --experiment_name chb_bspll_rf --trial_name pt"$ii" \
        --dataset chb --features $features \
        --train_manifest Manifests/chb/pt"$ii"-train.csv \
        --val_manifest Manifests/chb/pt"$ii"-test.csv \
        --model_type RandomForest

done
