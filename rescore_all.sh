#!/bin/bash

experiments=( jhu_k_rf jhu_k_svm jhu_rf_b jhu_rf_bspll jhu_rf_spll jhu_svm_b jhu_svm_bspll jhu_svm_spll )

for exp in ${experiments[@]}
do
    for ((ii=1; ii<2; ii++))
    do
        echo $exp $ii
        echo python rescore_experiment.py \
            --config_fn Experiments/"$exp"/pt"$ii"/config.ini \
            --experiment_name "$exp"_samples20 \
            --load_model_fn Experiments/"$exp"/pt"$ii"/models/model.pt \
            --fps_per_hour 5.0  --max_samples_before_sz 20 \
            --visualize_train 1
    done
done