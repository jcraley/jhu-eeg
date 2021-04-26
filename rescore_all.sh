#!/bin/bash

# UW Experiments

for exp in uw_b_rf_sz_only uw_spec_rf_sz_only chb_to_uw_b_rf_sz_only chb_to_uw_spec_rf_sz_only;
do
    for pt in A B C D E F G H J K L M
    do
	sbatch gpu_header.sh python rescore_from_preds.py \
	    --config_fn Experiments/"$exp"/pt"$pt"/config.ini \
	    --max_samples_before_sz 20
    done
done

for exp in chb_b_rf_all chb_spec_rf_all
do
    for (( pp=1; pp<25; pp++ ))
    do
	pt=chb"$pp"
	if [ "$pp" -lt 10 ]
	then

	    pt=chb0"$pp"
	fi

	sbatch gpu_header.sh python rescore_from_preds.py \
	    --config_fn Experiments/"$exp"/"$pt"/config.ini \
	    --max_samples_before_sz 20
    done
done
