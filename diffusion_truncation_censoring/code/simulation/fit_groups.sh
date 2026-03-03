#!/bin/bash

module load math/R/4.1.2

#To sample for increasing sampling iterations
#for samp in {250..1550..100}
#do
#	Rscript S3_sample.R --warmup 500 --sampling $samp --threads_per_chain $threads_per_chain \
#		--subject_range_lower $sub_lower --subject_range_upper $sub_upper --model $model
#done

#To plot traceplots
#Rscript S2_print_traceplots.R --inputdir 'failed_fits/HDDM/Experiment_1/group_01/subject_012/trials_020_230210-154737/'

#To extract results
#Rscript S3_extract_results.R --folder '_'$model
Rscript S3_SBC_lp.R --folder '_'$model --cores $threads_per_chain \
	--subject_range_lower $sub_lower --subject_range_upper $sub_upper

