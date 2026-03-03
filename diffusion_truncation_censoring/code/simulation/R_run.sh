#!/bin/bash
#SBATCH --job-name=run_R_1
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --time=02:00:00
#SBATCH --mem=1gb
#SBATCH --export=NONE
#SBATCH -o run_R_1.out
#SBATCH -e run_R_1.err

#To compile the model directly with cmdstan
#cd ~/Study_3-CDF/cmdstan
#make ~/Study_3-CDF/S3_basic_fix_cdf1_cdf0_log_sum_exp
#cd ~/Study_3-CDF

module load math/R/4.1.2

#To simulate data
#Rscript S3_simulate.R --model 'basic_fix_cdf1_cdf0_log_sum_exp' # 'basic_cens' #'basic_fix_cdf1_cdf0'

#To sample truncated model
#Rscript S3_sample.R --warmup 100 --sampling 300 --threads_per_chain 10 \
#       	--subject_range_lower 6 --subject_range_upper 6 --model 'basic_cens'

#To extract results
#Rscript S3_extract_results.R --folder '_basic_fix_cdf1_cdf0_log_sum_exp'
#Rscript S3_SBC_lp.R --folder '_basic_fix_cdf1_cdf0' --cores 60 #lieber die start_groups.sh hierfür nutzen, dauert sonst ewig

#To combine results
Rscript S3_combine_results.R --folder '_basic_fix_cdf1_cdf0_log_sum_exp'

#To sample for increasing sampling iterations
#for iterator in {1..1} 
#do
#	Rscript analysis_johnson/S3_sample_johnson.R --model 'full' --warmup 500 --sampling 250 --threads 60
#done


