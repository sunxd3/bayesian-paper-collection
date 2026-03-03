threads_per_chain=15
model="basic_fix_cdf1_cdf0_log_sum_exp"
#model="basic_cens"
for sub_lower in {1..2000..40}
#for sub_lower in {1..20..5}
#for sub_lower in {1961..2000..1}
do
	sub_upper=`expr $sub_lower + 39`
	#sub_upper=`expr $sub_lower + 4`
	#sub_upper=${sub_lower}
	sbatch -p single -t 24:00:00 --ntasks=4 --ntasks-per-node=4 \
		--cpus-per-task=${threads_per_chain} --mem=500mb --job-name=trc_${sub_lower} \
		--output=logs/trunc_${sub_lower}_${model}.out \
		--error=logs/trunc_${sub_lower}_${model}.err \
	       	--export=sub_lower=${sub_lower},sub_upper=${sub_upper},threads_per_chain=${threads_per_chain},model=${model} fit_groups.sh

done

