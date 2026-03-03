# Study 3 - CDF
# This study will be a recovery study the analysis of truncated data with the diffusion model. 
# full diffusion model, non-hierarchical

library(WienR)
library(truncnorm)
library(optparse) # for input arguments
library(tidyverse)

argument_list = list (
  make_option(c('-m', '--model'), type='character', default='basic_fix_cdf1_cdf0_log_sum_exp',
              help='name of model: basic_fix, basic_cens, basic_fix_new, basic_fix_cdf1_cdf0', metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

## Set configuration and choose which setup to simulate with input arguments
output_dir = paste0("S3_truncated_", args$model)

NTrials = c(100, 500) 
NSubj = 2000

## Prepare the data
simulate_exp = function(k=2, N=100, a=1, w=0.5, v=c(+1,-1), t0=0.3, sw=0, sv=0, st0=0, upper_bound=50, method='ars') {
  data = data.frame();
  for (c in 1:k) {
    N.c  = ifelse(is.na(N[c]),  N[1],  N[c])
    a.c  = ifelse(is.na(a[c]),  a[1],  a[c])
    w.c = ifelse(is.na(w[c]), w[1], w[c])
    v.c  = ifelse(is.na(v[c]),  v[1],  v[c])
    t0.c = ifelse(is.na(t0[c]), t0[1], t0[c])
    
    sample = WienR::sampWiener(N.c, a.c, v.c, w.c, t0.c, sv, sw, st0, bound=upper_bound, method=method)  # from t0 to t0 + st0

    data = rbind(
      data,
      data.frame(
        cnd = c,
        resp = ifelse(sample$response == "upper", 1, 0),
        rt = sample$q
      )
    )
  }
  return(data)
}
for (subj in 1:NSubj) {
  print(paste0('Simulate subject ', subj))
  set.seed(subj*123456)

  ## Draw the individual parameters
  params = list(
    a   = rtruncnorm(1, a=0.5, b=3, mean=1, sd=1),  # own choice based on Matzke & Wagenmakers (2009)
    t0  = rtruncnorm(1, a=0.2, b=1, mean=0.435, sd=0.12),  # own choice based on Matzke & Wagenmakers (2009)
    st0 = 0, #rtruncnorm(1, a=0, b=0.5, mean=0.183, sd=0.09),  # based on Matzke & Wagenmakers (2009), table 3
    sw  = 0, #rbeta(1, 1, 3),  # from HDDM, https://hddm.readthedocs.io/en/latest/methods.html
    v   = rtruncnorm(2, a=0, b=5, mean=2, sd=3) * c(1, -1),  # from HDDM
    sv  = 0 #rtruncnorm(1, a=0, b=3, mean=1, sd=3)  # own choice based on Matzke & Wagenmakers (2009)
  )
  # w and sw have to be compatible -> prior of w depends on sw, to ensure
  # w - sw / 2 > 0 and w + sw / 2 < 1
#  params$w = rtruncnorm(1, a=max(.3, params$sw / 2), b=min(.7, 1 - params$sw / 2), mean=.5, sd=0.1)  # based on Matzke & Wagenmakers (2009)
  params$w = rtruncnorm(1, a=.3, b=.7, mean=.5, sd=0.1)  # based on Matzke & Wagenmakers (2009)
  params$upper_bound = 0.91 # this is the mean of the upper_bounds of the S3_truncated_basic Analysis
  
  base_path = file.path(output_dir, formatC(subj, width = 4, flag = "0"))
  if (!dir.exists(base_path)) {
    dir.create(base_path, recursive = TRUE, showWarnings = FALSE)
  }

  for (NumTrials in 1:length(NTrials)){ 
    if (!dir.exists(file.path(base_path, NTrials[NumTrials]))) {
      dir.create(file.path(base_path, NTrials[NumTrials]), recursive = TRUE, showWarnings = FALSE)
    }
#    sim_data_tmp = simulate_exp(k = 2, N=NTrials[NumTrials]/2, t0=params$t0, st0=params$st0, a=params$a,
#                            w = params$w, sw=params$sw, v = params$v, sv=params$sv, 
#                            upper_bound=100, method='rs')
#    upper_bound = round(sort(sim_data_tmp$rt)[NTrials[NumTrials]*0.8],2)
#    params$upper_bound[NumTrials] = upper_bound

    if (grepl('fix', args$model)) {
      sim_data = simulate_exp(k = 2, N=NTrials[NumTrials]/2, t0=params$t0, st0=params$st0, a=params$a,
                              w = params$w, sw=params$sw, v = params$v, sv=params$sv, 
                              upper_bound=params$upper_bound, method='rs')
    } else { # simulate data with no bound here, censor data in S3_sample.R
      sim_data = simulate_exp(k = 2, N=NTrials[NumTrials]/2, t0=params$t0, st0=params$st0, a=params$a,
                              w = params$w, sw=params$sw, v = params$v, sv=params$sv, 
                              method='rs')
      params$censored = sum(sim_data$rt>params$upper_bound)
      params$censored_prop = sum(sim_data$rt>params$upper_bound)/length(sim_data$rt)
    }
    # save data in .dat: cnd (1,2) resp (0=lower, 1=upper) rt
    write.csv(sim_data, file.path(base_path, NTrials[NumTrials], "data.csv"))
  }#end for NumTrials
  write.csv(params, file.path(base_path, "params.csv"))
}#end for subj
#bounds
