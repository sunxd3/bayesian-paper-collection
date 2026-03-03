#sample Study1 of Pleskac
#N=56, NTrials=100, 
#within-subject manipulation: race (0=W, 1=B), object (0=NG, 1=G)
library(cmdstanr)
library(parallel) # for detectCores()
library(ff) # for moving directories
library(bayesplot)
library(optparse) # for input arguments
library(tidyverse)

## IMPORTANT: Set cmdstan location:
source("setup.local.R")

argument_list = list (
  make_option(c('-w', '--warmup'), type='integer', default=100,
              help='Number warmup iterations', metavar='integer'),
  make_option(c('-s', '--sampling'), type='integer', default=400,
              help='Number sampling iterations', metavar='integer'),
  make_option(c('-i', '--thin_to'), type='integer', default=1,
              help='thinning steps', metavar='integer'),
  make_option(c('-t', '--threads_per_chain'), type='integer', default=1,
              help='threads_per_chain', metavar='integer'),
  make_option(c('-m', '--model'), type='character', default='basicS1_cens',
              help='model: basicS1, basicS1_trunc_over_cdf0+cdf1, basicS1_cens_prob', metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

## Set configuration and choose which setup to sample with input arguments
max_treedepth = 5 # 5 or 10


#read in data and separate censored data from the rest
base_dir = 'analysis_pleskac'

data = read_csv(paste0(base_dir, '/Pleskac_Daten/Study1/Study1TrialData.csv'))

upper_bound = data$upperLim[1]

data_cens = data[is.na(data$rt),]
data_rest = data[!is.na(data$rt),]



output_dir = file.path(base_dir, 'Reanalysis_Study_1', 'fits', paste0('fit_', args$model))
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}
if ("inference.txt" %in% list.files(output_dir)) { # check if fit was already done
  next
}

#Data processing
data_for_stan = data_rest # for the not-censored models (basic, trunc)
if (grepl('cens', args$model)) {
  data_for_stan = data
  data_for_stan[is.na(data_for_stan$rt),]$rt = 42000
  data_for_stan[is.na(data_for_stan$resp0DS1S),]$resp0DS1S = 42
}

stan_data = list(
  nSub = length(unique(data_for_stan$subject)),    
  nWConTarget = length(unique(data_for_stan$conditionRaceObj)),
  nWCon = length(unique(data_for_stan$conditionRace)), # 0 White, 1 Black
  nData = length(data_for_stan$subject),
  wCon = data_for_stan$conditionRace, # 1 W, 2 B
  wConTarget = data_for_stan$conditionRaceObj, # 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
  sub = data_for_stan$subject,
  rt = data_for_stan$rt/1000, # in seconds
  resp = data_for_stan$resp0DS1S, # response coded: 0 = tool/NS, 1 = gun/S
  upper_bound = data$upperLim[1], # 0.851 s in Experiment 1
  y_bin = data$ybin #data$ybin: 0 lower boundary (NG?), 2 upper boundary (G?)
)


min_rt = min(data_for_stan$rt)/1000
# configure init values and fixed values for different models
init.stan = function(){
  L = list()
  for (i in 1:4) {
    L[[i]]  = list(
      muAlpha  = matrix(runif(stan_data$nWCon, .2, 1.2), stan_data$nWCon, 1),
      muBeta  = matrix(runif(stan_data$nWCon, .45, .55), stan_data$nWCon, 1),
      muNDT = matrix(runif(stan_data$nWConTarget, .05, .1), 
                     stan_data$nWConTarget, 1),
      muDelta = matrix(runif(stan_data$nWConTarget, -.2, .2), 
                    stan_data$nWConTarget, 1),
      
      sigmaAlpha = 0.4 + runif(1, -0.05, 0.05),
      sigmaBeta = 0.1 + runif(1, -0.05, 0.05),
      sigmaDelta = 0.4 + runif(1, -0.05, 0.05),
      sigmaNDT = 0.1 + runif(1, -0.05, 0.05),
      
      alpha  = matrix(runif(stan_data$nWCon*stan_data$nSub, .8, 1.2), stan_data$nWCon, stan_data$nSub),
      beta  = matrix(runif(stan_data$nWCon*stan_data$nSub, .45, .55), stan_data$nWCon, stan_data$nSub),
      ndt = matrix(runif(stan_data$nWConTarget*stan_data$nSub, .05, .1), stan_data$nWConTarget, stan_data$nSub),
      delta  = matrix(runif(stan_data$nWConTarget*stan_data$nSub, -.2, .2), stan_data$nWConTarget, stan_data$nSub),
      
      sv = runif(1, .1, 3.5),
      sw = runif(1, .05, .2),
      st = runif(1, .01, .2)
    )
  } # end for
  return (L)
}


# compile the model
file = paste0(base_dir, "/Reanalysis_Study_1/models/S3_pleskac_", args$model, ".stan")
mc = cmdstan_model(file, cpp_options = list(stan_threads = T))

print(paste0('Sampling model ', args$model))
# do the sampling
L = init.stan()
m = mc$sample(data=stan_data,
              init = L,
              max_treedepth = max_treedepth,
              adapt_delta = 0.8,
              refresh = 50,
              iter_sampling = args$sampling,
              iter_warmup = args$warmup,
              thin = args$thin_to,
              chains = 4,
              parallel_chains = 1,
              threads_per_chain = args$threads_per_chain,
              output_dir = output_dir,
              output_basename = 'fit',
              save_warmup = TRUE)

summary = m$summary()
summary = summary[summary$variable != "lp__",]
sink(file.path(output_dir, "inference.txt"))
print(m$cmdstan_summary())
sink()
file.copy(file, output_dir)

# check whether diagnotic criteria for NEff and r_hat are fine
NEff_val = min(summary$ess_bulk)
Rhat_val = max(summary$rhat)

# if not, move the  directory
if (NEff_val < 400 | Rhat_val > 1.01) {
  failed_dir = paste0(base_dir, "/Reanalysis_Study_1/failed_fits/", args$model, format(Sys.time(), format="_%y%m%d-%H%M%S"))
  dir.create(failed_dir, recursive = TRUE, showWarnings = FALSE)
  file.move(output_dir, failed_dir)
  print(paste0('Moved to failed_fits.'))
  print(paste0('NEff = ', NEff_val, ', Rhat_val = ', Rhat_val))
} #end if move
