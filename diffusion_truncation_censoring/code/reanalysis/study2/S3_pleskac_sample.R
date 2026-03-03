#sample Study2 of Pleskac
#N=116, NTrials=80
#within-subject manipulation: race, object
#between-subject manipulation: dangerousness of the context
library(cmdstanr)
library(parallel) # for detectCores()
library(ff) # for moving directories
library(bayesplot)
library(optparse) # for input arguments
library(tidyverse)

## IMPORTANT: Set cmdstan location:
# Create a local setup.local.R file that calls set_cmdstan_path():
# $ cp setup.template.R setup.local.R
# Set the cmdstan location in setup.local.R. This preserves the location
# even after future code changes
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
  make_option(c('-m', '--model'), type='character', default='basicS2_cens',
              help='model: basicS2, basicS2_trunc, basicS2_cens, fullS2'
              , metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

## Set configuration and choose which setup to sample with input arguments
max_treedepth = 5 # 5 or 10


#read in data and separate censored data from the rest
base_dir = 'analysis_pleskac/Reanalysis_Study_2/'

#data_alt = read_csv('analysis_pleskac/Pleskac_Daten/Study2/Study2TrialData.csv')
data = read_csv('analysis_pleskac/Pleskac_Daten/Study2/dataTable_fromPleskacViaMail.csv') %>%
  rename('Subject' = 'subject', 'Race012B' = 'race', 'RT' = 'rt',
         'upper' = 'upperCensorLimit', 'Resp0NS1Sh' = 'resp',
         'Object0NG1G' = 'targetOb', 'Context1Safe2Danger' = 'targetDanger',
         'RaceObject' = 'conditionRT')


data_cens = data[is.na(data$RT),]
data_rest = data[!is.na(data$RT),]



output_dir = file.path(base_dir, 'fits', paste0('fit_', args$model))
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}
if ("inference.txt" %in% list.files(output_dir)) { # check if fit was already done
  next
}

#Data processing
data_for_stan = data_rest
if (grepl('cens', args$model)) {
  data_for_stan = data
  data_for_stan[is.na(data_for_stan$RT),]$RT = 42000
  data_for_stan[is.na(data_for_stan$Resp0NS1Sh),]$Resp0NS1Sh = 42
  
  #to model between subject condition separately
  #data_for_stan = data_for_stan[data_for_stan$Context1Safe2Danger == 0,]
}

stan_data = list(
  nSub = length(unique(data_for_stan$Subject)),    
  nBtwn = length(unique(data_for_stan$Context1Safe2Danger)),
  nWConTarget = length(unique(data_for_stan$RaceObject)),
  nWCon = length(unique(data_for_stan$Race012B)), # 0 White, 1 Black
  nData = length(data_for_stan$Subject),
  wCon = data_for_stan$Race012B + 1, # 1 W, 2 B
  wConTarget = data_for_stan$RaceObject, # 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
  sub = data_for_stan$Subject,
  btwnCon = data_for_stan$Context1Safe2Danger + 1, # 1 safe, 2 danger
  rt = data_for_stan$RT/1000, # in seconds
  resp = data_for_stan$Resp0NS1Sh, # variable data_for_stan$Resp0NS1Sh is response coded: 0 = tool/NS, 1 = gun/S
  upper_bound = data$upper[1], # 631 ms in Experiment 2
  y_bin = data$ybin #data$ybin: 0 lower boundary (NG?), 2 upper boundary (G?)
)

min_rt = min(data_for_stan$RT)/1000
# configure init values and fixed values for different models
init.stan = function(){
  L = list()
  for (i in 1:4) {
    L[[i]]  = list(
      muAlpha  = matrix(runif(stan_data$nWCon*stan_data$nBtwn, .2, 1.2), stan_data$nWCon, stan_data$nBtwn),
      muBeta  = matrix(runif(stan_data$nWCon*stan_data$nBtwn, .45, .55), stan_data$nWCon, stan_data$nBtwn),
      muNDT = matrix(runif(stan_data$nWConTarget*stan_data$nBtwn, .05, .1), 
                     stan_data$nWConTarget, stan_data$nBtwn),
      muDelta = matrix(runif(stan_data$nWConTarget*stan_data$nBtwn, -.2, .2), 
                    stan_data$nWConTarget, stan_data$nBtwn),
      
      sigmaAlpha = 0.4 + runif(stan_data$nBtwn, -0.05, 0.05),
      sigmaBeta = 0.1 + runif(stan_data$nBtwn, -0.05, 0.05),
      sigmaDelta = 0.4 + runif(stan_data$nBtwn, -0.05, 0.05),
      sigmaNDT = 0.1 + runif(stan_data$nBtwn, -0.05, 0.05),
      
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
model = args$model
file = paste0(base_dir, "/models/S3_pleskac_", model, ".stan")
mc = cmdstan_model(file, cpp_options = list(stan_threads = T))

print(paste0('Sampling model ', model))
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
  failed_dir = paste0(base_dir, "/failed_fits/", args$model, format(Sys.time(), format="_%y%m%d-%H%M%S"))
  dir.create(failed_dir, recursive = TRUE, showWarnings = FALSE)
  file.move(output_dir, failed_dir)
  print(paste0('Moved to failed_fits.'))
  print(paste0('NEff = ', NEff_val, ', Rhat_val = ', Rhat_val))
} #end if move
