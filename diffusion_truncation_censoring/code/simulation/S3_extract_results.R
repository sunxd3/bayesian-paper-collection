#Analyze results from study 3. We want to report:
#1. mcmc_neff (Visuell)
#2. mcmc_rhat (Visuell)
#3. Violin-Plots for squared Bias (Visuell) -> hier vielleicht auch die Diagonaldiagramme ...
#4. Coverage (Tabelle)
#5. Mean MCSE (Tabelle)
#6. Correlations: posterior median with true value
#Table: Param, Correlation, Coverage 50%, Coverage 95%, MCSE


library(cmdstanr)
suppressPackageStartupMessages(library(tidyverse))
library(bayesplot)
library(bayestestR)
library(mcmcse)
library(rlist)  #for list.rbind()
library(optparse) # for input arguments

source("setup.local.R")

argument_list = list (
  make_option(c('-f', '--folder'), type='character', default='_basic_fix',
              help='empty or _basic', metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)


data_dir =  paste0("S3_truncated", args$folder)
files = list.dirs(data_dir, full.names=F, recursive=F)

# Preparation of the data
names_params = c("a", "v1", "v2", "t0", "w", "sv", "sw", "st")
pars_params = list("a"="a", "w"="w", "t0"="t0", "sv"="sv", "sw"="sw", "st"="st0")
pars_summary  = list("a"="a", "w"="zr_m", "v1"="v_m[1]", "v2"="v_m[2]", "t0"="t0_m", "sv"="v_s", "sw"="zr_s", "st"="t0_s")

if(grepl('basic', args$folder)) {
  names_params = names_params[1:5]
  pars_params = pars_params[1:3]
  pars_summary  = pars_summary[1:5]
}

results = data.frame()

trials = c(100, 500)

L = 399
for (i in 1:length(files)) {
  parameter_set = files[i] 
  true_params = read.csv(file.path(data_dir, parameter_set, "params.csv"))
  for (NTrials in 1:length(trials)) { 
    base_path = file.path(data_dir, parameter_set, trials[NTrials], paste0("fit", args$folder))
    if (!("inference.txt" %in% list.files(base_path))) { # directory with completed fit?
      next
    }
    print(paste0("Read parameter_set ", parameter_set, ", trials ", trials[NTrials]))
    fit = as_cmdstan_fit(paste0(base_path,  "/fit-", 1:4, ".csv"))
    timetosample <- sum(fit$time()$chains[,4])
    summary = fit$summary()
    hdi_50 = bayestestR::hdi(fit$draws(), ci=.50)
    hdi_95 = bayestestR::hdi(fit$draws(), ci=.95)
    # SBC
    num_draws <- 4 * fit$metadata()$iter_sampling
    SBC_thinning = floor(num_draws / L)
    SBC_thinned_idx <- seq(1, by = SBC_thinning, length.out = L)
    SBC_draws <- fit$draws(variables = unlist(pars_summary), format = "data.frame")[SBC_thinned_idx,]

    for (par in names_params) {
      if (pars_summary[[par]] %in% summary$variable) {
        index = nrow(results) + 1
        results[index,"parameter"] = par
        results[index,"parameter_set"] = parameter_set
        results[index,"trial_number"] = trials[NTrials] 
        if (par == "v1")      {results[index,"true"] = true_params[["v"]][1]}
        else if (par == "v2") {results[index,"true"] = true_params[["v"]][2]}
        else                  {results[index,"true"] = true_params[[pars_params[[par]]]][1]}
        results[index,"50%"] = summary$median[which(summary$variable == pars_summary[[par]])] 
        results[index,"mean"] = summary$mean[which(summary$variable == pars_summary[[par]])]
        results[index,"sd"] = summary$sd[which(summary$variable == pars_summary[[par]])]
        results[index,"bias(50%-true)"] = results[index,"50%"] - results[index,"true"]
        results[index,"50%-HDI_low"] = hdi_50$CI_low[which(hdi_50$Parameter == pars_summary[[par]])] 
        results[index,"50%-HDI_high"] = hdi_50$CI_high[which(hdi_50$Parameter == pars_summary[[par]])] 
        results[index,"95%-HDI_low"] = hdi_95$CI_low[which(hdi_95$Parameter == pars_summary[[par]])] 
        results[index,"95%-HDI_high"] = hdi_95$CI_high[which(hdi_95$Parameter == pars_summary[[par]])]
        # suppress Warning message: Dropping 'draws_df' class as required metadata was removed.
        results[index,"rank"] = suppressWarnings(sum(SBC_draws[,pars_summary[[par]]] < results[index,"true"]))
        results[index,"rank_equals"] = suppressWarnings(sum(SBC_draws[,pars_summary[[par]]] == results[index,"true"]))
        results[index,"NEff"] = summary$ess_bulk[which(summary$variable == pars_summary[[par]])]
        results[index,"NEff_ratio"] = neff_ratio(fit)[which(names(neff_ratio(fit)) == pars_summary[[par]])]
        results[index,"Rhat"] = summary$rhat[which(summary$variable == pars_summary[[par]])]
        results[index,"timetosample"] = timetosample
        results[index,"threads_per_chain"] = fit$metadata()$threads_per_chain
        results[index,"upper_bound"] = true_params$upper_bound[NTrials]
      }
    } 
  }
}  

results_dir = paste0("results", args$folder)
if (!dir.exists(results_dir)) {
  dir.create(results_dir, showWarnings = FALSE)
}
write.csv(results, file.path(results_dir, "results.csv"))

