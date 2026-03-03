library(cmdstanr)
suppressPackageStartupMessages(library(tidyverse))
library(rlist)  #for list.rbind()
library(doParallel)
library(optparse) #for input arguments
library(WienR)


source("setup.local.R")

argument_list = list (
  make_option(c('-f', '--folder'), type='character', default='_basic',
              help='empty or _basic', metavar='character'),
  make_option(c('-c', '--cores'), type='integer', default='4',
              help='number cores for pdf computation', metavar='integer'),
  make_option(c('-l', '--subject_range_lower'), type='integer', default=10,
              help='where to start sampling', metavar='integer'),
  make_option(c('-u', '--subject_range_upper'), type='integer', default=20,
              help='where to end sampling', metavar='integer')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

data_dir =  paste0("S3_truncated", args$folder)
files = list.dirs(data_dir, full.names=F, recursive=F)

results_dir <- paste0("results", args$folder)
if (!dir.exists(results_dir)) {
  dir.create(results_dir, showWarnings = FALSE)
}
if (paste0("lp_ranks_", args$subject_range_lower, ".csv") %in% list.files(results_dir)) {
  print(paste0("lp_ranks_", args$subject_range_lower, ".csv existiert bereits"))
} else {

  cl <- makeCluster(args$cores)  #make cluster with number of cores
  registerDoParallel(cl) #register cluster to be used by %dopar%
  
  foreach::getDoParRegistered() #check if registered
  foreach::getDoParWorkers() #check how many workers are availabe



  
  # Preparation of the data
  pars_summary  = list("a"="a", "w"="zr_m", "v1"="v_m[1]", "v2"="v_m[2]", "t0"="t0_m", "sv"="v_s", "sw"="zr_s", "st"="t0_s")
  
  if(grepl('basic', args$folder)) {
    pars_summary  = pars_summary[1:5]
  }
  
  L = 399
  
  trials = c(100, 500)
  #lp_ranks = as.data.frame(matrix(nrow=length(files)*length(trials), ncol=3))
  lp_ranks = data.frame()
  #names(lp_ranks) = c("parameter_set", "trial_number", "lp_rank")
  
  subject_range = args$subject_range_lower:args$subject_range_upper   #set subject range for each analysis-group
  
  print(paste("Extracting subjects:", str(subject_range))) 
  for (i in subject_range) {
    parameter_set = files[i] 
    true_params = read.csv(file.path(data_dir, parameter_set, "params.csv"))
    for (NTrials in 1:length(trials)) { 
      base_path = file.path(data_dir, parameter_set, trials[NTrials], paste0("fit", args$folder))
      if (!("inference.txt" %in% list.files(base_path)))  # directory with completed fit
        next
      print(paste0("Read parameter_set ", parameter_set, ", trials ", trials[NTrials]))
      fit = as_cmdstan_fit(paste0(base_path,  "/fit-", 1:4, ".csv"))
  
      # SBC
      num_draws <- 4 * fit$metadata()$iter_sampling
      SBC_thinning = floor(num_draws / L)
      SBC_thinned_idx <- seq(1, by = SBC_thinning, length.out = L)
      SBC_draws <- fit$draws(variables = c(unlist(pars_summary), lp="lp__"), format = "data.frame")[SBC_thinned_idx,]
      data = read.csv(file.path(data_dir, parameter_set, trials[NTrials], "data.csv"))
      names(data) = c("X", "cond", "resp", "rt")
      if (args$folder == '_basic_fix') {
        true_lp = sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=true_params$a[1], 
                                              v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                              w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                              st0=0)[2]))
        true_lp = true_lp - sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=data$resp+1, a=true_params$a[1], 
                                               v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                               w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                               st0=0)[2]))
    
        SBC_lp = foreach (sbc = 1:L) %dopar% {
          sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                      st0=0)[2]))
        }
        SBC_lc = foreach (sbc = 1:L) %dopar% {
          sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=data$resp+1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                      st0=0)[2]))
        }
        SBC_lp = unlist(SBC_lp) - unlist(SBC_lc)
      } else if (args$folder %in% c('_basic_fix_cdf1_cdf0', '_basic_fix_cdf1_cdf0_log_sum_exp')) {
        true_cdf_0 = unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=1, a=true_params$a[1], 
                                             v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                             w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                             st0=0)[1])
        true_cdf_1 = unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=2, a=true_params$a[1], 
                                             v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                             w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                             st0=0)[1])
        true_lp = sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=true_params$a[1], 
                                              v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                              w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                              st0=0)[2]))
        true_lp = true_lp - sum(log(true_cdf_0 + true_cdf_1))
        
        SBC_lp = foreach (sbc = 1:L) %dopar% {
          sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                      st0=0)[2]))
        }
        SBC_lc = foreach (sbc = 1:L) %dopar% {
          sum(log(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                      st0=0)[1]) +
          unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=2, a=SBC_draws[[sbc,"a"]], 
                                  v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                  w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                  st0=0)[1])))
        }
        SBC_c_1 = foreach (sbc = 1:L) %dopar% {
          unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=2, a=SBC_draws[[sbc,"a"]], 
                                  v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                  w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                  st0=0)[1])
        }
        SBC_lp = unlist(SBC_lp) - unlist(SBC_lc)
        
      } else if (args$folder == '_basic_cens'){
        data_cens = data[data$rt < true_params$upper_bound[NTrials],]
        rest = data[data$rt > true_params$upper_bound[NTrials],]
        if (length(data_cens$rt) > 0) {
          true_lp = sum(unlist(WienR::WienerPDF(t=data_cens$rt, response=data_cens$resp+1, a=true_params$a[1], 
                                                v=ifelse(data_cens$cond==1, true_params$v[1], true_params$v[2]), 
                                                w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                                st0=0)[2]))
        } else {
          true_lp = 0
        }
        if (length(rest$rt) > 0) {
          lccdf = length(rest$rt) * (sum(unlist(WienR::WienerCDF(t=1000, response=rest$resp+1, a=true_params$a[1], 
                                              v=ifelse(rest$cond==1, true_params$v[1], true_params$v[2]), 
                                              w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                              st0=0)[2])) - 
                                       sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=rest$resp+1, a=true_params$a[1], 
                                   v=ifelse(rest$cond==1, true_params$v[1], true_params$v[2]), 
                                   w=true_params$w[1], t0=true_params$t0[1], sv=0, sw=0, 
                                   st0=0)[2])))
        } else {
          lccdf = 0
        }
        true_lp = true_lp + lccdf
        
        if (length(data_cens$rt) > 0) {
          SBC_lp = foreach (sbc = 1:L) %dopar% {
            sum(unlist(WienR::WienerPDF(t=data_cens$rt, response=data_cens$resp+1, a=SBC_draws[[sbc,"a"]], 
                                        v=ifelse(data_cens$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                        w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                        st0=0)[2]))
          }
        } else {
          SBC_lp = 0
        }
        if (length(rest$rt) > 0) {
          SBC_lcc = foreach (sbc = 1:L) %dopar% {
            length(rest$rt) * (sum(unlist(WienR::WienerCDF(t=1000, response=rest$resp+1, a=SBC_draws[[sbc,"a"]], 
                                        v=ifelse(rest$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                        w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                        st0=0)[2])) -
                                 sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=rest$resp+1, a=SBC_draws[[sbc,"a"]], 
                                                             v=ifelse(rest$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                                             w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=0, sw=0, 
                                                             st0=0)[2])))
          }
        } else {
          SBC_lcc = 0
        }
        SBC_lp = unlist(SBC_lp) + unlist(SBC_lcc)
      } else {
        true_lp = sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=true_params$a[1], 
                                              v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                              w=true_params$w[1], t0=true_params$t0[1], sv=true_params$sv, sw=true_params$sw, 
                                              st0=true_params$st0)[2]))
        true_lp = true_lp - sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=data$resp+1, a=true_params$a[1], 
                                              v=ifelse(data$cond==1, true_params$v[1], true_params$v[2]), 
                                              w=true_params$w[1], t0=true_params$t0[1], sv=true_params$sv, sw=true_params$sw, 
                                              st0=true_params$st0)[2])) 
        
        SBC_lp = foreach (sbc = 1:L) %dopar% {
          sum(unlist(WienR::WienerPDF(t=data$rt, response=data$resp+1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=SBC_draws[[sbc,"v_s"]], sw=SBC_draws[[sbc,"zr_s"]], 
                                      st0=SBC_draws[[sbc,"t0_s"]])[2]))
        }
        SBC_lc = foreach (sbc = 1:L) %dopar% {
          sum(unlist(WienR::WienerCDF(t=true_params$upper_bound[NTrials], response=data$resp+1, a=SBC_draws[[sbc,"a"]], 
                                      v=ifelse(data$cond==1, SBC_draws[[sbc,"v_m[1]"]], SBC_draws[[sbc,"v_m[2]"]]), 
                                      w=SBC_draws[[sbc,"zr_m"]], t0=SBC_draws[[sbc,"t0_m"]], sv=SBC_draws[[sbc,"v_s"]], sw=SBC_draws[[sbc,"zr_s"]], 
                                      st0=SBC_draws[[sbc,"t0_s"]])[2]))
        }
        SBC_lp = unlist(SBC_lp) - unlist(SBC_lc)
      } # else
      index = nrow(lp_ranks) + 1
      #lp_ranks[(i-1)*2+NTrials, "parameter_set"] = parameter_set
      #lp_ranks[(i-1)*2+NTrials, "trial_number"] = trials[NTrials]
      #lp_ranks[(i-1)*2+NTrials, "lp_rank"] = suppressWarnings(sum(SBC_lp < true_lp))
      lp_ranks[index, "parameter_set"] = parameter_set
      lp_ranks[index, "trial_number"] = trials[NTrials]
      lp_ranks[index, "lp_rank"] = suppressWarnings(sum(SBC_lp < true_lp))
    }
  } # for files
    
    
  #head(lp_ranks)
  stopCluster(cl) #advice: stop the cluster at the end!
  
  
  if(length(subject_range) < 2000) {
    write.csv(lp_ranks, file.path(results_dir, paste0("lp_ranks_", args$subject_range_lower, ".csv")))
  } else {
    write.csv(lp_ranks, file.path(results_dir, "lp_ranks.csv"))
  }
}

