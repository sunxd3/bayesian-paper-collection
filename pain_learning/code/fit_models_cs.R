### - Model fitting code, run using terminal where you specify 5 arguments - specM_fname,list_names,specS_fname,cond_name_index,sup_results_dir
# Initialise -------------
library(rstan)
library(loo)
library(data.table)

cond_names = c('LVLS', 'LVHS', 'HVLS', 'HVHS')
if (Sys.info()["sysname"]=="Darwin"){
  library(shinystan)
  # setwd('/Volumes/GoogleDrive/Shared drives/Cam Nox Lab/tsl_paper')
  setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/pain_control_git/paper_submission/experiment1')
  specM_fname <- commandArgs(trailingOnly = TRUE)[1]
  list_names <- commandArgs(trailingOnly = TRUE)[2]
  specS_fname <- commandArgs(trailingOnly = TRUE)[3]
  cond_name_index <- as.integer(commandArgs(trailingOnly = TRUE)[4])
  sup_results_dir <- commandArgs(trailingOnly = TRUE)[5]
  if (is.na(specM_fname)){
    specM_fname <-'m1_RL_sliced2_halfStudentT_reparam_confEs_norm.stan' #example, from IDE
    list_names <- 'All'
    specS_fname <-'spec_test'
    cond_name_index <- 1
    sup_results_dir <- ''
  }
  rstan_options(auto_write = TRUE)
}else{
  setwd('...')
  specM_fname <- commandArgs(trailingOnly = TRUE)[1] #model spec txt, contains parameter list
  list_names <- commandArgs(trailingOnly = TRUE)[2] #group names
  specS_fname <- commandArgs(trailingOnly = TRUE)[3] #stan fitting spec - cores, iterations, warmup, adapt delta, max tree depth
  cond_name_index <- as.integer(commandArgs(trailingOnly = TRUE)[4]) #name of the condition
  sup_results_dir <- commandArgs(trailingOnly = TRUE)[5] #directory where to save stan results and parameter CSVs
  rstan_options(auto_write = FALSE)
}
source('stan_utility.R')
cond_name = cond_names[cond_name_index]
print(specM_fname)
print(cond_name_index)
print(cond_name)
# rel_dir = 'try_nick/jo_modelling/'
rel_dir = 'model_fit_analysis/'
date.time.append <-dget("date_time_append.R")
script_stamp = date.time.append(paste("_",as.character(round(runif(1)*10e2)),sep=''))

# Load and set up specs, dirs etc. -------------
specsM=transpose(read.table(file=paste(rel_dir,'hpc/specs/models/',specM_fname,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
colnames(specsM) <- as.character(specsM[1, ])
specsM <- specsM[-1,]
rownames(specsM)<-NULL

specsS=transpose(read.table(file=paste(rel_dir,'hpc/specs/',specS_fname,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
colnames(specsS) <- as.character(specsS[1, ])
specsS <- specsS[-1,]
rownames(specsS)<-NULL

stan_cores = as.integer(specsS$stanCores)
iters = as.integer(specsS$stanIters)
warmups = as.integer(specsS$stanWarmup)
adapt_delta_value = as.double(specsS$stanAdaptDelta)
max_treedepth_value = as.double(specsS$stanMTD)
shortname = specM_fname
model_pars_list=eval(parse(text=specsM$stanParamList))

options(mc.cores = stan_cores)

specfname = gsub('\\.','-',paste('iter',iters,'delta',adapt_delta_value,'mtd',max_treedepth_value,sep='_'));
model_name = paste(rel_dir,'models/',shortname,'.stan',sep='')
model_name_short = paste(shortname,cond_name,sep='_')
model_par_names = model_pars_list
model_par_names_full = model_par_names

outputdir = paste(rel_dir,'output/',sup_results_dir,'/',cond_name,'/',shortname,'_',cond_name,'/',sep='')
if(!file.exists(outputdir)){dir.create(outputdir,showWarnings = TRUE,recursive = TRUE)}

# Load/save data -------------
list_names = 'All' # which group - 'All' corresponds to just one group

# model_data=readRDS(dir('model_fit_analysis/data/',full.names = T,pattern = paste('.*lin.*',cond_name,'.rds',sep='')))
model_data=readRDS(dir(paste(rel_dir,'data/',sep=''),full.names = T,pattern = paste('.*lin.*',cond_name,'.rds',sep='')))
stan_data = list(N=model_data$N,Tn=model_data$Tn,N_obs_VAS=model_data$N_obs_VAS,N_obs_CP=model_data$N_obs_CP,
                 noxInputTrans=model_data$TransfIntesSeq,painRating=model_data$PercVals,painPred=model_data$PredVals,
                 RatingConf = model_data$PercConf,PredConf = model_data$PredConf,
                 TrialType=model_data$TrialType,PainValsAll=model_data$PainValsAll,PercIndexArrayLong=model_data$PercIndexArrayLong,
                 PredIndexArrayLong=model_data$PredIndexArrayLong,PercIndexArray=model_data$PercIndexArray,
                 PredIndexArray=model_data$PredIndexArray,IndexMissingAll=model_data$IndexMissingAll,TrialTypeAllWMissing=model_data$TrialTypeAllWMissing)
# Setup and run -------------
fitted_params = list()
fitted_params_summary = list()
save_outputs=TRUE

print(paste('started fitting data for',list_names,'with',model_name_short))
print(paste('stan specs are: ',specfname,sep=''))

HBA_ind_fit = stan(file=model_name,data=stan_data,cores=stan_cores,verbose=FALSE,save_warmup=FALSE,pars=c('lp_'),include=FALSE,
                   iter=iters,warmup=warmups, control = list(adapt_delta=adapt_delta_value,max_treedepth=max_treedepth_value))

warnings()

# Save results ----
print(paste('done fitting data for',list_names,'with',model_name_short))
try(if (save_outputs){
  rdsName = paste(outputdir,"HBA_ind_fit",'_',list_names,'_',model_name_short,'_',specfname,script_stamp,'.rds',sep='')
  print(rdsName)
  saveRDS(HBA_ind_fit,file=rdsName)
  print(paste('done saving fitted parameters for',list_names,'with',model_name_short))
})

fitted_ind_params = list();
for (j in 1:length(model_par_names)){
  fitted_ind_params[[length(fitted_ind_params)+1]] <-summary(HBA_ind_fit,pars=model_par_names[j])$summary[,'mean']
}

fitted_params[[length(fitted_params)+1]] <-fitted_ind_params

fitted_params_ind_summary = list()
for (j in 1:length(model_par_names_full)){
  fitted_params_ind_summary[[length(fitted_params_ind_summary)+1]]=c(mean(fitted_params[[length(fitted_params)]][[j]]),sd(fitted_params[[length(fitted_params)]][[j]]))
}
names(fitted_params_ind_summary)<-model_par_names_full
fitted_params_summary[[length(fitted_params_summary)+1]]=fitted_params_ind_summary

fitted_params_ind_summary_df = data.frame(fitted_params_ind_summary)
names(fitted_params_ind_summary_df)<-model_par_names_full
fitted_ind_params_df<-data.frame(fitted_ind_params)
names(fitted_ind_params_df)<-model_par_names_full
rownames(fitted_ind_params_df)<-NULL

group_par_names = c(HBA_ind_fit@model_pars[grepl('mu_pr.*',HBA_ind_fit@model_pars)],'sigma')
group_pars= data.frame(summary(HBA_ind_fit,pars=group_par_names)$summary[,'mean'])
names(group_pars) <-'value'

try(if (save_outputs){
  write.csv(fitted_params_ind_summary_df,file=paste(outputdir,"HBA_summary_",list_names,'_',model_name_short,'_',specfname,script_stamp,'.csv',sep=''),row.names=FALSE)
  write.csv(fitted_ind_params_df,file=paste(outputdir,"HBA_ind_fit",'_',list_names,'_',model_name_short,'_',specfname,script_stamp,'.csv',sep=''),row.names=FALSE)
  write.csv(group_pars,file=paste(outputdir,'group_pars_',list_names,'_',model_name_short,'_',specfname,script_stamp,'.csv',sep=''),row.names=TRUE)
  print(paste('saved parameters and parameter summary for',list_names,'with',model_name_short))
})

warnings()