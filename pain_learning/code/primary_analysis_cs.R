# Initialise -------------
if (Sys.info()["sysname"]=="Darwin"){
  setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/elife_revision/')
  # setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/pain_control_git/paper_submission/experiment1')
  save_to_csv=FALSE
  save_plots=TRUE
  load_csv=TRUE
  
  post_analysis=TRUE #local analysis after the hpc initial analysis run
  plot_together=TRUE
  do_plot_seq = TRUE
  
  out_dir_sh = commandArgs(trailingOnly = TRUE)[1]
  m_no = commandArgs(trailingOnly = TRUE)[2]
  if (is.na(out_dir_sh)){
    out_dir_sh = 'cs_results' #example, from IDE
  }
  if (is.na(m_no)){
    m_no=''
  }
}else{
  setwd('...')
  
  save_to_csv=TRUE
  save_plots=TRUE
  load_csv=FALSE
  
  post_analysis=FALSE
  do_plot_seq = TRUE
  
  out_dir_sh = commandArgs(trailingOnly = TRUE)[1]
  m_no=''
}
# Load libraries and scripts -------------
source('stan_utility.R')
get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}
library(rstan)
library(loo)
library(data.table)
library(reshape)
library(gridExtra)
library(grid)
library(gtools)
library(bayesplot)
library(ggsci)
library(ggpubr)
library(bayestestR)

# Set directories -------------
output_dir_res = paste('model_fit_analysis/output/',out_dir_sh,sep='')
extra_analysis_res = paste('model_fit_analysis/output/extra_analysis/',sep='')
specs_dir = 'model_fit_analysis/hpc/specs/models/'
specific_model = paste('m',m_no,'.*sliced.*',sep='') 
specfic_run = ''
model_type = '.*'

results_sub_dir = ''
anres_plots_dir = paste(output_dir_res,"/",results_sub_dir,"/plots/series",sep='')
if(!file.exists(anres_plots_dir)){dir.create(anres_plots_dir,showWarnings = TRUE,recursive = TRUE)}
anres_plots_dir = paste(output_dir_res,"/",results_sub_dir,"/plots/ind",sep='')
if(!file.exists(anres_plots_dir)){dir.create(anres_plots_dir,showWarnings = TRUE,recursive = TRUE)}
anres_plots_dir = paste(output_dir_res,"/",results_sub_dir,"/plots/group",sep='')
if(!file.exists(anres_plots_dir)){dir.create(anres_plots_dir,showWarnings = TRUE,recursive = TRUE)}
anres_plots_dir = paste(output_dir_res,"/",results_sub_dir,"/plots/traces",sep='')
if(!file.exists(anres_plots_dir)){dir.create(anres_plots_dir,showWarnings = TRUE,recursive = TRUE)}


# Set extra KF param names -------------
kf_ind_extra = c('final_alpha','final_gam','final_w','avg_alpha','avg_gam')
kf_group_extra = c('mu_final_alpha','mu_final_gam','mu_final_w','mu_avg_alpha','mu_avg_gam')
kf_np_ind_extra = c('final_alpha','final_w','avg_alpha')
kf_np_group_extra = c('mu_final_alpha','mu_final_w','mu_avg_alpha')
letters_list = toupper(letters)

# Prepare lists and load stan fit objects -------------
cond_names = c('LVLS', 'LVHS', 'HVLS', 'HVHS')
cond_names_paper_order = c('HVHS','HVLS','LVHS','LVLS')
cond_names_paper_order_labes = c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low')
cond_names_alter = c('High AC 2','Low AC 2','High AC 1','Low AC 1')
rename_me2 = TRUE

stoch_ac_name = 'Stochasticity'
vol_inst_name = 'Volatility'

col_list = c('#117733','#882255','#44AA99','#AA4499')

groups = c('All')
pptIds = read.table('model_fit_analysis/data/pptIds.csv',header=TRUE)
g=1
if (post_analysis){ #analysis after HPC run
  # modelNames = list('m2_KF_sliced2_halfStudentT_reparam_confP_norm') #winning model
  modelNames = list('m2_KF_sliced2_halfStudentT_reparam_confEs_norm') #winning model
  model_list = c('m2_KF_sliced2_halfStudentT_reparam_confEs_norm') #winning model
  modelNamesTitle = list('eKF - expectation weighted')
  library(latex2exp)
  
  par_gr_latex_names = c("$\\epsilon_{\\mu}","$s_{\\mu}","$v_{\\mu}","$\\xi_{\\mu}","$E^0_{\\mu}","$w^0_{\\mu}","$C_{\\mu}")
  par_sig_latex_names = c("$\\epsilon_{\\sigma}","$s_{sigma}","$v_{\\sigma}","$\\xi_{\\sigma}","$E^0_{\\sigma}","$w^0_{\\sigma}","$C_{\\sigma}")
  par_gr_latex_names_extra = c("$\\alpha^{f}_{\\mu}$","$\\gamma^{f}_{\\mu}$","$w^{f}_{\\mu}$","$\\bar{\\alpha}_{\\mu}$","$\\bar{\\gamma}_{\\mu}$")
  par_ind_latex_names = c("$\\epsilon","$s","$v","$\\xi","$E^0","$w^0","C")
  par_ind_latex_names_extra = c("$\\alpha^{f}$","$\\gamma^{f}$","$w^{f}$","$\\bar{\\alpha}$","$\\bar{\\gamma}$")
  gr_lims = c(4,4,0.15,5.5,4.5,0.8,5)
}else{
  specNames = list(vector(mode = "list", length = length(cond_names)))
  stan_fits = list(vector(mode = "list", length = length(cond_names)))
  file_names = vector(mode = "list", length = length(cond_names))
  
  modelNames = vector(mode = "list", length = length(cond_names))
  loo_comp_list = vector(mode = "list", length = length(cond_names))
  for (c in 1:length(cond_names)){
    specNames[[c]] = dir(paste(output_dir_res,'/',cond_names[c],sep=''),full.names = T,pattern = paste('',specific_model,sep=''))
    loo_comp_list[[c]] = vector(mode = "list", length = length(specNames[[c]]))
    modelNames[[c]] = basename(specNames[[c]])
  }
  
  if (!load_csv){
    ## Load stan fit objects ----
    print('Started loading stan RDS files')
    for (c in 1:length(cond_names)){
      stan_fits[[c]] = list(vector(mode = "list", length = length(specNames[[c]])))
      file_names[[c]] = vector(mode = "list", length = length(specNames[[c]]))
      for (n in 1:length(specNames[[c]])){
        print(specNames[[c]][n])
        file_names[[c]][[n]]= dir(specNames[[c]][n],full.names=T,pattern=paste('.*',groups[g],'.*_',model_type,'_.*',specfic_run,'.*.rds',sep=''))[1];
        stan_fits[[c]][[1]][[n]] = readRDS(file_names[[c]][[n]])
      }
    }
    ## Run diagnostics for each model -------------
    diagnosticsResults = data.frame()
    options(warn=-1)
    sink('nul')
    print('Started loading diagnostics')
    for (c in 1:length(cond_names)){
      for (n in 1:length(modelNames[[c]])){
        print(modelNames[[c]][n])
        g=1
        if (!is.null(stan_fits[[c]][[g]][[n]])){
          string_split = strsplit(file_names[[c]][[n]],'_')[[1]]
          L = length(string_split)
          toJoin = sub('.rds','',string_split[(L-1):L])
          mID = paste(toJoin[1:2],sep='',collapse='')
          loo(extract_log_lik(stan_fits[[c]][[g]][[n]]))
          paret_vals = loo(extract_log_lik(stan_fits[[c]][[g]][[n]]))$diagnostics$pareto_k
          pareto_dist = toString(c(sum(paret_vals<=0.5), sum(paret_vals>0.5 & paret_vals<=0.7), sum(paret_vals>0.7 & paret_vals<=1), sum(paret_vals>1)))
          diagnosticsResults=rbind(diagnosticsResults,cbind(mID,modelNames[[c]][n],cond_names[[c]],groups[g],loo(extract_log_lik(stan_fits[[c]][[g]][[n]]),save_psis = TRUE,cores=4)$estimates['looic',1],
                                                            loo(extract_log_lik(stan_fits[[c]][[g]][[n]]),save_psis = TRUE,cores=4)$estimates['looic',2],check_treedepth(stan_fits[[c]][[g]][[n]]),
                                                            check_energy(stan_fits[[c]][[g]][[n]]),check_div(stan_fits[[c]][[g]][[n]]),pareto_dist,toString(loo(extract_log_lik(stan_fits[[c]][[g]][[n]]))$diagnostics$pareto_k)))
          
          loo_comp_list[[c]][[n]] = loo(stan_fits[[c]][[g]][[n]],save_psis = TRUE,cores=4)
        }
      }
    }
    
    options(warn=0)
    colnames(diagnosticsResults)<-c('mID','modelName','cond','group','looic','looicSE','treedepth','energy','div','pareto_dist','pareto_vals')
    sink()
    
    validDiag = diagnosticsResults[as.numeric(diagnosticsResults$looic)<(min(as.numeric(diagnosticsResults$looic))*10) & !(diagnosticsResults$looicSE=='Inf'),]
    badDiag = diagnosticsResults[!(as.numeric(diagnosticsResults$looic)<(min(as.numeric(diagnosticsResults$looic))*10) & !(diagnosticsResults$looicSE=='Inf')),]
    if (save_to_csv){
      write.csv(diagnosticsResults,file=paste(output_dir_res,"/",results_sub_dir,"/diagnostics_cond",'.csv',sep=''),row.names=FALSE)
      write.csv(validDiag,file=paste(output_dir_res,"/",results_sub_dir,"/diagnostics_cond_validDiag",'.csv',sep=''),row.names=FALSE)
      write.csv(badDiag,file=paste(output_dir_res,"/",results_sub_dir,"/diagnostics_cond_badDiag",'.csv',sep=''),row.names=FALSE)
    }
    
    # Run model comparison -------------
    print('Started model comparison')
    comp_models_list = vector(mode = "list", length = length(cond_names))
    for (c in 1:length(cond_names)){
      comp_models_list[[c]] = data.frame(loo_compare(loo_comp_list[[c]][1:length(modelNames[[c]])]))
      comp_models_list[[c]]$modelName =''
      
      for (n in 1:dim(comp_models_list[[c]])[1]){
        comp_models_list[[c]]$modelName[n] = modelNames[[c]][as.integer(sub("model","",row.names(comp_models_list[[c]]))[n])]
        comp_models_list[[c]]$sigma_effect = abs(comp_models_list[[c]]$elpd_diff/comp_models_list[[c]]$se_diff)
      }
      l_c = dim(comp_models_list[[c]])[2]
      # comp_models_list[[c]] = comp_models_list[[c]][,c(dim(comp_models_list[[c]])[2],1:(dim(comp_models_list[[c]])[2]-1))]
      comp_models_list[[c]] = comp_models_list[[c]][,c((l_c-1),1:2,l_c,3:(l_c-2))]
      if (save_to_csv){
        write.csv(comp_models_list[[c]],file=paste(output_dir_res,"/",results_sub_dir,"/model_comp_cond_",cond_names[c],'.csv',sep=''),row.names=FALSE)
      }
    }
  }
}


## Group-level HDI comparison ----------------
if (!load_csv){
  print('Started extracting parameter summaries')
  condPairs = combn(1:length(stan_fits),2)
  parGroupComp = data.frame()
  parSigmaComp = data.frame()
  for (cp in 1:dim(condPairs)[2]){
    for (n in 1:length(modelNames[[1]])){
      submodelName = sub(substr(modelNames[[1]][[n]],nchar(modelNames[[1]][[n]])-5+1,nchar(modelNames[[1]][[n]])),'',modelNames[[1]][[n]])
      specsM=transpose(read.table(file=paste(specs_dir,submodelName,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
      colnames(specsM) <- as.character(specsM[1, ])
      specsM <- specsM[-1,]
      rownames(specsM)<-NULL
      par_list_ind=eval(parse(text=specsM$stanParamList))
      par_list_gr=paste('mu_',par_list_ind,sep='')
      par_list_sigma = sub('mu','sigma',par_list_gr)
      if(grepl('KF.*np',modelNames[[1]][[n]])){
        par_list_ind_extra = kf_np_ind_extra
        par_list_gr_extra = kf_np_group_extra 
      } else if(grepl('KF',modelNames[[1]][[n]])){
        par_list_ind_extra = kf_ind_extra
        par_list_gr_extra = kf_group_extra 
      } else{
        par_list_ind_extra = ''
        par_list_gr_extra = ''
      }
  
      c1 = condPairs[1,cp]
      c2 = condPairs[2,cp]
      condDiff_str = paste(cond_names[c1],'-',cond_names[c2],sep='')
      #mean
      for (p in 1:length(par_list_gr)){
        diffDisPar = extract(stan_fits[[c1]][[1]][[n]],permuted=T,pars=par_list_gr[p])[[par_list_gr[p]]] - extract(stan_fits[[c2]][[1]][[n]],permuted=T,pars=par_list_gr[p])[[par_list_gr[p]]]
        parHDI = as.numeric(hdi(diffDisPar))[2:3]
        parGroupComp = rbind(parGroupComp,cbind(submodelName,par_list_gr[p],condDiff_str,parHDI[1],parHDI[2]))
      }
      #variance
      for (p in 1:length(par_list_sigma)){
        pname = paste('sigma[',p,']',sep='')
        diffDisPar = extract(stan_fits[[c1]][[1]][[n]],permuted=T,pars=pname)[[pname]] - extract(stan_fits[[c2]][[1]][[n]],permuted=T,pars=pname)[[pname]]
        parHDI = as.numeric(hdi(diffDisPar))[2:3]
        parSigmaComp = rbind(parSigmaComp,cbind(submodelName,par_list_sigma[p],condDiff_str,parHDI[1],parHDI[2]))
      }
      
        if(grepl('KF',modelNames[[1]][[n]],fixed=TRUE)){
          for (p in 1:length(par_list_gr_extra)){
            diffDisPar = extract(stan_fits[[c1]][[1]][[n]],permuted=T,pars=par_list_gr_extra[p])[[par_list_gr_extra[p]]] - extract(stan_fits[[c2]][[1]][[n]],permuted=T,pars=par_list_gr_extra[p])[[par_list_gr_extra[p]]]
            parHDI = as.numeric(hdi(diffDisPar))[2:3]
            parGroupComp = rbind(parGroupComp,cbind(submodelName,par_list_gr_extra[p],condDiff_str,parHDI[1],parHDI[2]))
          }  
        }
        
    }
  }
  colnames(parGroupComp)<-c('modelName','parName','condDiff','L_HDI','U_HDI')
  rownames(parGroupComp) = NULL
  colnames(parSigmaComp)<-c('modelName','parName','condDiff','L_HDI','U_HDI')
  rownames(parSigmaComp) = NULL
  sigDiff = rbind(parGroupComp[(parGroupComp$L_HDI<0 & parGroupComp$U_HDI<0),],parGroupComp[(parGroupComp$L_HDI>0 & parGroupComp$U_HDI>0),])
  sigDiff_sigma = rbind(parSigmaComp[(parSigmaComp$L_HDI<0 & parSigmaComp$U_HDI<0),],parSigmaComp[(parSigmaComp$L_HDI>0 & parSigmaComp$U_HDI>0),])
  if (save_to_csv){
    write.csv(parGroupComp,file=paste(output_dir_res,"/",results_sub_dir,"/parGroupComp_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(sigDiff,file=paste(output_dir_res,"/",results_sub_dir,"/parGroupComp_cond_sigDiff",'.csv',sep=''),row.names=FALSE)
    write.csv(parSigmaComp,file=paste(output_dir_res,"/",results_sub_dir,"/parSigmaComp_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(sigDiff_sigma,file=paste(output_dir_res,"/",results_sub_dir,"/parSigmaComp_cond_sigDiff",'.csv',sep=''),row.names=FALSE)
  }
  
  ## Group-level and individual level parameters ----------------
  parIndSum = data.frame()
  parGroupSum = data.frame()
  parExtraSum = data.frame()
  gr_draws = data.frame()
  sig_draws = data.frame()
  for (c in 1:length(cond_names)){
    for (n in 1:length(modelNames[[1]])){
      submodelName = sub(substr(modelNames[[1]][[n]],nchar(modelNames[[1]][[n]])-5+1,nchar(modelNames[[1]][[n]])),'',modelNames[[1]][[n]])
      specsM=transpose(read.table(file=paste(specs_dir,submodelName,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
      colnames(specsM) <- as.character(specsM[1, ])
      specsM <- specsM[-1,]
      rownames(specsM)<-NULL
      par_list_ind=eval(parse(text=specsM$stanParamList))
      par_list_gr=paste('mu_',par_list_ind,sep='')
      par_list_pr=paste(par_list_ind,'_pr_R',sep='')
      if(grepl('KF.*np',modelNames[[1]][[n]])){
        par_list_ind_extra = kf_np_ind_extra
        par_list_gr_extra = kf_np_group_extra 
      } else if(grepl('KF',modelNames[[1]][[n]])){
         par_list_ind_extra = kf_ind_extra
         par_list_gr_extra = kf_group_extra 
      } else{
        par_list_ind_extra = ''
        par_list_gr_extra = ''
      }
      # Group level draws - mean
      gr_draws_tmp = data.frame(extract(stan_fits[[c]][[g]][[n]],pars=par_list_gr))
      gr_draws_tmp$i = 1:dim(gr_draws_tmp)[1]
      gr_draws_tmp = melt(gr_draws_tmp,id='i')
      gr_draws_tmp$model = submodelName
      gr_draws_tmp$cond = cond_names[c]
      gr_draws_tmp <- subset(gr_draws_tmp, select = -c(i))
      gr_draws = rbind(gr_draws,gr_draws_tmp)
      
      # Group level draws - variance
      sig_draws_tmp = data.frame(extract(stan_fits[[c]][[g]][[n]],pars=paste('sigma[',1:length(par_list_gr),']',sep='')))
      sig_draws_tmp$i = 1:dim(sig_draws_tmp)[1]
      sig_draws_tmp = melt(sig_draws_tmp,id='i')
      sig_draws_tmp$model = submodelName
      sig_draws_tmp$cond = cond_names[c]
      sig_draws_tmp <- subset(sig_draws_tmp, select = -c(i))
      sig_draws_tmp$variable = sub('a.','a[',sig_draws_tmp$variable)
      sig_draws_tmp$variable = sub('\\.',']',sig_draws_tmp$variable)
      for (p in 1:length(par_list_gr)){
        sig_draws_tmp$variable[sig_draws_tmp$variable == paste('sigma[',p,']',sep='')] =sub('mu','sigma',par_list_gr[p])
      }
      sig_draws = rbind(sig_draws,sig_draws_tmp)
      
      if(grepl('KF',modelNames[[1]][[n]],fixed=TRUE)){
        gr_draws_tmp = data.frame(extract(stan_fits[[c]][[g]][[n]],pars=par_list_gr_extra))
        gr_draws_tmp$i = 1:dim(gr_draws_tmp)[1]
        gr_draws_tmp = melt(gr_draws_tmp,id='i')
        gr_draws_tmp$model = submodelName
        gr_draws_tmp$cond = cond_names[c]
        gr_draws_tmp <- subset(gr_draws_tmp, select = -c(i))
        gr_draws = rbind(gr_draws,gr_draws_tmp)
      }
      
      if(grepl('conf',submodelName,fixed=TRUE)){
        extra_tmp = cbind(submodelName,summary(stan_fits[[c]][[1]][[n]],par=c('MissingData','MissingConf','sigma','sigma_a','sigma_b',par_list_pr))$summary[,c('mean','Rhat')],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
      }else{
        extra_tmp = cbind(submodelName,summary(stan_fits[[c]][[1]][[n]],par=c('MissingData','sigma','sigma_a','sigma_b',par_list_pr))$summary[,c('mean','Rhat')],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
      }
      parExtraSum = rbind(parExtraSum,extra_tmp)
      
      
      if (save_plots){
        tracplot <- plot(stan_fits[[c]][[1]][[n]], plotfun = "trace", pars = c('sigma','sigma_a','sigma_b'), inc_warmup = FALSE)
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/traces/",submodelName,"traces_sigma_",cond_names[c],".png",sep=''))
        print(tracplot)
        dev.off()
      }
      
      # Individual level summary
      for (p in 1:length(par_list_ind)){
        indpar_tmp = cbind(submodelName,par_list_ind[p],pptIds[,],summary(stan_fits[[c]][[1]][[n]],pars=par_list_ind[p])$summary[,'mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_ind[p])$summary[,'Rhat'],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
        parIndSum = rbind(parIndSum,indpar_tmp)
        
        grpar_tmp = cbind(submodelName,par_list_gr[p],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr[p])$summary[,'mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr[p])$summary[,'sd'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr[p])$summary[,'se_mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr[p])$summary[,'Rhat'],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
        parGroupSum = rbind(parGroupSum,grpar_tmp)
      }
      if (save_plots){
        tracplot <-plot(stan_fits[[c]][[1]][[n]], plotfun = "trace", pars = par_list_gr, inc_warmup = FALSE)
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/traces/",submodelName,"traces_gr_",cond_names[c],".png",sep=''))
        print(tracplot)
        dev.off()
      }
      
      
      if(grepl('KF',modelNames[[1]][[n]],fixed=TRUE)){
        if (save_plots){
          tracplot <-plot(stan_fits[[c]][[1]][[n]], plotfun = "trace", pars = par_list_gr_extra, inc_warmup = FALSE)
          png(paste(output_dir_res,"/",results_sub_dir,"/plots/traces/",submodelName,"tracesExtra_gr_",cond_names[c],".png",sep=''))
          print(tracplot)
          dev.off()
        }
        
        for (p in 1:length(par_list_ind_extra)){
          indpar_tmp = cbind(submodelName,par_list_ind_extra[p],pptIds[,],summary(stan_fits[[c]][[1]][[n]],pars=par_list_ind_extra[p])$summary[,'mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_ind_extra[p])$summary[,'Rhat'],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
          parIndSum = rbind(parIndSum,indpar_tmp)
          
          grpar_tmp = cbind(submodelName,par_list_gr_extra[p],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr_extra[p])$summary[,'mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr_extra[p])$summary[,'sd'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr_extra[p])$summary[,'se_mean'],summary(stan_fits[[c]][[1]][[n]],pars=par_list_gr_extra[p])$summary[,'Rhat'],substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c])
          parGroupSum = rbind(parGroupSum,grpar_tmp)
        }
      }
    }
    
  }
  colnames(parIndSum)<-c('modelName','parName','PID','value','Rhat','vol','stoch','cond')
  colnames(parGroupSum)<-c('modelName','parName','value','se_mean','sd','Rhat','vol','stoch','cond')
  colnames(parExtraSum)<-c('modelName','value','Rhat','vol','stoch','cond')
  parExtraSum$parName = sub('].*',']',rownames(parExtraSum))
  row.names(parIndSum)=NULL
  row.names(parGroupSum)=NULL
  row.names(parExtraSum)=NULL
  
  parIndSum = setorder(parIndSum,modelName,cond,parName)
  parGroupSum = setorder(parGroupSum,modelName,cond,parName)
  parExtraSum = setorder(parExtraSum,modelName,cond,parName)
  parIndSum$value = as.numeric(parIndSum$value); parGroupSum$value = as.numeric(parGroupSum$value); parGroupSum$sd = as.numeric(parGroupSum$sd)
  parGroupSum$se_mean = as.numeric(parGroupSum$se_mean); parIndSum$Rhat = as.numeric(parIndSum$Rhat); parGroupSum$Rhat = as.numeric(parGroupSum$Rhat)
  
  parIndSum_badRhat = parIndSum[((parIndSum$Rhat>1.1) | (parIndSum$Rhat<0.9) | (parIndSum$Rhat=='Inf')),]
  parGroupSum_badRhat = parGroupSum[((parGroupSum$Rhat>1.1) | (parGroupSum$Rhat<0.9) | (parGroupSum$Rhat=='Inf')),]
  parExtraSum_badRhat = parExtraSum[((parExtraSum$Rhat>1.1) | (parExtraSum$Rhat<0.9) | (parExtraSum$Rhat=='Inf')),]
  
  if (save_to_csv){
    write.csv(parIndSum,file=paste(output_dir_res,"/",results_sub_dir,"/parIndSum_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(parGroupSum,file=paste(output_dir_res,"/",results_sub_dir,"/parGroupSum_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(parIndSum_badRhat ,file=paste(output_dir_res,"/",results_sub_dir,"/parIndSum_badRhat_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(parGroupSum_badRhat,file=paste(output_dir_res,"/",results_sub_dir,"/parGroupSum_badRhat_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(gr_draws,file=paste(output_dir_res,"/",results_sub_dir,"/gr_draws_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(sig_draws,file=paste(output_dir_res,"/",results_sub_dir,"/sig_draws_cond",'.csv',sep=''),row.names=FALSE)
    write.csv(parExtraSum_badRhat,file=paste(output_dir_res,"/",results_sub_dir,"/parExtraSum_badRhat_cond",'.csv',sep=''),row.names=FALSE)
  }
}

if (load_csv){
  parIndSum = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","parIndSum_cond.csv",sep=''))
  parGroupSum = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","parGroupSum_cond.csv",sep=''))
  gr_draws = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","gr_draws_cond.csv",sep=''))
  sig_draws = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","sig_draws_cond.csv",sep=''))
  parGroupComp = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","parGroupComp_cond.csv",sep=''))
  parSigmaComp = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","parSigmaComp_cond.csv",sep=''))
  parIndSum[is.na(parIndSum)]=''
  parGroupSum[is.na(parGroupSum)]=''
  gr_draws[is.na(gr_draws)]=''
  sig_draws[is.na(sig_draws)]=''
}
if (post_analysis){
  parIndSum = parIndSum[parIndSum$model %in% model_list,]
  parGroupSum = parGroupSum[parGroupSum$model %in% model_list,]  
  gr_draws = gr_draws[gr_draws$model %in% model_list,]
  sig_draws = sig_draws[sig_draws$model %in% model_list,] 
}

# Plot group level and individual level parameters----------------
hr = 7.5/1.5
wr = 6.25
fix_height_scale = 1
print('Started plotting parameters')
for (n in 1:length(modelNames[[1]])){
  if (!post_analysis){
    submodelName = sub(substr(modelNames[[1]][[n]],nchar(modelNames[[1]][[n]])-5+1,nchar(modelNames[[1]][[n]])),'',modelNames[[1]][[n]])  
  }else{
    submodelName = model_list[[n]]
  }
  specsM=transpose(read.table(file=paste(specs_dir,submodelName,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
  colnames(specsM) <- as.character(specsM[1, ])
  specsM <- specsM[-1,]
  rownames(specsM)<-NULL
  par_list_ind=eval(parse(text=specsM$stanParamList))
  par_list_gr=paste('mu_',par_list_ind,sep='')
  if(grepl('KF.*np',modelNames[[1]][[n]])){
    par_list_ind_extra = kf_np_ind_extra
    par_list_gr_extra = kf_np_group_extra 
  } else if(grepl('KF',modelNames[[1]][[n]])){
    par_list_ind_extra = kf_ind_extra
    par_list_gr_extra = kf_group_extra 
  } else{
    par_list_ind_extra = ''
    par_list_gr_extra = ''
  }
  
  # Plot group level density - mean
  if (!plot_together){
    l_total = length(par_list_gr)
    add_base = 0
    pl = vector(mode = "list", length = l_total)  
    colNo = ceiling(sqrt(l_total))
    colR = floor(length(l_total)/floor(sqrt(length(l_total))))
  }else{
    l_total = length(par_list_gr)+length(par_list_gr_extra)
    add_base = length(par_list_gr)
    pl = vector(mode = "list", length = l_total) 
    colNo = floor(sqrt(l_total))
    colR = floor(l_total/floor(sqrt(l_total)))
    hr = 12
    wr = 6.25
  }
  
  for (p in 1:length(par_list_gr)){
    df = gr_draws[(gr_draws$model==submodelName & gr_draws$variable==par_list_gr[p]),]
    pl[[p]] <- ggplot(data=df,aes(x=value,fill=cond))+geom_density(alpha=0.4,lwd=1.25)+
      ylab('Density')+
      xlab(par_list_gr[p])+
      labs(fill='Condition',subtitle=letters_list[p])+
      theme(plot.subtitle = element_text(size = 20,face='bold',hjust=-0.1))+
      theme(text = element_text(family = "Arial"))+
      scale_color_d3()+
      theme(legend.position = if(p==length(l_total)) 'bottom' else 'none')+
      theme(text=element_text(size=25),
            legend.text = element_text(size = 15),
            legend.title = element_text(size = 15))+
      # scale_fill_discrete(name = "Condition")+
      scale_fill_manual(labels=c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low'), values = c('#1E88E5','#D81B60','#FFC107',"#A9D0C9"))+
      coord_cartesian(xlim = c(0, gr_lims[p]))
    pl[[p]]
    
    if (post_analysis){
      pl[[p]] = pl[[p]] + xlab(TeX(par_gr_latex_names[p]))
        # if (rename_me2){
        #   pl[[p]] = pl[[p]] + scale_fill_discrete(labels=c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low'))
        # }
        
    }
  }
  
  if (!plot_together){
    if (save_plots){
      legend <- get_legend(pl[[length(par_list_gr)]])    
      p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
      if (post_analysis){
        eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off()
        ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDens",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
        # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/FigS8.eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
        # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDens",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,height=376*hr*colR*0.9*fix_height_scale,units="px")
      }else{
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDens",".png",sep=''),res=300,width=614*wr,height=376*hr*colR)
        grid.arrange(grobs=p_no_legend,top=submodelName,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off()
      }
    }
  }
  
  # Plot group and individual level violin/density for extra KF parameters
  if(grepl('KF',modelNames[[1]][[n]],fixed=TRUE)){
    if (!plot_together){
      l_total = length(par_list_gr_extra)
      pl = vector(mode = "list", length = length(par_list_gr_extra))
      colNo = ceiling(sqrt(length(par_list_gr_extra)))
      colR = floor(length(par_list_gr_extra)/floor(sqrt(length(par_list_gr_extra))))  
    }
    for (p in 1:length(par_list_gr_extra)){
      df = gr_draws[(gr_draws$model==submodelName & gr_draws$variable==par_list_gr_extra[p]),]
      pl[[p+add_base]] <- ggplot(geom="raster",data=df,aes(x=value,fill=cond))+geom_density(alpha=0.4,lwd=1.25)+
        # labs(subtitle=par_list_gr_extra[p])+
        ylab('Density')+
        xlab(par_list_gr_extra[p])+
        scale_color_d3()+
        labs(fill='Condition',subtitle=letters_list[p+add_base])+
        theme(plot.subtitle = element_text(size = 20,face='bold',hjust=-0.1))+
        theme(text = element_text(family = "Arial"))+
        theme(text=element_text(size=25),
              legend.text = element_text(size = 15),
              legend.title = element_text(size = 15))+
        # scale_fill_discrete()
        scale_fill_manual(values = c('#1E88E5','#D81B60','#FFC107',"#A9D0C9"))+
        # scale_fill_discrete(name = "Condition")
      pl[[p+add_base]]
      
      if (post_analysis){
        pl[[p+add_base]] = pl[[p+add_base]] + xlab(TeX(par_gr_latex_names_extra[p]))
          
      }
    }
    
    if (save_plots){
      legend <- get_legend(pl[[length(l_total)]])
      # legend <- get_legend(pl[[length(l_total)]])    
      p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
      if (post_analysis){
        eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off()
        # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDensExtra",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,height=376*hr*colR*0.9*fix_height_scale,units="px")
        if (!plot_together){
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDensExtra",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
        }else{
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDensAll",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/Fig_S12.eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
        }
      }else{
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDensExtra",".png",sep=''),res=300,width=614*wr,height=376*hr*colR)
        grid.arrange(grobs=p_no_legend,top=submodelName,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off()  
      }
      
    }
  }

  # Plot group level density - variance
  par_list_sigma = sub('mu','sigma',par_list_gr)
  pl = vector(mode = "list", length = length(par_list_sigma))
  colNo = ceiling(sqrt(length(par_list_sigma)))
  colR = floor(length(par_list_sigma)/floor(sqrt(length(par_list_sigma))))
  for (p in 1:length(par_list_sigma)){
    df = sig_draws[(sig_draws$model==submodelName & sig_draws$variable==par_list_sigma[p]),]
    pl[[p]] <- ggplot(data=df,aes(x=value,fill=cond))+geom_density(alpha=0.4)+
      # labs(subtitle=par_list_sigma[p])+
      ylab('Density')+
      xlab(par_list_sigma[p])+
      scale_color_d3()+
      labs(fill='Condition')+
      theme(legend.position = if(p==length(par_list_sigma)) 'bottom' else 'none')+
      theme(text=element_text(size=20),
            legend.text = element_text(size = 15),
            legend.title = element_text(size = 15))+
      scale_fill_discrete(name = "Condition")
    pl[[p]]
    if (post_analysis){
      pl[[p]] = pl[[p]] + xlab(TeX(par_sig_latex_names[p]))
        
    }
  }
  if (save_plots){
    legend <- get_legend(pl[[length(par_list_sigma)]])    
    p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
    if (post_analysis){
      eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
      dev.off()
      # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_sigmaDens",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,height=376*hr*colR*0.9*fix_height_scale,units="px")
      ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_sigmaDens",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
    }else{
      png(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_sigmaDens",".png",sep=''),res=300,width=614*wr,height=376*hr*colR)
      grid.arrange(grobs=p_no_legend,top=submodelName,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
      dev.off()  
    }
  }
  
}

# Plot ANOVA results for individual level parameters----------------
hr = 7.5/1.5
wr = 5.75
for (n in 1:length(modelNames[[1]])){
  if (!post_analysis){
    submodelName = sub(substr(modelNames[[1]][[n]],nchar(modelNames[[1]][[n]])-5+1,nchar(modelNames[[1]][[n]])),'',modelNames[[1]][[n]])
  }else{
    submodelName = submodelName = model_list[[n]]
  }
  
  specsM=transpose(read.table(file=paste(specs_dir,submodelName,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
  colnames(specsM) <- as.character(specsM[1, ])
  specsM <- specsM[-1,]
  rownames(specsM)<-NULL
  par_list_ind=eval(parse(text=specsM$stanParamList))
  par_list_gr=paste('mu_',par_list_ind,sep='')
  if(grepl('KF.*np',modelNames[[1]][[n]])){
    par_list_ind_extra = kf_np_ind_extra
    par_list_gr_extra = kf_np_group_extra 
  } else if(grepl('KF',modelNames[[1]][[n]])){
    par_list_ind_extra = kf_ind_extra
    par_list_gr_extra = kf_group_extra 
  } else{
    par_list_ind_extra = ''
    par_list_gr_extra = ''
  }
  
  
  if (!plot_together){
    l_total = length(par_list_ind)
    add_base = 0
    pl = vector(mode = "list", length = l_total)  
    colNo = ceiling(sqrt(l_total))
    colR = floor(length(l_total)/floor(sqrt(length(l_total))))
  }else{
    l_total = length(par_list_gr)+length(par_list_ind_extra)
    add_base = length(par_list_gr)
    pl = vector(mode = "list", length = l_total) 
    colNo = floor(sqrt(l_total))
    colR = floor(l_total/floor(sqrt(l_total)))
    hr = 12
    wr = 6.25
  }
  # Individual level
  for (p in 1:length(par_list_ind)){
    parIndSum_anova = parIndSum
    parIndSum_anova$vol <- factor(parIndSum_anova$vol,levels=c('L','H'))
    parIndSum_anova$stoch <- factor(parIndSum_anova$stoch,levels=c('L','H'))
    
    my_data = parIndSum_anova[(parIndSum_anova$modelName==submodelName) & (parIndSum_anova$parName==par_list_ind[p]),]
    
    anova_res <- aov(value ~ vol + stoch + vol:stoch, data = my_data)
    anova_res_summary=summary(anova_res)
    tukey_res = TukeyHSD(anova_res, which = c("stoch","vol"))
    
    vol_pval=as.numeric(unlist(anova_res_summary)['Pr(>F)1'][1]); stoch_pval=as.numeric(unlist(anova_res_summary)['Pr(>F)2'][1]); vs_pval =as.numeric(unlist(anova_res_summary)['Pr(>F)3'][1])
    
    vol_pval_str = if(vol_pval<0.001) {'<0.001*'} else (if (vol_pval<=0.05) format(round(vol_pval,3),nsmall=3) else 'NS')
    stoch_pval_str = if(stoch_pval<0.001) {'<0.001*'} else (if (stoch_pval<=0.05) format(round(stoch_pval,3),nsmall=3) else 'NS')
    vs_pval_str = if(vs_pval<0.001) {'<0.001*'} else (if (vs_pval<=0.05) format(round(vs_pval,3),nsmall=3) else 'NS')
    
    stoch_tuk = data.frame(tukey_res$stoch); stoch_tuk$factor = 'stoch'; stoch_tuk$cond_diff = row.names(stoch_tuk)
    row.names(stoch_tuk) = NULL; stoch_tuk = stoch_tuk[,c(5,6,4,1:3)]; stoch_tuk[,3:6] = format(round(stoch_tuk[,3:6],3),nsmall=3)
    stoch_tuk[stoch_tuk[,3]<0.001,3]='<0.001*'
    
    vol_tuk = data.frame(tukey_res$vol); vol_tuk$factor = 'vol'; vol_tuk$cond_diff = row.names(vol_tuk)
    row.names(vol_tuk) = NULL; vol_tuk = vol_tuk[,c(5,6,4,1:3)]; vol_tuk[,3:6] = format(round(vol_tuk[,3:6],3),nsmall=3)
    vol_tuk[vol_tuk[,3]<0.001,3]='<0.001*'
    
    stat_text=paste("AV -  s:",stoch_pval_str,
                    "; v:",vol_pval_str,
                    "; sv:",vs_pval_str,
                    # '. Tukey: \n',
                    # toString(colnames(vol_tuk)),
                    # '\n',
                    # toString(stoch_tuk),
                    # '\n',
                    # sub('vol', 'vol    ',toString(vol_tuk)),
                    sep='')
    
    palsch = if (vol_pval<=0.05 | stoch_pval<=0.05 | vs_pval<=0.05) 'jco' else 'igv'
    
    levels(my_data$vol)<-c("Low","High")
    levels(my_data$stoch)<-c("Low","High")
    my_data$vol <- factor(my_data$vol,levels=c("High","Low"))
    my_data$stoch <- factor(my_data$stoch,levels=c("High","Low"))
    dodge <- position_dodge(width = 0.9)
    
    if (!post_analysis){
      pl[[p]] = ggplot(data=my_data)+
        geom_violin(aes(x = stoch, y = value, fill= vol),position = dodge)+
        geom_boxplot(aes(x = stoch, y = value, fill= vol),width=0.2, alpha=0.2,position = dodge) +
        geom_jitter(aes(x = stoch, y = value, fill=vol),col='black',shape=19,size=1,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
        ylab(par_list_ind[p])+
        xlab('Stochasticity')+
        labs(fill='Volatility')+
        labs(subtitle=stat_text)+ theme(plot.title = element_text(size = 12)) + theme(plot.subtitle = element_text(size = 10))+
        theme(legend.position = if(p==length(l_total)) 'bottom' else 'none')+
        theme(axis.title.y = element_text(angle=0, vjust = 0.5,hjust=2))+
        theme(text=element_text(size=20),
              legend.text = element_text(size = 15),
              legend.title = element_text(size = 15))
        pl[[p]] 
        
    }else{
      pl[[p]] = ggplot(data=my_data)+
        geom_violin(aes(x = stoch, y = value, fill= vol),position = dodge,lwd=1.25)+
        geom_boxplot(aes(x = stoch, y = value, fill= vol),width=0.2, alpha=0.2,position = dodge,lwd=0.75) +
        geom_jitter(aes(x = stoch, y = value, fill=vol),col='black',shape=19,size=1.5,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
        ylab(TeX(par_ind_latex_names[p]))+
        xlab(stoch_ac_name)+
        labs(fill=vol_inst_name)+
        scale_color_d3()+
        scale_fill_d3()+
        labs(subtitle=letters_list[p])+
        theme(plot.subtitle = element_text(size = 20,face='bold',hjust=-0.1))+
        theme(text = element_text(family = "Arial"))+
        # labs(title=stat_text)+ theme(plot.title = element_text(size = 12))+
        theme(axis.title.y = element_text(angle=0, vjust = 0.5))+
        guides(fill = guide_legend(nrow = 1,title.position = "left"))+
        theme(axis.title = element_text(size = 25),
              text = element_text(size = 25),
              legend.text = element_text(size = 15),
              legend.title = element_text(size = 15))
            
        pl[[p]] 
    }
    
    
  }
  if (!plot_together){
    if (save_plots){
      if (post_analysis){
        legend <- get_legend(pl[[length(par_list_ind)]])    
        p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
        eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off()
        ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaInd",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr*1.5,units="px")
        # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/FigS9",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr*1.5,units="px")
        # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaInd",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
      }else{
        legend <- get_legend(pl[[length(par_list_ind)]])    
        p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaInd",".png",sep=''),res=300,width=614*wr,height=376*hr*colR)
        grid.arrange(grobs=p_no_legend,top=submodelName,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off() 
      }
    }
  }
  
  
  # Individual level extra parameters for KF model
  if (!plot_together){
    l_total = length(par_list_ind_extra)
    colNo = ceiling(sqrt(length(par_list_ind_extra)))
    colR = floor(length(par_list_ind_extra)/floor(sqrt(length(par_list_ind_extra))))
    pl = vector(mode = "list", length = length(par_list_ind_extra))
  }
  if(grepl('KF',modelNames[[1]][[n]],fixed=TRUE)){
    
    for (p in 1:length(par_list_ind_extra)){
      parIndSum_anova = parIndSum
      parIndSum_anova$vol <- factor(parIndSum_anova$vol,levels=c('L','H'))
      parIndSum_anova$stoch <- factor(parIndSum_anova$stoch,levels=c('L','H'))
      
      my_data = parIndSum_anova[(parIndSum_anova$modelName==submodelName) & (parIndSum_anova$parName==par_list_ind_extra[p]),]
      
      anova_res <- aov(value ~ vol + stoch + vol:stoch, data = my_data)
      anova_res_summary=summary(anova_res)
      tukey_res = TukeyHSD(anova_res, which = c("stoch","vol"))
      
      vol_pval=as.numeric(unlist(anova_res_summary)['Pr(>F)1'][1]); stoch_pval=as.numeric(unlist(anova_res_summary)['Pr(>F)2'][1]); vs_pval =as.numeric(unlist(anova_res_summary)['Pr(>F)3'][1])
      
      vol_pval_str = if(vol_pval<0.001) {'<0.001*'} else (if (vol_pval<=0.05) format(round(vol_pval,3),nsmall=3) else 'NS')
      stoch_pval_str = if(stoch_pval<0.001) {'<0.001*'} else (if (stoch_pval<=0.05) format(round(stoch_pval,3),nsmall=3) else 'NS')
      vs_pval_str = if(vs_pval<0.001) {'<0.001*'} else (if (vs_pval<=0.05) format(round(vs_pval,3),nsmall=3) else 'NS')
      
      stoch_tuk = data.frame(tukey_res$stoch); stoch_tuk$factor = 'stoch'; stoch_tuk$cond_diff = row.names(stoch_tuk)
      row.names(stoch_tuk) = NULL; stoch_tuk = stoch_tuk[,c(5,6,4,1:3)]; stoch_tuk[,3:6] = format(round(stoch_tuk[,3:6],3),nsmall=3)
      stoch_tuk[stoch_tuk[,3]<0.001,3]='<0.001*'
      
      vol_tuk = data.frame(tukey_res$vol); vol_tuk$factor = 'vol'; vol_tuk$cond_diff = row.names(vol_tuk)
      row.names(vol_tuk) = NULL; vol_tuk = vol_tuk[,c(5,6,4,1:3)]; vol_tuk[,3:6] = format(round(vol_tuk[,3:6],3),nsmall=3)
      vol_tuk[vol_tuk[,3]<0.001,3]='<0.001*'
      
      stat_text=paste("AV -  s:",stoch_pval_str,
                      "; v:",vol_pval_str,
                      "; sv:",vs_pval_str,
                      # '. Tukey: \n',
                      # toString(colnames(vol_tuk)),
                      # '\n',
                      # toString(stoch_tuk),
                      # '\n',
                      # sub('vol', 'vol    ',toString(vol_tuk)),
                      sep='')
      
      palsch = if (vol_pval<0.05 | stoch_pval<=0.05 | vs_pval<=0.05) 'jco' else 'igv'
      levels(my_data$vol)<-c("Low","High")
      levels(my_data$stoch)<-c("Low","High")
      my_data$vol <- factor(my_data$vol,levels=c("High","Low"))
      my_data$stoch <- factor(my_data$stoch,levels=c("High","Low"))
      dodge <- position_dodge(width = 0.9)
      
      if (!post_analysis){
        pl[[p+add_base]] = ggplot(data=my_data)+
          geom_violin(aes(x = stoch, y = value, fill= vol),position = dodge)+
          geom_boxplot(aes(x = stoch, y = value, fill= vol),width=0.2, alpha=0.2,position = dodge) +
          geom_jitter(aes(x = stoch, y = value, fill=vol),col='black',shape=19,size=1,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
          ylab(par_list_ind_extra[p])+
          xlab('Stochasticity')+
          labs(fill='Volatility')+
          theme(legend.position = if(p==length(l_total)) 'bottom' else 'none')+
          labs(subtitle=stat_text)+ theme(plot.title = element_text(size = 12)) + theme(plot.subtitle = element_text(size = 10))+
          theme(axis.title.y = element_text(angle=0, vjust = 0.5,hjust=2))+
          theme(text=element_text(size=20),
                legend.text = element_text(size = 15),
                legend.title = element_text(size = 15))
      }else{
        pl[[p+add_base]] = ggplot(data=my_data)+
          geom_violin(aes(x = stoch, y = value, fill= vol),position = dodge,lwd=1.25)+
          geom_boxplot(aes(x = stoch, y = value, fill= vol),width=0.2, alpha=0.2,position = dodge,lwd=0.75) +
          geom_jitter(aes(x = stoch, y = value, fill=vol),col='black',shape=19,size=1.5,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
          ylab(TeX(par_ind_latex_names_extra[p]))+
          xlab(stoch_ac_name)+
          labs(fill=vol_inst_name)+
          labs(subtitle=letters_list[p+add_base])+
          scale_color_d3()+
          scale_fill_d3()+
          theme(plot.subtitle = element_text(size = 20,face='bold',hjust=-0.1))+
          theme(text = element_text(family = "Arial"))+
          # labs(title=stat_text)+ theme(plot.title = element_text(size = 12))+
          # labs(title=stat_text)+ theme(plot.title = element_text(size = 12))+
          theme(axis.title.y = element_text(angle=0,vjust=0.5))+
          guides(fill = guide_legend(nrow = 1,title.position = "left"))+
          theme(axis.title = element_text(size = 25,),
                text = element_text(size = 25),
                legend.text = element_text(size = 15),
                legend.title = element_text(size = 15))
        pl[[p]] 
      }
    }
    if (save_plots){
      if (post_analysis){
        legend <- get_legend(pl[[length(par_list_ind_extra)]])    
        p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
        eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=colNo,bottom=legend,guide_legend(nrow=1, byrow=TRUE))
        dev.off()
        if (!plot_together){
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaIndExtra",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")  
        }else{
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaIndAll",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")  
          ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/Fig_S9.eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")  
        }
        
      }else{
        legend <- get_legend(pl[[length(par_list_ind_extra)]])    
        p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
        png(paste(output_dir_res,"/",results_sub_dir,"/plots/ind/",submodelName,"_anovaIndExtra",".png",sep=''),res=300,width=614*wr,height=376*hr*colR)
        grid.arrange(grobs=p_no_legend,top=submodelName,ncol=colNo,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
        dev.off() 
      }
    }
  }
}

# Accuracy of model predictions -------------
if (!load_csv){
  print('Started getting accuracies and tbt values')
  PainValsData = data.frame()
  PainValsModel = data.frame()
  PainValsModel_95 = data.frame()
  NormValsData = data.frame()
  PainValsData_reordered = data.frame()
  PainValsModel_reordered = data.frame()
  NormValsData_reordered = data.frame()
  rmseScore = data.frame()
  rmseScore_norm = data.frame()
  pptRmseScore = data.frame()
  indpptRmseScore = data.frame()
  ci_names = c('ci_2_5','ci_97_5')
  for (c in 1:length(cond_names)){
    model_data=readRDS(paste('model_fit_analysis/data/data_for_stan_lin_',cond_names[c],'.rds',sep=''))
    NT = model_data$Tn; NS = model_data$N
    
    # Get behavioural responses
    PainVals = data.frame(model_data$PainValsAll)
    TrialTypeInd = model_data$TrialTypeAll
    colnames(PainVals) = paste('',1:NT,sep='')
    PainVals = cbind(1:NS,pptIds[,1],PainVals)
    
    colnames(PainVals)[1] = 'sub'
    colnames(PainVals)[2] = 'PID'
    
    PainVals_melt=melt(PainVals,id=c("sub","PID"))
    PainVals_melt$type = 0
    
    for (i in 1:dim(PainVals_melt)[1]){
      if (PainVals_melt$value[i]<0){
        PainVals_melt$value[i] = NaN
      }
      
      PainVals_melt$type[i]=TrialTypeInd[PainVals_melt$sub[i],as.integer(PainVals_melt$variable[i])]
    }
    PainVals_melt$source = 'data' 
    PainVals_melt$source = factor(PainVals_melt$source)
    PainVals_melt$vol = substr(cond_names[c],1,1)
    PainVals_melt$stoch = substr(cond_names[c],3,3)
    PainVals_melt$cond = cond_names[c]
    PainValsData = rbind(PainValsData,PainVals_melt)
    
    # Get 'normative' stimulus inputs
    NormVals = data.frame(model_data$TransfIntesSeq)
    
    colnames(NormVals) = paste('',1:NT,sep='')
    NormVals = cbind(1:NS,pptIds[,1],NormVals)
    
    colnames(NormVals)[1] = 'sub'
    colnames(NormVals)[2] = 'PID'
    
    NormVals_melt=melt(NormVals,id=c("sub","PID"))
    NormVals_melt$type = 0
    
    for (i in 1:dim(NormVals_melt)[1]){
      if (NormVals_melt$value[i]<0){
        NormVals_melt$value[i] = NaN
      }
      
      NormVals_melt$type[i]=TrialTypeInd[NormVals_melt$sub[i],as.integer(NormVals_melt$variable[i])]
    }
    NormVals_melt$type[NormVals_melt$type == '2']=0
    
    NormVals_melt$source = 'data'
    NormVals_melt$source = factor(NormVals_melt$source)
    NormVals_melt$vol = substr(cond_names[c],1,1)
    NormVals_melt$stoch = substr(cond_names[c],3,3)
    NormVals_melt$cond = cond_names[c]
    NormVals_melt$variable = as.numeric(NormVals_melt$variable)
    
    NormValsData = rbind(NormValsData,NormVals_melt)
    
    # Calculate group ppt accuracy - reponse vs normative
    NormVals_melt_reordered = NormVals_melt[mixedorder(paste(NormVals_melt$cond,NormVals_melt$sub,NormVals_melt$variable)),]
    rownames(NormVals_melt_reordered)=NULL
    PainVals_melt_reordered = PainVals_melt[mixedorder(paste(PainVals_melt$cond,PainVals_melt$sub,PainVals_melt$variable)),]
    rownames(PainVals_melt_reordered)=NULL
    
    PainValsData_reordered = rbind(PainValsData_reordered,PainVals_melt_reordered)
    NormValsData_reordered = rbind(NormValsData_reordered,NormVals_melt_reordered)
    
    ppt_rmse_rate_dif = (PainVals_melt_reordered[PainVals_melt_reordered$type=='1' & PainVals_melt_reordered$cond == cond_names[c],]$value - NormVals_melt_reordered[NormVals_melt_reordered$type==1 & NormVals_melt_reordered$cond == cond_names[c],]$value)^2
    ppt_rmse_rate = round(sqrt(sum(ppt_rmse_rate_dif)/length(ppt_rmse_rate_dif)),4)
    
    prd_id = PainVals_melt_reordered$type=='2' & PainVals_melt_reordered$cond == cond_names[c] & !is.na(PainVals_melt_reordered$value)
    prd_id_which = which(PainVals_melt_reordered$type=='2' & PainVals_melt_reordered$cond == cond_names[c] & !is.na(PainVals_melt_reordered$value))+1
  
    ppt_rmse_pred_dif = (PainVals_melt_reordered[prd_id,]$value-NormVals_melt_reordered[prd_id_which,]$value)^2
    ppt_rmse_pred = round(sqrt(sum(ppt_rmse_pred_dif,na.rm=TRUE)/length(ppt_rmse_pred_dif)),4)
    
    ppt_rmse_dif = rbind(as.matrix(ppt_rmse_rate_dif,dim=c(length(ppt_rmse_rate_dif),1)),as.matrix(ppt_rmse_pred_dif,dim=c(length(ppt_rmse_pred_dif),1)))
    ppt_rmse = round(sqrt(sum(ppt_rmse_dif,na.rm=TRUE)/length(ppt_rmse_dif)),4)
    
    pptRmseScore = rbind(pptRmseScore,cbind(ppt_rmse,ppt_rmse_rate,ppt_rmse_pred,substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c]))
    
    # Calculate individual ppt accuracy - reponse vs normative
    for (s in unique(NormVals_melt_reordered$sub)){
      indppt_rmse_rate_dif = (PainVals_melt_reordered[PainVals_melt_reordered$sub==s & PainVals_melt_reordered$type=='1' & PainVals_melt_reordered$cond == cond_names[c],]$value - NormVals_melt_reordered[NormVals_melt_reordered$sub==s & NormVals_melt_reordered$type==1 & NormVals_melt_reordered$cond == cond_names[c],]$value)^2
      indppt_rmse_rate = round(sqrt(sum(indppt_rmse_rate_dif)/length(indppt_rmse_rate_dif)),4)
      
      indprd_id = PainVals_melt_reordered$sub==s & PainVals_melt_reordered$type=='2' & PainVals_melt_reordered$cond == cond_names[c] & !is.na(PainVals_melt_reordered$value)
      indprd_id_which = which(PainVals_melt_reordered$sub==s & PainVals_melt_reordered$type=='2' & PainVals_melt_reordered$cond == cond_names[c] & !is.na(PainVals_melt_reordered$value))+1
      
      indppt_rmse_pred_dif = (PainVals_melt_reordered[indprd_id,]$value-NormVals_melt_reordered[indprd_id_which,]$value)^2
      indppt_rmse_pred = round(sqrt(sum(indppt_rmse_pred_dif,na.rm=TRUE)/length(indppt_rmse_pred_dif)),4)
      
      indppt_rmse_dif = rbind(as.matrix(indppt_rmse_rate_dif,dim=c(length(indppt_rmse_rate_dif),1)),as.matrix(indppt_rmse_pred_dif,dim=c(length(indppt_rmse_pred_dif),1)))
      indppt_rmse = round(sqrt(sum(indppt_rmse_dif,na.rm=TRUE)/length(indppt_rmse_dif)),4)
      
      indpptRmseScore = rbind(indpptRmseScore,cbind(s,pptIds[s,],indppt_rmse,indppt_rmse_rate,indppt_rmse_pred,substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c]))
    }
    
    #Model accuracy and predictions
    for (n in 1:length(modelNames[[1]])){
      submodelName = sub(substr(modelNames[[1]][[n]],nchar(modelNames[[1]][[n]])-5+1,nchar(modelNames[[1]][[n]])),'',modelNames[[1]][[n]])
      specsM=transpose(read.table(file=paste(specs_dir,submodelName,'.txt',sep=''),header = FALSE, sep=';',quote = ""))
      colnames(specsM) <- as.character(specsM[1, ])
      specsM <- specsM[-1,]
      rownames(specsM)<-NULL
      par_list_ind=eval(parse(text=specsM$stanParamList))
      par_list_gr=paste('mu_',par_list_ind,sep='')
      if(grepl('KF.*np',modelNames[[1]][[n]])){
        par_list_ind_extra = kf_np_ind_extra
        par_list_gr_extra = kf_np_group_extra 
      } else if(grepl('KF',modelNames[[1]][[n]])){
        par_list_ind_extra = kf_ind_extra
        par_list_gr_extra = kf_group_extra 
      } else{
        par_list_ind_extra = ''
        par_list_gr_extra = ''
      }
      
      PainValsAll_pred = data.frame(summary(stan_fits[[c]][[g]][[n]],pars=c('PainValsAll_pred'))$summary[,'mean'])
      PainValPreds = data.frame(matrix(ncol=NT,nrow=NS))
      
      for (s in 1:NS){
        id_from = 1+NT*(s-1)
        id_to = NT*s
        PainValPreds[s,] = PainValsAll_pred[id_from:id_to,]
      }
      colnames(PainValPreds) = paste('',1:NT,sep='')
      PainValPreds = cbind(1:NS,pptIds[,1],PainValPreds)
      colnames(PainValPreds)[1] = 'sub'
      colnames(PainValPreds)[2] = 'PID'
      
      PainValPreds_melt=melt(PainValPreds,id=c("sub","PID"))
      PainValPreds_melt$type = 0
      
      for (i in 1:dim(PainVals_melt)[1]){
        PainValPreds_melt$type[i]=TrialTypeInd[PainVals_melt$sub[i],as.integer(PainVals_melt$variable[i])]
      }
      
      PainValPreds_melt$source = 'model'
      PainValPreds_melt$type = factor(PainValPreds_melt$type)
      PainValPreds_melt$modelName = submodelName
      PainValPreds_melt$vol = substr(cond_names[c],1,1)
      PainValPreds_melt$stoch = substr(cond_names[c],3,3)
      PainValPreds_melt$cond = cond_names[c]
      
      # Calculate model accuracy - reponse vs model prediction
      rate_ind = PainVals_melt$type=='1'
      pred_ind = PainVals_melt$type=='2'
      sqe_dif = (PainVals_melt$value-PainValPreds_melt$value)^2
      sqe_rate_dif = (PainVals_melt[rate_ind,]$value-PainValPreds_melt[rate_ind,]$value)^2
      sqe_pred_dif = (PainVals_melt[pred_ind,]$value-PainValPreds_melt[pred_ind,]$value)^2
  
      sqe = round(sqrt(sum((sqe_dif)/(dim(PainVals_melt)[1]-sum(is.na(sqe_dif))),na.rm = TRUE)),4)
      sqe_rate = round(sqrt(sum((sqe_rate_dif)/(sum(rate_ind)-sum(is.na(sqe_rate_dif))),na.rm = TRUE)),4)
      sqe_pred = round(sqrt(sum((sqe_pred_dif)/(sum(pred_ind)-sum(is.na(sqe_pred_dif))),na.rm = TRUE)),4)
      
      rmseScore = rbind(rmseScore,cbind(submodelName,sqe,sqe_rate,sqe_pred,substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c]))
      
      # Calculate model normative accuracy - normative vs model prediction
      PainValPreds_melt_reorderd = PainValPreds_melt[mixedorder(paste(PainValPreds_melt$cond,PainValPreds_melt$sub,PainValPreds_melt$variable)),]
      rownames(PainValPreds_melt_reorderd)=NULL
      PainValsModel_reordered = rbind(PainValsModel_reordered,PainValPreds_melt_reorderd)
      
      norm_rmse_rate_dif = (PainValPreds_melt_reorderd[PainValPreds_melt_reorderd$type=='1' & PainValPreds_melt_reorderd$cond == cond_names[c],]$value - NormVals_melt_reordered[NormVals_melt_reordered$type==1 & NormVals_melt_reordered$cond == cond_names[c],]$value)^2
      norm_rmse_rate = round(sqrt(sum(norm_rmse_rate_dif)/length(norm_rmse_rate_dif)),4)
      
      prd_id_norm = PainValPreds_melt_reorderd$type=='2' & PainValPreds_melt_reorderd$cond == cond_names[c] & !is.na(PainValPreds_melt_reorderd$value)
      prd_id_norm_which = which(PainValPreds_melt_reorderd$type=='2' & PainValPreds_melt_reorderd$cond == cond_names[c] & !is.na(PainValPreds_melt_reorderd$value))+1
      
      norm_rmse_pred_dif = (PainValPreds_melt_reorderd[prd_id_norm,]$value-NormVals_melt_reordered[prd_id_norm_which,]$value)^2
      norm_rmse_pred = round(sqrt(sum(norm_rmse_pred_dif,na.rm=TRUE)/length(norm_rmse_pred_dif)),4)
      
      norm_rmse_dif = rbind(as.matrix(norm_rmse_rate_dif,dim=c(length(norm_rmse_rate_dif),1)),as.matrix(norm_rmse_pred_dif,dim=c(length(norm_rmse_pred_dif),1)))
      norm_rmse = round(sqrt(sum(norm_rmse_dif,na.rm=TRUE)/length(norm_rmse_dif)),4)
      
      rmseScore_norm = rbind(rmseScore_norm,cbind(submodelName,norm_rmse,norm_rmse_rate,norm_rmse_pred,substr(cond_names[c],1,1),substr(cond_names[c],3,3),cond_names[c]))
      
    
      PainValsModel = rbind(PainValsModel,PainValPreds_melt)
      
      
      PainValsAll_pred_95 = data.frame(summary(stan_fits[[c]][[g]][[n]],pars=c('PainValsAll_pred'))$summary[,c('2.5%','97.5%')])
      PainValPreds_95 = list(data.frame(matrix(ncol=NT,nrow=NS)),data.frame(matrix(ncol=NT,nrow=NS)))
      PainValPreds_95_melt = vector(mode = "list", length = 2)
      for (j in 1:2){
        for (s in 1:NS){
          id_from = 1+NT*(s-1)
          id_to = NT*s
          PainValPreds_95[[j]][s,] = PainValsAll_pred_95[id_from:id_to,j]
        }
        colnames(PainValPreds_95[[j]]) = paste('',1:NT,sep='')
        PainValPreds_95[[j]] = cbind(1:NS,pptIds[,1],PainValPreds_95[[j]])
        colnames(PainValPreds_95[[j]])[1] = 'sub'
        colnames(PainValPreds_95[[j]])[2] = 'PID'
        
        PainValPreds_95_melt[[j]]=melt(PainValPreds_95[[j]],id=c("sub","PID"))
        PainValPreds_95_melt[[j]]$type = 0
        
        for (i in 1:dim(PainVals_melt)[1]){
          PainValPreds_95_melt[[j]]$type[i]=TrialTypeInd[PainVals_melt$sub[i],as.integer(PainVals_melt$variable[i])]
        }
        
        PainValPreds_95_melt[[j]]$source = 'model'
        PainValPreds_95_melt[[j]]$type = factor(PainValPreds_95_melt[[j]]$type)
        PainValPreds_95_melt[[j]]$modelName = submodelName
        PainValPreds_95_melt[[j]]$vol = substr(cond_names[c],1,1)
        PainValPreds_95_melt[[j]]$stoch = substr(cond_names[c],3,3)
        PainValPreds_95_melt[[j]]$cond = cond_names[c]
        colnames(PainValPreds_95_melt[[j]])[4] = ci_names[j]
      }
      
      PainValPreds_95_melt[[1]]$ci_97_5 = PainValPreds_95_melt[[2]]$ci_97_5 
      PainValPreds_95_melt = PainValPreds_95_melt[[1]]
      PainValPreds_95_melt = PainValPreds_95_melt[,c(1:4,dim(PainValPreds_95_melt)[2],(5:(dim(PainValPreds_95_melt)[2]-1)))]
      
      PainValsModel_95 = rbind(PainValsModel_95,PainValPreds_95_melt)
    }
    
  }

  colnames(rmseScore) = c('modelName','RMSE','RMSE_rate','RMSE_pred','vol','stoch','cond')
  rmseScore = rmseScore[mixedorder(paste(rmseScore$cond,rmseScore$RMSE)),]
  rownames(rmseScore) = NULL
  
  colnames(pptRmseScore) = c('ppt_rmse','ppt_rmse_rate','ppt_rmse_pred','vol','stoch','cond')
  pptRmseScore = pptRmseScore[mixedorder(paste(pptRmseScore$ppt_rmse)),]
  rownames(pptRmseScore) = NULL
  
  colnames(indpptRmseScore) = c('sub','PID','indppt_rmse','indppt_rmse_rate','indppt_rmse_pred','vol','stoch','cond')
  indpptRmseScore = indpptRmseScore[mixedorder(paste(indpptRmseScore$cond,indpptRmseScore$indppt_rmse)),]
  
  colnames(rmseScore_norm) = c('modelName','RMSE_norm','RMSE_norm_rate','RMSE_norm_pred','vol','stoch','cond')
  rmseScore_norm = rmseScore_norm[mixedorder(paste(rmseScore_norm$cond,rmseScore_norm$RMSE_norm)),]
  rownames(rmseScore_norm) = NULL
  
  PainValsData$variable = as.numeric(PainValsData$variable)
  PainValsModel$variable = as.numeric(PainValsModel$variable)
  PainValsModel_95$variable = as.numeric(PainValsModel_95$variable)
  NormValsData$variable = as.numeric(NormValsData$variable)
  
  PainValsData_reordered$variable = as.numeric(PainValsData_reordered$variable)
  PainValsModel_reordered$variable = as.numeric(PainValsModel_reordered$variable)
  NormValsData_reordered$variable = as.numeric(NormValsData_reordered$variable)
  
}
if (save_to_csv){
  write.csv(rmseScore,file=paste(output_dir_res,"/",results_sub_dir,"/rmseScore_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(rmseScore_norm,file=paste(output_dir_res,"/",results_sub_dir,"/rmseScore_norm_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(pptRmseScore,file=paste(output_dir_res,"/",results_sub_dir,"/pptRmseScore_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(indpptRmseScore,file=paste(output_dir_res,"/",results_sub_dir,"/indpptRmseScore_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(NormValsData,file=paste(output_dir_res,"/",results_sub_dir,"/NormValsData_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(PainValsData,file=paste(output_dir_res,"/",results_sub_dir,"/PainValsData_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(PainValsModel,file=paste(output_dir_res,"/",results_sub_dir,"/PainValsModel_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(NormValsData_reordered,file=paste(output_dir_res,"/",results_sub_dir,"/NormValsData_reordered_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(PainValsData_reordered,file=paste(output_dir_res,"/",results_sub_dir,"/PainValsData_reordered_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(PainValsModel_reordered,file=paste(output_dir_res,"/",results_sub_dir,"/PainValsModel_reordered_cond",'.csv',sep=''),row.names=FALSE)
  write.csv(PainValsModel_95,file=paste(output_dir_res,"/",results_sub_dir,"/PainValsModel_95_cond",'.csv',sep=''),row.names=FALSE)
}

if (load_csv){
  rmseScore = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","rmseScore_cond.csv",sep=''))
  rmseScore_norm = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","rmseScore_norm_cond.csv",sep=''))
  pptRmseScore = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","pptRmseScore_cond.csv",sep=''))
  indpptRmseScore = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","indpptRmseScore_cond.csv",sep=''))
  NormValsData = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","NormValsData_cond.csv",sep=''))
  PainValsData = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","PainValsData_cond.csv",sep=''))
  PainValsModel = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","PainValsModel_cond.csv",sep=''))
  NormValsData_reordered = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","NormValsData_reordered_cond.csv",sep=''))
  PainValsData_reordered = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","PainValsData_reordered_cond.csv",sep=''))
  PainValsModel_reordered = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","PainValsModel_reordered_cond.csv",sep=''))
  PainValsModel_95 = read.csv(paste(output_dir_res,"/",results_sub_dir,"/","PainValsModel_95_cond.csv",sep=''))
  NS = length(unique(PainValsModel$sub))
}
if (load_csv){
  NS = length(unique(PainValsModel$sub))
  if (post_analysis){
    NS = length(unique(PainValsModel$sub))
    PainValsModel = PainValsModel[PainValsModel$modelName %in% model_list,]
    PainValsModel_95 = PainValsModel_95[PainValsModel_95$modelName %in% model_list,]
  }
    
  PainValsModel[is.na(PainValsModel)]=''
  PainValsModel_95[is.na(PainValsModel_95)]=''
  PainValsData[is.na(PainValsData)]=''
  NormValsData[is.na(NormValsData)]=''
  PainValsData_reordered[is.na(PainValsData_reordered)]=''
  NormValsData_reordered[is.na(NormValsData_reordered)]=''
}


cond_names = cond_names_paper_order
cond_names_alter = cond_names_paper_order_labes
save_plots=TRUE
# Plot Accuracy of model predictions -------------
for (s in 1:NS){
  pl = vector(mode = "list", length = length(cond_names)*2)
  for (c in 1:length(cond_names)){
    # Get rating sequences
    tmp_data = PainValsData[PainValsData$cond==cond_names[c] & PainValsData$sub==s & PainValsData$type==1 & PainValsData$source=='data',]
    tmp_norm = NormValsData[NormValsData$cond==cond_names[c] & NormValsData$sub==s & NormValsData$type==1 & NormValsData$source=='data',]
    tmp_norm$source = 'input'
    tmp_data = rbind(tmp_data,tmp_norm)
    
    if (load_csv){
      tmp_data$value = as.numeric(tmp_data$value)
    }
    
    tmp_model = PainValsModel[PainValsModel$cond==cond_names[c] & PainValsModel$sub==s & PainValsModel$type==1 & PainValsModel$source=='model',]
    tmp_model_ci = PainValsModel_95[PainValsModel_95$cond==cond_names[c] & PainValsModel_95$sub==s & PainValsModel_95$type==1 & PainValsModel_95$source=='model',]
    tmp_model_mean = tmp_model_ci
    tmp_model_mean$mean = tmp_model$value
    tmp_model_mean=tmp_model_mean[,c(1:4,dim(tmp_model_mean)[2],(5:(dim(tmp_model_mean)[2]-1)))]
    
    if (do_plot_seq){
      breaks = c('input','data',unique(tmp_model_mean$modelName))
      labels = breaks
      values = c('black','white',pal_igv()(length(unique(tmp_model_mean$modelName))))
      if (post_analysis){
        pl[[2*c-1]]<- ggplot(data=tmp_model_mean,aes(x=variable,y=mean),show.legend=TRUE,linetype='solid')+
          geom_ribbon(aes(ymin=ci_2_5,ymax=ci_97_5),fill="#5050FFFF",alpha = 0.5,show.legend=FALSE,key_glyph = "rect")
      }else{
        pl[[2*c-1]]<- ggplot(data=tmp_model_mean,aes(x=variable,y=mean,col=modelName),show.legend=TRUE,linetype='solid')+
          geom_ribbon(aes(ymin=ci_2_5,ymax=ci_97_5, col=modelName,fill=modelName),alpha = 0.5,show.legend=FALSE,key_glyph = "rect")
      }
      
      pl[[2*c-1]] = pl[[2*c-1]]+
        geom_line(aes(x=variable,y=mean,col=modelName),size = 1,show.legend = FALSE,linetype='solid')+
        geom_line(data=tmp_data,aes(x=variable,y=value,col=source),size = 1,linetype='dashed')+
        labs(fill = NULL, color = NULL, linetype = 'modelName')+
        theme(legend.position = if(c==4) 'bottom' else 'none')+
        guides(fill=guide_legend(nrow=2, byrow=TRUE))+
        xlab('Trial')+
        ylab('Perception')+
        coord_cartesian(ylim = c(-10, 110))+
        theme(plot.subtitle = element_text(size = 18,face='bold',hjust=-0.075),plot.title = element_text(hjust=0.5))+
        labs(subtitle = letters_list[c])+
        ggtitle(paste('Perception for: ',cond_names_alter[c],sep=''))+
        # ggtitle(paste('Rating for ',cond_names_alter[c],'. Subject ',pptIds[s,],sep=''))+
        scale_y_continuous(breaks=seq(0,100,by=20))+
        scale_x_continuous(breaks=seq(0,80,by=20))+
        theme(text=element_text(size=20),
              legend.text = element_text(size = 15),
              legend.title = element_text(size = 15))
      
      if (post_analysis){
        pl[[2*c-1]] = pl[[2*c-1]] + scale_colour_manual(labels=c('Input','Response',modelNamesTitle[[1]]), breaks=breaks, values = values)
      }else{
        pl[[2*c-1]] = pl[[2*c-1]]+
          scale_colour_manual(labels=labels, breaks=breaks, values = values) +
          scale_fill_manual(labels=labels[3:(length(breaks))],breaks=breaks[3:(length(breaks))], values = values[3:(length(breaks))])
      }
    }
    
    
    # Get prediction sequences
    PainValsData_reordered_tmp = PainValsData_reordered[PainValsData_reordered$cond==cond_names[c] & PainValsData_reordered$sub==s,]
    NormValsData_reordered_tmp = NormValsData_reordered[NormValsData_reordered$cond==cond_names[c] & NormValsData_reordered$sub==s,]
    prd_id = PainValsData_reordered_tmp$type=='2' & !is.na(PainValsData_reordered_tmp$value)
    prd_id_which = which(PainValsData_reordered_tmp$type=='2' & !is.na(PainValsData_reordered_tmp$value))+1
    tmp_data_pred = PainValsData_reordered_tmp[prd_id,]
    tmp_data_pred$variable = tmp_data_pred$variable+1
    tmp_norm_pred = NormValsData_reordered_tmp[prd_id_which,]
    tmp_norm_pred$source='input'
    tmp_data_pred = rbind(tmp_data_pred,tmp_norm_pred)
    if (load_csv){
      tmp_data_pred$value = as.numeric(tmp_data_pred$value)
    }
    
    prd_id_m = PainValsModel$type=='2' & !is.na(PainValsModel$value) & PainValsModel$cond==cond_names[c] & PainValsModel$sub==s
    prd_id_m_which = which(PainValsModel$type=='2' & !is.na(PainValsModel$value) & PainValsModel$cond==cond_names[c] & PainValsModel$sub==s)+1
    tmp_model_pred = PainValsModel[prd_id_m,]
    tmp_model_pred$variable = tmp_model_pred$variable+1
    tmp_model_pred_ci = PainValsModel_95[PainValsModel_95$cond==cond_names[c] & PainValsModel_95$sub==s & PainValsModel_95$type==2 & PainValsModel_95$source=='model',]
    tmp_model_pred_mean = tmp_model_pred_ci
    tmp_model_pred_mean$mean = tmp_model_pred$value
    tmp_model_pred_mean=tmp_model_pred_mean[,c(1:4,dim(tmp_model_pred_mean)[2],(5:(dim(tmp_model_pred_mean)[2]-1)))]
    tmp_model_pred_mean$variable = tmp_model_pred_mean$variable+1
    
    if (do_plot_seq){
      breaks = c('input','data',unique(tmp_model_pred_mean$modelName))
      labels = breaks
      values = c('black','white',pal_igv()(length(unique(tmp_model_mean$modelName))))
      if (post_analysis){
        pl[[2*c]]<- ggplot(data=tmp_model_pred_mean,aes(x=variable,y=mean),show.legend=TRUE,linetype='solid')+
          geom_ribbon(aes(ymin=ci_2_5,ymax=ci_97_5),fill="#5050FFFF",alpha = 0.5,show.legend=FALSE,key_glyph = "rect") 
      }else{
        pl[[2*c]]<- ggplot(data=tmp_model_pred_mean,aes(x=variable,y=mean,col=modelName),show.legend=TRUE,linetype='solid')+
          geom_ribbon(aes(ymin=ci_2_5,ymax=ci_97_5, col=modelName,fill=modelName),alpha = 0.5,show.legend=FALSE,key_glyph = "rect") 
      }
      # count_letter_plot=count_letter_plot+1
      pl[[2*c]]= pl[[2*c]]+ 
        geom_line(aes(x=variable,y=mean,col=modelName),size = 1,show.legend = FALSE,linetype='solid')+
        geom_line(data=tmp_data_pred,aes(x=variable,y=value,col=source),size = 1,linetype='dashed')+
        # labs(col='modelName',fill='modelName',linetype='modelName')+
        labs(fill = NULL, color = NULL, linetype = '')+
        theme(legend.position = if(c==4) 'bottom' else 'none')+
        guides(fill=guide_legend(nrow=2, byrow=TRUE))+
        xlab('Trial')+
        ylab('Prediction')+
        labs(subtitle = letters_list[c+4])+
        coord_cartesian(ylim = c(-10, 110))+
        theme(plot.subtitle = element_text(size = 18,face='bold',hjust=-0.075),plot.title = element_text(hjust=0.5,vjust=-0.4))+
        ggtitle(paste('Prediction for: ',cond_names_alter[c],sep=''))+
        # ggtitle(paste('Prediction for ',cond_names_alter[c],'. Subject ',pptIds[s,],sep=''))+
        scale_y_continuous(breaks=seq(0,100,by=20))+
        scale_x_continuous(breaks=seq(0,80,by=20))+
        theme(text=element_text(size=20),
            legend.text = element_text(size = 15),
            legend.title = element_text(size = 15))
      
      if (post_analysis){
        pl[[2*c]] = pl[[2*c]] + scale_colour_manual(labels=c('Input','Response',modelNamesTitle[[1]]), breaks=breaks, values = values)
      }else{
        pl[[2*c]] = pl[[2*c]]+
          scale_colour_manual(labels=labels, breaks=breaks, values = values) +
          scale_fill_manual(labels=labels[3:(length(breaks))],breaks=breaks[3:(length(breaks))], values = values[3:(length(breaks))])
      }
    }
    
  }
  
  #%%% Figure S4 example participant plot
  if (do_plot_seq){
    legend <- get_legend(pl[[8]])    
    wr=5*1.5;hr=7*1.5
    p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
    if (save_plots){
      # png(paste(output_dir_res,"/",results_sub_dir,"/plots/series/seriesPlot_sub_",pptIds[s,],".png",sep=''),res=300,width=614*wr,height=376*hr)
      seq_plot_tmp = grid.arrange(grobs=p_no_legend,ncol=2,bottom=legend,guide_legend(nrow=2, byrow=TRUE))
      # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/series/seriesPlot_sub_",pptIds[s,],".svg",sep=''),plot=seq_plot_tmp,dpi=300,width=614*wr,height=376*hr,units="px")
      # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/series/seriesPlot_sub_",pptIds[s,],".eps",sep=''),plot=seq_plot_tmp,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
      if (pptIds[s,] == '706'){ #example sequence for the paper
        ggplot2::ggsave(paste(extra_analysis_res,"example_series_FigS4/seriesPlot_example_FigS5.eps",sep=''),plot=seq_plot_tmp,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px") 
        ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/series/seriesPlot_example_FigS4.eps",pptIds[s,],".eps",sep=''),plot=seq_plot_tmp,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
      }
      
      # ggplot2::ggsave(paste(output_dir_res,"/",results_sub_dir,"/plots/group/",submodelName,"_grDens",".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr*0.9,units="px")
      dev.off()  
    }
  }
  
  
}
