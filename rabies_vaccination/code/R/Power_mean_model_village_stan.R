rm(list=ls())

library(rstan) 
library(loo)
library(brms)


## Load data
#______________________

data_vill <- read.csv("Output/incidence_coverage_model_data_village.csv")
data_vill$susc_last2monthMean <- 1-data_vill$vax_last2monthMean
data <- readRDS("Output/power_mean_model_village_data.rds")
colnames(data$X)
load("Output/neighbour_notNeighbour_susceptibilities.Rdata")


# Variations of the data for different models
data_standardised <- data
data_standardised$X[,2:ncol(data_standardised$X)] <- scale(data_standardised$X[,2:ncol(data_standardised$X)])
data_wo_distant_cases <- data
data_wo_distant_cases$X <- data$X[,which(!colnames(data$X) %in% c("log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean"))]
data_wo_distant_cases$K <- data_wo_distant_cases$K-2
data_wo_distant_cases_standardised <- data_wo_distant_cases
data_wo_distant_cases_standardised$X[,2:ncol(data_wo_distant_cases_standardised$X)] <- scale(data_wo_distant_cases_standardised$X[,2:ncol(data_wo_distant_cases_standardised$X)])
data_wo_cases <- data
data_wo_cases$X <- data$X[,which(!colnames(data$X) %in% c("log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean"))]
data_wo_cases$K <- data_wo_cases$K-3
data_wo_cases_standardised <- data_wo_cases
data_wo_cases_standardised$X[,2:ncol(data_wo_cases_standardised$X)] <- scale(data_wo_cases_standardised$X[,2:ncol(data_wo_cases_standardised$X)])

vax_vill <- as.matrix(read.csv("Output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) 

source("R/Functions/Lmeans.R")




# Fit models
#______________________

# Full model
model_vill_month_stan <- stan("stan/power_mean_model_village.stan", data = data, iter = 3000,
                              warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                              control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                              pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"))
traceplot(model_vill_month_stan,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"))
summary(model_vill_month_stan,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan,"output/stan_models/incidence_from_vax_model_village_stan.rds")

# Without distant cases
model_vill_month_stan_wo_distant_cases <- stan("stan/power_mean_model_village.stan", data = data_wo_distant_cases, iter = 3000,
                                               warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                                               control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                               pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan_wo_distant_cases,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"))
traceplot(model_vill_month_stan_wo_distant_cases,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"))
summary(model_vill_month_stan_wo_distant_cases,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan_wo_distant_cases,"output/stan_models/incidence_from_vax_model_village_stan_wo_distant_cases.rds")

# Without cases
model_vill_month_stan_wo_cases <- stan("stan/power_mean_model_village.stan", data = data_wo_cases, iter = 3000,
                                       warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                                       control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                       pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan_wo_cases,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"))
traceplot(model_vill_month_stan_wo_cases,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"))
summary(model_vill_month_stan_wo_cases,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan_wo_cases,"output/stan_models/incidence_from_vax_model_village_stan_wo_cases.rds")


# Standardised
#-----------

# Full model
model_vill_month_stan_standardised <- stan("stan/power_mean_model_village_standardised.stan", data = data_standardised, iter = 3000,
                                           warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                                           control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                           pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"))
traceplot(model_vill_month_stan_standardised,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"))
summary(model_vill_month_stan_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[8]","beta[9]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan_standardised,"output/stan_models/incidence_from_vax_model_village_stan_standardised.rds")

# Without distant cases
model_vill_month_stan_wo_distant_cases_standardised <- stan("stan/power_mean_model_village_standardised.stan", data = data_wo_distant_cases_standardised, iter = 3000,
                                                            warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                                                            control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                                            pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan_wo_distant_cases_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"))
traceplot(model_vill_month_stan_wo_distant_cases_standardised,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"))
summary(model_vill_month_stan_wo_distant_cases_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan_wo_distant_cases_standardised,"output/stan_models/incidence_from_vax_model_village_stan_wo_distant_cases_standardised.rds")

# Without cases
model_vill_month_stan_wo_cases_standardised <- stan("stan/power_mean_model_village_standardised.stan", data = data_wo_cases_standardised, iter = 3000,
                                                    warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=5,
                                                    control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                                    pars = c("beta","p","phi","Intercept","sigma_village","gamma_t"))
plot(model_vill_month_stan_wo_cases_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"))
traceplot(model_vill_month_stan_wo_cases_standardised,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"))
summary(model_vill_month_stan_wo_cases_standardised,pars=c("Intercept","beta[2]","beta[3]","beta[4]","beta[5]","beta[6]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_vill_month_stan_wo_cases_standardised,"output/stan_models/incidence_from_vax_model_village_stan_wo_cases_standardised.rds")


# WAIC
#-----------

# Get pointwise log-likelihoods
model <- readRDS("output/stan_models/incidence_from_vax_model_village_stan.rds")
model_wo_distant_cases <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_distant_cases.rds")
model_wo_cases <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_cases.rds")
set.seed(0)
n_samples <-500 # used 6000 for values in Table 1, setting to 500 here to keep files manageably sized
for(model_sim in c("model","model_wo_cases","model_wo_distant_cases")){
  samples_pars <- posterior_samples(get(model_sim), pars = c("Intercept","beta","p","phi"))
  samples <- sample(1:nrow(samples_pars),n_samples)
  samples_pars <- samples_pars[samples,]
  samples_reffs <- posterior_samples(get(model_sim), pars = c("gamma_t"))
  samples_reffs <- samples_reffs[samples,]
  samples_pars <- samples_pars[,-which(colnames(samples_pars)=="beta[1]")]
  log_lik_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
  if(model_sim=="model"){pars_sim <- c("susc_last2monthMean","log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean","log_dog_density","HDR")
  }else if(model_sim=="model_wo_cases"){pars_sim <- c("susc_last2monthMean","log_dog_density","HDR")
  }else if(model_sim=="model_wo_distant_cases"){pars_sim <- c("susc_last2monthMean","log_case_rate_last2monthMean","log_dog_density","HDR")}
  X <- t(as.matrix(cbind(1,data_vill[,pars_sim],
                         "power_mean_neighbours_last2MonthMean"=NA,"power_mean_notNeighbours_last2MonthMean"=NA,"village_re"=NA,log(data_vill$dogs))))
  power_mean_neighbours <- power_mean_notNeighbours <- matrix(NA, nrow=nrow(vax_vill),ncol=ncol(vax_vill))
  for(i in 1:nrow(samples_pars)){
    for(v in 1:nrow(vax_vill)){
      power_mean_neighbours[v,] <- sapply(1:ncol(vax_vill),function(x) powMean(x=S_n[[v]][,x],p=samples_pars[i,"p"],wts = W_n[[v]][,x]))
      power_mean_notNeighbours[v,] <- sapply(1:ncol(vax_vill),function(x) powMean(x=S_nn[[v]][,x],p=samples_pars[i,"p"],wts = W_nn[[v]][,x]))
    }
    X["power_mean_neighbours_last2MonthMean",] <- (c(cbind(NA,power_mean_neighbours[,-ncol(vax_vill)]))+c(cbind(NA,NA,power_mean_neighbours[,-((ncol(vax_vill)-1):ncol(vax_vill))])))/2
    X["power_mean_notNeighbours_last2MonthMean",] <- (c(cbind(NA,power_mean_notNeighbours[,-ncol(vax_vill)]))+c(cbind(NA,NA,power_mean_notNeighbours[,-((ncol(vax_vill)-1):ncol(vax_vill))])))/2
    X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
    mu <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(length(pars_sim)+3))]),1,1)))
    log_lik_mat[,i] <- dnbinom(data_vill$cases,mu=mu,size=samples_pars[i,"phi"],log = T)
  }

  saveRDS(log_lik_mat,
       file=paste0("output/obs_loglik_stan_village_",model_sim,".rds"))
}

log_lik <- readRDS("output/obs_loglik_stan_village_model.rds")
log_lik_wo_distant_cases <- readRDS("output/obs_loglik_stan_village_model_wo_distant_cases.rds")
log_lik_wo_cases <- readRDS("output/obs_loglik_stan_village_model_wo_cases.rds")
waic(t(log_lik[(88*2+1):nrow(log_lik),]))
waic(t(log_lik_wo_distant_cases[(88*2+1):nrow(log_lik_wo_distant_cases),]))
waic(t(log_lik_wo_cases[(88*2+1):nrow(log_lik_wo_cases),]))




