rm(list=ls())

library(rstan) 
library(loo)
library(brms)


## Load data
#______________________

data_vill <- read.csv("Output/incidence_coverage_model_data_village.csv")
data_dist <- read.csv("Output/incidence_coverage_model_data_district.csv")

# Dog population
dogs <- as.matrix(read.csv("Output/dogPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))

# Monthly village-level vaccination
vax_vill <- as.matrix(read.csv("Output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) 


# Prep data for stan
#______________________

# Data for full model
n = exp(colMeans(log(1-vax_vill)))
data <- list(
  N = nrow(data_dist)-2,
  Nv = nrow(vax_vill),
  Y = data_dist$cases[-c(1:2)],
  n = n,
  Sn = (1-vax_vill)/rep(n,each=nrow(vax_vill)),
  W = t(t(dogs)/colSums(dogs)),
  K = 3,
  X = cbind(1,data_dist[-c(1:2),c("log_case_rate_last2monthMean","log_dog_density")]),
  offsets = log(data_dist$dogs[-c(1:2)]),
  eps = 1e-5
)
data_standardised <- data
data_standardised$X[,2:ncol(data_standardised$X)] <- scale(data_standardised$X[,2:ncol(data_standardised$X)])

# Data for model without prior incidence
data_woIncidence <- list(
  N = nrow(data_dist)-2,
  Nv = nrow(vax_vill),
  Y = data_dist$cases[-c(1:2)],
  n = n,
  Sn = (1-vax_vill)/rep(n,each=nrow(vax_vill)),
  W = t(t(dogs)/colSums(dogs)),
  K = 2,
  X = cbind(1,data_dist[-c(1:2),c("log_dog_density")]),
  offsets = log(data_dist$dogs[-c(1:2)]),
  eps = 1e-5
)
data_woIncidence_standardised <- data_woIncidence
data_woIncidence_standardised$X[,2:ncol(data_woIncidence_standardised$X)] <- scale(data_woIncidence_standardised$X[,2:ncol(data_woIncidence_standardised$X)])



# Fit models
#______________________

if(!dir.exists("output/stan_models")){dir.create("output/stan_models")}

# Full model
model_dist_month_stan <- stan("stan/power_mean_model_district.stan", data = data, iter = 3000,
                              warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=11,
                              control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                              pars = c("beta","p","phi","Intercept","log_lik"))
log_lik <- extract_log_lik(model_dist_month_stan, merge_chains = FALSE)
r_eff <- relative_eff(exp(log_lik), cores = 4) 
loo <- loo(log_lik, r_eff = r_eff, cores = 4)
traceplot(model_dist_month_stan,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"))
plot(model_dist_month_stan,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"))
summary(model_dist_month_stan,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_dist_month_stan,"output/stan_models/incidence_from_vax_model_district_stan.rds")

model_dist_month_stan_standardise <- stan("stan/power_mean_model_district_standardise.stan", data = data_standardised, iter = 3000,
                              warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=11,
                              control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                              pars = c("beta","p","phi","Intercept","log_lik"))
log_lik <- extract_log_lik(model_dist_month_stan_standardise, merge_chains = FALSE)
r_eff <- relative_eff(exp(log_lik), cores = 4) 
loo_standardise <- loo(log_lik, r_eff = r_eff, cores = 4)
traceplot(model_dist_month_stan_standardise,inc_warmup=F,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"))
plot(model_dist_month_stan_standardise,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"))
summary(model_dist_month_stan_standardise,pars=c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"),probs = c(0.025,0.5, 0.975))$summary
saveRDS(model_dist_month_stan_standardise,"output/stan_models/incidence_from_vax_model_district_stan_standardise.rds")


# Without prior incidence
model_dist_month_stan_woIncidence <- stan("stan/power_mean_model_district.stan", data = data_woIncidence, iter = 3000,
                                       warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=79,
                                       control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                       pars = c("beta","p","phi","Intercept","log_lik","log_post"))
log_lik <- extract_log_lik(model_dist_month_stan_woIncidence, merge_chains = FALSE)
r_eff <- relative_eff(exp(log_lik), cores = 4) 
loo_woIncidence <- loo(log_lik, r_eff = r_eff, cores = 4)
traceplot(model_dist_month_stan_woIncidence,inc_warmup=T,pars=c("Intercept","beta[2]","beta[3]","p","phi")) # problem with local optima...
plot(model_dist_month_stan_woIncidence,pars=c("Intercept","beta[2]","beta[3]","p","phi"))
saveRDS(model_dist_month_stan_woIncidence,"output/stan_models/incidence_from_vax_model_district_stan_woPriorCases.rds")

model_dist_month_stan_woIncidence_standardise <- stan("stan/power_mean_model_district_standardise.stan", data = data_woIncidence_standardised, iter = 3000,
                                          warmup = 1500, thin = 1, chains = 4, verbose = TRUE, cores=4, seed=79,
                                          control = list(adapt_delta = 0.95, max_treedepth = 10),include = TRUE, 
                                          pars = c("beta","p","phi","Intercept","log_lik","log_post"))
log_lik <- extract_log_lik(model_dist_month_stan_woIncidence_standardise, merge_chains = FALSE)
r_eff <- relative_eff(exp(log_lik), cores = 4) 
loo_woIncidence_standardise <- loo(log_lik, r_eff = r_eff, cores = 4)
traceplot(model_dist_month_stan_woIncidence_standardise,inc_warmup=T,pars=c("Intercept","beta[2]","beta[3]","p","phi")) # problem with local optima...
plot(model_dist_month_stan_woIncidence_standardise,pars=c("Intercept","beta[2]","beta[3]","p","phi"))
saveRDS(model_dist_month_stan_woIncidence_standardise,"output/stan_models/incidence_from_vax_model_district_stan_woPriorCases_standardise.rds")

loo 
loo_woIncidence 
# without incidence gives big reduction in fit
