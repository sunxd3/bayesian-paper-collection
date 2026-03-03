/////
// Jakub Onysk based on Jepma et al. 2018 and Gregory 2022
// Basic perceptive RL model with group-level scale parameters from a reparametrised half-student_t(3,0,1) distribution
// non-perceptive
///// 
data {
  int<lower=1> N; // number of subjects
  int<lower=1> Tn; // number of trials
  int<lower=1> N_obs_VAS; // number of pain ratings
  int<lower=1> N_obs_CP; // number of predictions
  matrix[N,Tn] noxInputTrans; // sigmoid transformed noxious input
  matrix[N,N_obs_VAS] painRating; // observed pain ratings
  matrix[N,N_obs_CP] painPred; // observed pain predictions
  matrix[N,N_obs_VAS] RatingConf; // observed pain ratings
  matrix[N,N_obs_CP] PredConf; // observed pain predictions
  int TrialType[N,Tn]; //coding whether pain rating or prediction - when missing prediciton on last trial on sequence code as '-1'
  int TrialTypeAllWMissing[N,Tn]; //coding whether pain rating or prediction - the missing prediction is coded as '2'
  matrix[N,Tn] PainValsAll; //ratings and predictions together - missing predictions are as '-1'
  int PercIndexArrayLong[N,Tn]; // Indices of pain ratings - else coded as '-1' - including missing predictions
  int PredIndexArrayLong[N,Tn]; // Indices of preditions - else coded as '-1' - including missing predictions
  int PercIndexArray[N,N_obs_VAS]; // Indices of pain ratings only
  int PredIndexArray[N,N_obs_CP];  // Indices of pain predictions only
  int IndexMissingAll[N,N_obs_VAS-N_obs_CP]; // Indices of missing last of the sequences missing predicitons
  
}
transformed data {
  int<lower=0> N_missing = N_obs_VAS - N_obs_CP; // number of missing predicitons
}

parameters {
  // modelling missing last in the sequenences prediction as missing data - https://mc-stan.org/docs/2_29/stan-users-guide/sliced-missing-data.html
  matrix<lower=0, upper=100> [N,N_missing] MissingData; 
  matrix<lower=0, upper=1> [N,N_missing] MissingConf; 
  
  /// group-level paramters 
  vector[1] mu_pr01;
  vector<lower=0>[3] mu_pr;
  vector<lower=0>[4] sigma_a; // for student-t reparametrisation - https://mc-stan.org/docs/2_29/stan-users-guide/reparameterization.html#reparameterizing-a-student-t-distribution
  vector<lower=0>[4] sigma_b; // for student-t reparametrisation - https://mc-stan.org/docs/2_29/stan-users-guide/reparameterization.html#reparameterizing-a-student-t-distribution
  
  /// subject-level parameters [N(0,1)] before transformation - Matt Trick - https://mc-stan.org/docs/2_29/stan-users-guide/reparameterization.html#reparameterization.section
  vector[N] alpha_pr_R;
  // vector[N] gam_pr_R;
  vector<lower=0>[N] eta_pr_R;
  vector<lower=0>[N] E0_pr_R;
  vector<lower=0>[N] cs_pr_R;
  // vector[N] bet_pr_R;
  // vector<lower=0>[N] C_pr_R;
}

transformed parameters{
  /// subject-level paramters - transformed from N(0,1)
  vector<lower=0, upper=1>[N] alpha; //learning reate
  // vector<lower=0, upper=1>[N] gam; //perceptual weighting parameter Perception/Expectation
  vector<lower=0>[N] eta; //Normal behavioural response noise
  vector<lower=0>[N] E0; //initila expected pain intensity
  vector<lower=0>[N] cs; //initila expected pain intensity
  // vector<lower=0, upper=1>[N] bet;
  // vector<lower=0>[N] C;
  
  /// group-level scale prior transformation for student-t reparametrisation student_t(3,0,1)
  vector<lower=0>[4] sigma = sigma_a ./sqrt(sigma_b); 
  
  // combining observed and missing data - https://mc-stan.org/docs/2_29/stan-users-guide/sliced-missing-data.html
  matrix[N,Tn] slicedData;
  matrix[N,Tn] confData;
  for (s in 1:N){
    slicedData[s,PercIndexArray[s,:]] = painRating[s,:];
    slicedData[s,PredIndexArray[s,:]] = painPred[s,:];
    slicedData[s,IndexMissingAll[s,:]] = MissingData[s,:];
    
    confData[s,PercIndexArray[s,:]] = RatingConf[s,:];
    confData[s,PredIndexArray[s,:]] = PredConf[s,:];
    confData[s,IndexMissingAll[s,:]] = MissingConf[s,:];
  }
  
  //////// Matt Trick for subject level parameters
  alpha = Phi_approx(mu_pr01[1] + sigma[1] * alpha_pr_R)*1;
  // gam = Phi_approx(mu_pr[2] + sigma[2] * gam_pr_R)*1;
  eta = mu_pr[1] + sigma[2] * eta_pr_R;
  E0 = mu_pr[2] + sigma[3] * E0_pr_R;
  cs = mu_pr[3] + sigma[4] * cs_pr_R;
  // bet = Phi_approx(mu_pr01[2] + sigma[4] * bet_pr_R)*1;
  // C = mu_pr[3] + sigma[5] * C_pr_R;
}

model {
  // group-level hyperparameters
  mu_pr01 ~ normal(0,1);
  mu_pr ~ normal(0,1);
  sigma_a ~ normal(0,1); //reparametrisation student_t(3,0,1)
  sigma_b ~ gamma(1.5, 1.5); //reparametrisation student_t(3,0,1)
  
  // individual level standard-normal paramters for Matt-Trick
  alpha_pr_R ~ normal(0,1);
  // gam_pr_R ~ normal(0,1);
  eta_pr_R ~ normal(0,1);
  E0_pr_R ~ normal(0,1);
  cs_pr_R ~ normal(0,1);
  // bet_pr_R ~ normal(0,1);
  // C_pr_R ~ normal(0,1);
  
  for (s in 1:N){
    // real P; // model tracked pain perception
    real E; //  model tracked expected pain
    real PE; // model tracked predicition error
    vector[2] tmp_store; // store model rating and prediction values for current trial
    // real prev_store;
    
    E = E0[s]; // initial expectation
    // prev_store = E0[s];
    
    for (t in 1:Tn){
      PE = noxInputTrans[s,t] - E;
      E = E + alpha[s]*PE;
      
      tmp_store[1] = noxInputTrans[s,t];
      tmp_store[2] = E;
      
      slicedData[s,t] ~ normal(tmp_store[TrialTypeAllWMissing[s,t]],eta[s]*exp((1-confData[s,t])/cs[s]));
      // prev_store = slicedData[s,t];
    }
    
  }
  
}
generated quantities{
  real<lower=0, upper=1> mu_alpha;
  // real<lower=0, upper=1> mu_gam;
  real<lower=0> mu_eta;
  real<lower=0> mu_E0;
  real<lower=0> mu_cs;
  // real<lower=0, upper=1> mu_bet;
  // real<lower=0> mu_C;
  
  real log_lik[N];
  
  // // store values
  // real E_store[N,Tn];
  //// real P_store[N,Tn];
  // real PE_store[N,Tn];
  
   // For posterior predictive check
  real PainValsAll_pred[N, Tn];
  
  // Fill the values of the log-lik and posterior-pred matrices
  for (s in 1:N){
    log_lik[s] = 0;
    for (t in 1:Tn){
      PainValsAll_pred[s,t]=-1;  
    }
  }
  
  // group level parameters
  mu_alpha = Phi_approx(mu_pr01[1]);
  // mu_gam = Phi_approx(mu_pr[2]);
  mu_eta = mu_pr[1];
  mu_E0 = mu_pr[2];
  mu_cs = mu_pr[3];
  // mu_bet = Phi_approx(mu_pr01[2]);
  // mu_C = mu_pr[3];
  
  for (s in 1:N){
    // real P; // model tracked pain perception
    real E; //  model tracked expected pain
    real PE; // model tracked predicition error
    vector[2] tmp_store; // store model rating and prediction values for current trial
    // real prev_store;
    
    E = E0[s]; // initial expectation
    // prev_store = E0[s];
    
    for (t in 1:Tn){
      // // store values
      // E_store[s,t] = E;
      
      // update values
      PE = noxInputTrans[s,t] - E;
      E = E + alpha[s]*PE;
      
      // // store values
      // PE_store[s,t] = PE;
      //// P_store[s,t] = P;
      
      tmp_store[1] = noxInputTrans[s,t];
      tmp_store[2] = E;
      PainValsAll_pred[s,t] = normal_rng(tmp_store[TrialTypeAllWMissing[s,t]],eta[s]*exp((1-confData[s,t])/cs[s]));
      log_lik[s] +=normal_lpdf(slicedData[s,t] | tmp_store[TrialTypeAllWMissing[s,t]],eta[s]*exp((1-confData[s,t])/cs[s]));
      // prev_store = slicedData[s,t];
    }
    
  }
}

