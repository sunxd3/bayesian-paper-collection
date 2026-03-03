
// The input data.
data {
  int<lower=1> N;  // total number of observations
  int Nv; // number of villages
  int Y[N];  // response variable
  matrix[Nv,N+2] Sn;  // normalised susceptibility in each village each month
  vector[N+2] n; // normalisation constants
  matrix[Nv,N+2] W;  // weight for each village each month
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  vector[N] offsets; // offsets in GLM
  real eps; // small value required in power mean calculation to smoothly allow first-order approximation when p is small
}

// Transformed data
transformed data {
  int Kc = K - 1;
  matrix[N, K] Xc;  // centered version of X 
  vector[Kc] means_X;  // column means before centering
  Xc[, 1] = X[, 1];
  for (i in 2:K) {
    means_X[i-1] = mean(X[, i]);
    Xc[, i] = X[, i] - means_X[i-1];
  }
  row_vector[Nv] v_ones = rep_row_vector(1, Nv);
}

// The parameters accepted by the model. 
parameters {
  vector[K+1] beta;  // population-level effects
  real<lower=0> phi;  // shape parameter
  real p; // power for power mean susceptibility calculation
}


transformed parameters {
  matrix[N, K+1] Xa;  // centred population-level design matrix, now with power mean added
  vector[K] means_Xa;  // column means before centering, now with power mean added
  vector[N+2] WM; // weighted mean
  vector[N+2] PM; // inversed weighted mean
  vector[N] PPM; // averaged over prior 2 months
  vector[N] PPM_standardise; // and scaled
  
  // Calculate power means
  if(abs(p) > eps){
	  WM = (v_ones*(W.*((Sn^p - 1)/p)))';
	  PM = ((p*WM + 1)^(1/p)).*n;
  }else{
    WM = (v_ones*(W.*(log(Sn)+p*log(Sn)^2/2)))';
    PM = (exp(WM-p*WM^2/2)).*n;
  }
  for(m in 1:N){
    PPM[m] = mean(PM[m:(m+1)]);
  }
  PPM_standardise = (PPM - mean(PPM))/sd(PPM);

  // Add power mean to design matrix
  Xa[,1:K] = Xc;
  Xa[,K+1] = PPM_standardise - mean(PPM_standardise);
  means_Xa[K] = mean(PPM_standardise);
  means_Xa[1:Kc] = means_X;
}


// The model to be estimated. 
model {
  phi~gamma(0.01,0.01);
  beta~normal(0,100000);
  p~normal(1,2);
  Y ~ neg_binomial_2_log(Xa*beta + offsets, phi);
}


generated quantities {
  // actual intercept
  real Intercept = beta[1] - dot_product(means_Xa, beta[2:(K+1)]);

  real log_post;
  log_post=neg_binomial_2_log_lpmf(Y | Xa*beta + offsets, phi) + gamma_lpdf(phi | 0.01, 0.01) + normal_lpdf(p | 0, 10);
  

  // log-likelihood
  vector[N] log_lik;
  for (m in 1:N) {
    log_lik[m] = neg_binomial_2_log_lpmf(Y[m] | Xa[m]*beta + offsets[m], phi);
  }

}