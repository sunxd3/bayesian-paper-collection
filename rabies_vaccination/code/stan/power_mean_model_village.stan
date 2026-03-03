
// The input data.
data {
  int<lower=1> N;  // total number of observations
  int Nv; // number of villages
  int Nm; // number of observations per village (months)
  int Nn; // total number of neighbours summed over villages
  int Nnn; // total number of non-neighbours summed over villages
  int Nn_v[Nv]; // number of neighbouring villages per village
  int Nnn_v[Nv]; // number of non-neighbouring villages per village
  int n_index[Nn];
  int nn_index[Nnn];
  int Y[N];  // response variable
  matrix[Nn,Nm+2] S_n;  // normalised susceptibility in each neighbouring village each month
  matrix[Nnn,Nm+2] S_nn;  // normalised susceptibility in each non-neighbouring village each month
  matrix[Nv,Nm+2] norm_n; // normalisation constants for neighbour coverages
  matrix[Nv,Nm+2] norm_nn; // normalisation constants for non-neighbour coverages
  matrix[Nn,Nm+2] W_n;  // weight for each neighbouring village each month
  matrix[Nnn,Nm+2] W_nn;  // weight for each non-neighbouring village each month
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int V[N]; // random effect values
  vector[N] offsets; // offsets in GLM
  real eps; // small value required in power mean calculation to smoothly allow first-order approximation when p is small
}

// Transformed data
transformed data {
  int Kc = K - 1;
  matrix[N, K] Xc;  // centered version of X 
  vector[Kc] means_X;  // column means before centering
  vector[2] two_ones_vec = rep_vector(1,2);

  Xc[, 1] = X[, 1];
  for (i in 2:K) {
    means_X[i-1] = mean(X[, i]);
    Xc[, i] = X[, i] - means_X[i-1];
  }
}

// The parameters accepted by the model. 
parameters {
  vector[K+2] beta;  // population-level effects
  real<lower=0> phi;  // shape parameter
  real p; // power for power mean susceptibility calculation
  real<lower=0> sigma_village; // standard deviation of village RE
  vector[Nv] gamma; // standardised village effects
}

// Calculate quantities for model from data and parameters
transformed parameters {
  matrix[N, K+2] Xa;  // centred population-level design matrix, now with power means for neighbours and non-neighbours added
  vector[K+1] means_Xa;  // column means before centering, now with power means added
  matrix[Nn,Nm+2] WPS_n; // weighted powers of neighbour susceptibilities
  matrix[Nnn,Nm+2] WPS_nn; // weighted powers of non-neighbour susceptibilities
  matrix[Nv,Nm+2] WM_n; // weighted mean neighbours
  matrix[Nv,Nm+2] WM_nn; // weighted mean non-neighbours
  matrix[Nv,Nm+2] PM_n; // inversed weighted mean neighbours
  matrix[Nv,Nm+2] PM_nn; // inversed weighted mean non-neighbours
  matrix[Nv,Nm] PPM_n; // averaged over prior 2 months neighbours
  matrix[Nv,Nm] PPM_nn; // averaged over prior 2 months non-neighbours
  vector[Nv] gamma_t; // actual village effects

  // Calculate power means
  if(abs(p) > eps){
    int pos_n=1; // to count through neighbours
    int pos_nn=1; // to count through non-neighbours
    WPS_n = (W_n.*((S_n^p - 1)/p));
    WPS_nn = (W_nn.*((S_nn^p - 1)/p));
    for(v in 1:Nv){
      WM_n[v] = rep_row_vector(1, Nn_v[v])*WPS_n[segment(n_index,pos_n,Nn_v[v]),];
      WM_nn[v] = rep_row_vector(1, Nnn_v[v])*WPS_nn[segment(nn_index,pos_nn,Nnn_v[v]),];
      pos_n = pos_n + Nn_v[v];
      pos_nn = pos_nn + Nnn_v[v];
    }
	  PM_n = ((p*WM_n + 1)^(1/p)).*norm_n;
	  PM_nn = ((p*WM_nn + 1)^(1/p)).*norm_nn;
  }else{
    int pos_n=1; // to count through neighbours
    int pos_nn=1; // to count through non-neighbours
    WPS_n = W_n.*(log(S_n)+p*log(S_n)^2/2);
    WPS_nn = W_nn.*(log(S_nn)+p*log(S_nn)^2/2);
    for(v in 1:Nv){
      WM_n[v] = rep_row_vector(1, Nn_v[v])*WPS_n[segment(n_index,pos_n,Nn_v[v]),];
      WM_nn[v] = rep_row_vector(1, Nnn_v[v])*WPS_nn[segment(nn_index,pos_nn,Nnn_v[v]),];
      pos_n = pos_n + Nn_v[v];
      pos_nn = pos_nn + Nnn_v[v];
    }
    PM_n = (exp(WM_n-p*WM_n^2/2)).*norm_n;
    PM_nn = (exp(WM_nn-p*WM_nn^2/2)).*norm_nn;
  }
  for(m in 1:Nm){
    PPM_n[,m] = ((PM_n[,m:(m+1)])*two_ones_vec)/2;
    PPM_nn[,m] = ((PM_nn[,m:(m+1)])*two_ones_vec)/2;
  }

  // Add power means to design matrix
  Xa[,1:K] = Xc;
  Xa[,K+1] = to_vector(PPM_n) - mean(PPM_n);
  Xa[,K+2] = to_vector(PPM_nn) - mean(PPM_nn);
  means_Xa[K] = mean(PPM_n);
  means_Xa[K+1] = mean(PPM_nn);
  means_Xa[1:Kc] = means_X;
  
  // Village effects
  gamma_t = (sigma_village * gamma);
}


// The model to be estimated. 
model {
  beta~normal(0,100000);
  phi~gamma(0.01,0.01);
  p~normal(1,2);
  sigma_village~exponential(1e-3);
  gamma ~ normal(0, 1);
  
  Y ~ neg_binomial_2_log(Xa*beta + gamma_t[V] + offsets, phi);
}


generated quantities {
  // actual intercept
  real Intercept = beta[1] - dot_product(means_Xa, beta[2:(K+2)]);

}