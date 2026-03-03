data {
  int A;                                  // number of animals
  int R;                                  // number regions
  int G;                                  // number of groups
  array[A] int<lower=1,upper=G> group_idx;// group index per animal

  int<lower=1> N;                         // total number of datapoints
  array[N] int<lower=1,upper=A> a;        // animal index for each datapoint
  array[N] int<lower=1,upper=R> r;        // region index for each datapoint
  array[N] int<lower=0> y;                // raw count data
}
parameters {
  array[G] vector<lower=0>[R] tau;        // standard deviation of log cell counts
  array[G] vector[R]          theta;      // mean of log cell counts per group and region
  array[A] vector[R]          gamma_raw;  // relative mean of log cell counts, per datapoint
  array[A] vector<lower=0>[R] kappa;     // horseshoe multiplier, per datapoint
}
transformed parameters {
  array[A] vector[R] gamma;
  for (i in 1:A)
    gamma[i] = theta[group_idx[i]] + kappa[i] .* tau[group_idx[i]] .* gamma_raw[i];
}
model {
  // priors
  for (g in 1:G) {
    tau[g]   ~ normal(0, log(1.05));
    theta[g] ~ normal(5, 2);
  }
  for (i in 1:A) {
    gamma_raw[i] ~ normal(0, 1);
    kappa[i]     ~ normal(0, 1);
  }

  // likelihood only on observed cells
  for (n in 1:N)
    y[n] ~ poisson_log(gamma[a[n], r[n]]);
}
generated quantities {
  // posterior predictive for observed cells and (optionally) full grid
  array[N]    int y_rep_obs;
  array[A, R] int y_rep_full;
  for (n in 1:N)
    y_rep_obs[n] = poisson_log_rng(gamma[a[n], r[n]]);
  for (i in 1:A)
    for (j in 1:R)
      y_rep_full[i, j] = poisson_log_rng(gamma[i, j]);
      }