// Stan Model for Study_3, truncated model
functions {
  real partial_sum_fullddm(array[] real rt_slice, int start, int end,
    real a, real t0_m, real t0_s, real zr_m, real zr_s, array[] real v_m, real v_s,
    array[] int resp, array[] int cnd, real upper_bound) {
      real ans = 0;
      for (i in start:end) { //Truncation upper bound T[,upper_bound]
          real lcdf_resp_1 = wiener_lcdf(upper_bound | a, t0_m, zr_m, v_m[cnd[i]], v_s,  zr_s, t0_s);
          real lcdf_resp_0 = wiener_lcdf(upper_bound | a, t0_m, 1 - zr_m, -v_m[cnd[i]], v_s,  zr_s, t0_s);
        	if (resp[i] == 1) { // upper threshold
        	  ans += wiener_lpdf(rt_slice[i+1-start] | a, t0_m, zr_m, v_m[cnd[i]], v_s,  zr_s, t0_s);
          } else { // lower threshold (mirror drift and starting point!)
            ans += wiener_lpdf(rt_slice[i+1-start] | a, t0_m, 1 - zr_m, -v_m[cnd[i]], v_s,  zr_s, t0_s);
          }
          #ans += -(log(exp(lcdf_resp_0) + exp(lcdf_resp_1)));
          ans += -log_sum_exp(lcdf_resp_0, lcdf_resp_1);
      } //end for
      return ans;
    } //partial_sum_fullddm
}
  
data {
  int<lower=0> N;                     // No trials
  array[N] real rt;                   // response times (seconds)
  array[N] int<lower=1, upper=2> cnd;   // stimulus type/condition
  array[N] int<lower=0, upper=1> resp;      // responses (0,1)
  real upper_bound;
}

parameters {
  real<lower=0.5, upper=3> a;        // threshold separation
  real<lower=.3, upper=.7> zr_m;   // bias towards response 1
  array[2] real<lower=0,upper=5> v_m_pos;  // mean drift for 2 stimulus types, second is inverted
  real<lower=0.2, upper=min(rt)-1e-9> t0_m;     // min non-decision-time
}

transformed parameters {
  // the prior and the constraints are on positive drift rates
  // assign the correct sign to each of the drift rates
  array[2] real v_m;
  v_m[1] = v_m_pos[1];
  v_m[2] = -v_m_pos[2];
}

model {
  a ~ normal(1, 1) T[0.5, 3];
  zr_m ~ normal(0.5, .1) T[.3, .7];
  for (i in 1:2)
    v_m_pos[i]  ~ normal(2, 3) T[0, 5];
  t0_m ~ normal(0.435, .12) T[0.2, 1];
  
  
  // using reduce_sum to allow for parallel processing
  target += reduce_sum(partial_sum_fullddm, rt, 1,
    a, t0_m, 0, zr_m, 0, v_m, 0, resp, cnd, upper_bound);
}

