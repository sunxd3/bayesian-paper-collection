//Stan model file for hierarchical analysis of Plesckac et al. (2018), study 2
//with priors from Johnson!
functions {
  real partial_sum_fullddm(array[] real rt_slice, int start, int end,
    matrix a, matrix t0_m, real t0_s, matrix zr_m, real zr_s, matrix v_m, real v_s,
    array[] int resp, array[] int wCon, array[] int wConTarget, array[] int sub, real upper_bound) {
      real ans = 0;
      for (i in start:end) {
        real lcdf_resp_1 = wiener_lcdf(upper_bound | a[wCon[i], sub[i]], t0_m[wConTarget[i], sub[i]], zr_m[wCon[i], sub[i]], v_m[wConTarget[i], sub[i]], v_s,  zr_s, t0_s);
        real lcdf_resp_0 = wiener_lcdf(upper_bound | a[wCon[i], sub[i]], t0_m[wConTarget[i], sub[i]], 1 - zr_m[wCon[i], sub[i]], -v_m[wConTarget[i], sub[i]], v_s,  zr_s, t0_s);
        if (resp[i] == 1) {
          // upper threshold
          ans += wiener_lpdf(rt_slice[i+1-start] | a[wCon[i], sub[i]], t0_m[wConTarget[i], sub[i]], zr_m[wCon[i], sub[i]], v_m[wConTarget[i], sub[i]], v_s,  zr_s, t0_s);
        } else {
          // lower threshold (mirror drift and starting point!)
          ans += wiener_lpdf(rt_slice[i+1-start] | a[wCon[i], sub[i]], t0_m[wConTarget[i], sub[i]], 1 - zr_m[wCon[i], sub[i]], -v_m[wConTarget[i], sub[i]], v_s,  zr_s, t0_s);
        }
        ans += -log_sum_exp(lcdf_resp_0, lcdf_resp_1);
      } // end for
      return ans;
    }
}

data {
  int nSub;         // number subjects
  int nBtwn;        // between subject index
  int nWConTarget;  // within subject index w object
  int nWCon;        // within subject condition wo object
  int nData;        // number trials
  
  array[nData] real rt;                       // response times (seconds)
  array[nData] int wCon;
  array[nData] int wConTarget;
  array[nData] int resp;    // responses (0,1)
  array[nData] int sub;
  array[nData] int btwnCon;
  real upper_bound;
}


parameters {
  // group parameters: different sets per condition
  matrix [nWConTarget, nBtwn] muDelta;
  matrix [nWConTarget, nBtwn] muNDT;
  matrix [nWCon, nBtwn] muBeta;
  matrix [nWCon, nBtwn] muAlpha;
  
  array [nBtwn] real sigmaAlpha;
  array [nBtwn] real sigmaBeta;
  array [nBtwn] real sigmaDelta;
  array [nBtwn] real sigmaNDT;
  
  //individual parameters
  matrix [nWCon, nSub] beta;
  matrix [nWCon, nSub] alpha;
  matrix [nWConTarget, nSub] delta;
  matrix [nWConTarget, nSub] ndt;
}

transformed parameters {
  array [nBtwn] real tauDelta;
  array [nBtwn] real tauBeta;
  array [nBtwn] real tauAlpha;
  array [nBtwn] real tauNDT;
  
  for (btwn in 1:nBtwn) {
    tauAlpha[btwn] = 1/sigmaAlpha[btwn];
    tauBeta[btwn] = 1/sigmaBeta[btwn];
    tauNDT[btwn] = 1/sigmaNDT[btwn];
    tauDelta[btwn] = 1/sigmaDelta[btwn];
  }
} 

model {
  for (j in 1:nBtwn) { // between subject index 
  // remove between index completely if no btwn condtion 
     for (i in 1:nWConTarget){ // within subject index w object
         muDelta[i,j] ~ uniform(-5,5);
         muNDT[i,j] ~ uniform(.001,1);
      } 
      for (i in 1:nWCon){ // within subject condition wo object
         muBeta[i,j] ~ uniform(.1,.9);
         muAlpha[i,j]  ~ uniform(.1,5);
      } 
    tauDelta[j] ~ gamma(.001, .001);
    tauBeta[j] ~ gamma(.001, .001);
    tauAlpha[j] ~ gamma(.001, .001);
    tauNDT[j] ~ gamma(.001, .001);
  }

  
  for (j in 1:nSub){ // subject index
    for (i in 1:nWCon){ // within con index wo object
        beta[i,j] ~ normal( muBeta[i,btwnCon[j]] , sigmaBeta[btwnCon[j]] ) T[.1,.9];
        alpha[i,j]  ~ normal( muAlpha[i,btwnCon[j]] , sigmaAlpha[btwnCon[j]] ) T[.1,5];
    } 
    for (i in 1:nWConTarget){ // within con index w object
        delta[i,j] ~ normal( muDelta[i,btwnCon[j]] , sigmaDelta[btwnCon[j]] ) T[-5,5];
        ndt[i,j] ~ normal( muNDT[i,btwnCon[j]] , sigmaNDT[btwnCon[j]] ) T[.001,1];
    }
  }


  target += reduce_sum(partial_sum_fullddm, rt, 1,
    alpha, ndt, 0, beta, 0, delta, 0, resp, wCon, wConTarget,
    sub, upper_bound);
}
