//Stan model file for hierarchical analysis of Plesckac et al. (2018), study 4
//with priors from Johnson!
functions {
  real partial_sum_fullddm(array[] real rt_slice, int start, int end,
    matrix a, matrix t0_m, real t0_s, matrix zr_m, real zr_s, matrix v_m, real v_s,
    array[] int resp, array[] int wCon, array[] int wConTarget, array[] int sub, 
    real upper_bound, array[] int y_bin) {
      real ans = 0;
      for (i in start:end) {
        if (y_bin[i] == 1) {
          if (resp[i] == 1) {
            // upper threshold
            ans += wiener_lpdf(rt_slice[i+1-start] | a[wCon[i], sub[i]], 
                                                          t0_m[wConTarget[i], sub[i]], 
                                                          zr_m[wCon[i], sub[i]], 
                                                          v_m[wConTarget[i], sub[i]], 
                                                          v_s,  zr_s, t0_s);
          } else {
            // lower threshold (mirror drift and starting point!)
            ans += wiener_lpdf(rt_slice[i+1-start] | a[wCon[i], sub[i]], 
                                                          t0_m[wConTarget[i], sub[i]], 
                                                          1 - zr_m[wCon[i], sub[i]], 
                                                          -v_m[wConTarget[i], sub[i]], 
                                                          v_s,  zr_s, t0_s);
          }
        } else {
          ans += log(exp(wiener_lccdf(upper_bound | a[wCon[i], sub[i]], 
                                                          t0_m[wConTarget[i], sub[i]], 
                                                          zr_m[wCon[i], sub[i]], 
                                                          v_m[wConTarget[i], sub[i]], 
                                                          v_s,  zr_s, t0_s)) +
                  exp(wiener_lccdf(upper_bound | a[wCon[i], sub[i]], 
                                                          t0_m[wConTarget[i], sub[i]], 
                                                          1 - zr_m[wCon[i], sub[i]], 
                                                          -v_m[wConTarget[i], sub[i]], 
                                                          v_s,  zr_s, t0_s)));
        }
      } // end for
      return ans;
    }
}

data {
  int nSub;         // number subjects
  int nWConTarget;  // within subject index w object
  int nWCon;        // within subject condition wo object
  int nData;        // number trials

  array[nData] real rt;                       // response times (seconds)
  array[nData] int wCon;
  array[nData] int wConTarget;
  array[nData] int resp;    // responses (0,1)
  array[nData] int sub;
  real upper_bound;
  array[nData] int y_bin;   // stores bins for censored data, 0 censored resp 0, 2 censpred resp 1
}


parameters {
  // group parameters: different sets per condition
  array [nWConTarget] real muDelta;
  array [nWConTarget] real muNDT;
  array [nWCon] real muBeta;
  array [nWCon] real muAlpha;
  
  real sigmaAlpha;
  real sigmaBeta;
  real sigmaDelta;
  real sigmaNDT;
  
  //individual parameters
  matrix [nWCon, nSub] beta;
  matrix [nWCon, nSub] alpha;
  matrix [nWConTarget, nSub] delta;
  matrix [nWConTarget, nSub] ndt;
}

transformed parameters {
  real tauDelta = 1/sigmaDelta;
  real tauBeta = 1/sigmaBeta;
  real tauAlpha = 1/sigmaAlpha;
  real tauNDT = 1/sigmaNDT;
} 

model {
 for (i in 1:nWConTarget){ // within subject index w object
     muDelta[i] ~ uniform(-5,5);
     muNDT[i] ~ uniform(.001,1);
  } 
  for (i in 1:nWCon){ // within subject condition wo object
     muBeta[i] ~ uniform(.1,.9);
     muAlpha[i]  ~ uniform(.1,5);
  } 
  tauDelta ~ gamma(.001, .001);
  tauBeta ~ gamma(.001, .001);
  tauAlpha ~ gamma(.001, .001);
  tauNDT ~ gamma(.001, .001);

  
  for (j in 1:nSub){ // subject index
    for (i in 1:nWCon){ // within con index wo object
        beta[i,j] ~ normal( muBeta[i] , sigmaBeta) T[.1,.9];
        alpha[i,j]  ~ normal( muAlpha[i] , sigmaAlpha) T[.1,5];
    } 
    for (i in 1:nWConTarget){ // within con index w object
        delta[i,j] ~ normal( muDelta[i] , sigmaDelta) T[-5,5];
        ndt[i,j] ~ normal( muNDT[i] , sigmaNDT) T[.001,1];
    }
  }


  target += reduce_sum(partial_sum_fullddm, rt, 1,
    alpha, ndt, 0, beta, 0, delta, 0, resp, wCon, wConTarget,
    sub, upper_bound, y_bin);
}
