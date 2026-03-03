
## Transform to and from a stable scale that is linearly related to x^p
## The idea is to smoothly allow first-order approximation when p is small
powTrans <- function(x, p, eps=1e-5, warn=10){
	if(is.null(eps)) eps<-1e-5
	if(abs(p) > eps){
		return((x^p - 1)/p)
	}
	ell <- log(x)
	if ((m<-max(abs(ell)))>warn)
		warning("big value (abslog ", m, ") in powTrans")
	return(ell+p*ell^2/2)
}

invTrans <- function(y, p, eps=1e-5){
	if(is.null(eps)) eps<-1e-5
	if(abs(p) > eps){
+ 		return((p*y + 1)^(1/p))
	}
	return(exp(y-p*y^2/2))
}

wmean <- function(x, wts=1){
	return(sum(wts*x)/sum(wts+0*x))
}

powMean <- function(x, p, wts=1, eps=NULL, norm=TRUE){
	if(norm){
		xg <- exp(mean(log(x)))
		x <- x/xg
	}
	xtr <- powTrans(x, p, eps)
	mtr <- wmean(xtr, wts)
	tr <- invTrans(mtr, p, eps)
	if(norm) return(tr*xg)
	return(tr)
}

bpMean <- function(x, p, ...){
	# return(log(powMean(exp(x), p, ...)))
	return(log(powMean(exp(x), qlogis((p+1)/2), ...)))
}
