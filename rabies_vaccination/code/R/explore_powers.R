rm(list=ls())

library(viridis)

source("R/Functions/Lmeans.R")

# vaccination coverage in each of 100 villages
high_cov <- 0.7
low_cov <- 0.0
vax_vill <- matrix(low_cov,nrow=100,ncol=99)
for(i in 1:99){
  vax_vill[1:i,i]<-high_cov
}

# Susceptibilities
S <- 1-vax_vill

# Values of p to test
ps <- c(1:10,20,30)

# Calculate power Means for each vaccination ratio and each value of p
pMean <- matrix(NA,ncol=ncol(vax_vill),nrow=length(ps))
for(i in 1:length(ps)){
  p <- ps[i]
  pMean[i,] <- sapply(1:ncol(S),function(x) powMean(x=S[,x],p=ps[i]))
}

# Plot
pdf("Figs/explore_powers.pdf",width=7, height=3.4)
cex.axis<-0.7
cex.lab<-1
par(mfrow=c(1,2))
par(mar=c(3.5,3.5,1,1))
plot(NA,ylim=c(min(pMean),max(pMean)),xlim=c(0.01,0.99),bty="l",lwd=1,axes=F,xlab="",ylab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
mtext("Power mean susceptibility",side=2,line=1.8,cex=cex.lab)
mtext("Proportion area vaccinated to 70%",side=1,line=1.8,cex=cex.lab)
cols <- viridis(length(ps))
for(i in 0:10){
  lines(rep(i/10,2),c(0,1),lty=2,col="grey90")
  lines(c(0,1),rep(i/10,2),lty=2,col="grey90")
}
for(i in 1:length(ps)){
  lines(seq(0.01,0.99,0.01),pMean[i,],lty=1,lwd=1,col=cols[i])
}
legend("bottomleft",paste0("p=",c(ps)),lty=1,col=cols,lwd=1,ncol=2,box.col = "white",cex=cex.axis+0.1,bty="n")
legend("topright","A",text.font=2,bty="n")
box(bty="l")
plot(NA,ylim=c(min(1-pMean),max(1-pMean)),xlim=c(0.01,0.99),bty="l",lwd=1,axes=F,xlab="",ylab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
mtext("1 - power mean susceptibility\n(effective coverage)",side=2,line=1.8,cex=cex.lab)
mtext("Proportion area vaccinated to 70%",side=1,line=1.8,cex=cex.lab)
cols <- viridis(length(ps))
for(i in 0:10){
  lines(rep(i/10,2),c(0,1),lty=2,col="grey90")
  lines(c(0,1),rep(i/10,2),lty=2,col="grey90")
}
for(i in 1:length(ps)){
  lines(seq(0.01,0.99,0.01),1-pMean[i,],lty=1,lwd=1,col=cols[i])
}
legend("topleft","B",text.font=2,bty="n")
box(bty="l")
dev.off()


table <- round(t(1-pMean[,c(1,25,50,75,99)])*100,1)
colnames(table) <- paste0("p=",ps)
rownames(table) <- c(1,25,50,75,99)
write.csv(table,"output/explore_powers.csv")

