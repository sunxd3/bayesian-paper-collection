rm(list=ls())


# Read in data & models
#___________________

library(rstan) 
library(brms)
library(viridis)

model <- readRDS("output/stan_models/incidence_from_vax_model_district_stan.rds")
model_wo_incidence <- readRDS("output/stan_models/incidence_from_vax_model_district_stan_woPriorCases.rds")
model_standardise <- readRDS("output/stan_models/incidence_from_vax_model_district_stan_standardise.rds")
model_wo_incidence_standardise <- readRDS("output/stan_models/incidence_from_vax_model_district_stan_woPriorCases_standardise.rds")

data_dist <- read.csv("output/incidence_coverage_model_data_district.csv")
dogs <- as.matrix(read.csv("output/dogPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))
vax_vill <- as.matrix(read.csv("output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) 

S <- 1 - vax_vill
n <- exp(colMeans(log(S)))
Sn <- S/rep(n,each=nrow(vax_vill))

source("R/Functions/Lmeans.R")



# Extract model parameters
#___________________

model_summary <- as.data.frame(summary(model,pars = c("beta","p","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars <- round(model_summary[c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars) <- c("Intercept","Log cases/dog over last 2 months","Log dogs/km2","Power mean susceptibility over last 2 months","p","size")
model_standardise_summary <- as.data.frame(summary(model_standardise,pars = c("beta","p","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_standardise <- round(model_standardise_summary[c("Intercept","beta[2]","beta[3]","beta[4]","p","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_standardise) <- c("Intercept","Log cases/dog over last 2 months","Log dogs/km2","Power mean susceptibility over last 2 months","p","size")
model_wo_incidence_summary <- as.data.frame(summary(model_wo_incidence,pars = c("beta","p","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_incidence <- round(model_wo_incidence_summary[c("Intercept","beta[2]","beta[3]","p","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_incidence) <- c("Intercept","Log dogs/km2","Power mean susceptibility over last 2 months","p","size")
model_wo_incidence_standardise_summary <- as.data.frame(summary <- summary(model_wo_incidence_standardise,pars = c("beta","p","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_incidence_standardise <- round(model_wo_incidence_standardise_summary[c("Intercept","beta[2]","beta[3]","p","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_incidence_standardise) <- c("Intercept","Log dogs/km2","Power mean susceptibility over last 2 months","p","size")

pars_order <- c("Intercept","vax","Power mean susceptibility over last 2 months","Log cases/dog over last 2 months","Log dogs/km2","size","p")                               
pars_prep <- data.frame(matrix("", nrow=length(pars_order), ncol=2,dimnames = list(pars_order,c("Full model","Without prior cases"))))
pars_prep_1 <- paste0(pars$mean," (",pars$`2.5%`,", ",pars$`97.5%`,")")
pars_prep_2 <- paste0(pars_wo_incidence$mean," (",pars_wo_incidence$`2.5%`,", ",pars_wo_incidence$`97.5%`,")")
pars_prep[rownames(pars),1] <- pars_prep_1
pars_prep[rownames(pars_wo_incidence),2] <- pars_prep_2
write.csv(pars_prep,"output/model_dist_month_pars_all_power_mean_models_MSversion.csv")



# Plot standardised parameters
#___________________

cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5

pdf("Figs/MonthlyModelCases&PowerMeanSuscDist.pdf",width=4.5, height=4)

par(fig=c(0,1,0.3,1))
par(mar=c(2.5,7,0.5,1))
pars_sub <- pars_standardise[2:(nrow(pars_standardise)-2),]
pars_sub_wo_incidence <- pars_wo_incidence_standardise[2:(nrow(pars_wo_incidence_standardise)-2),]
plot(NA,xlim=c((min(pars_sub$`2.5%`,pars_sub_wo_incidence$`2.5%`)),(max(pars_sub$`97.5%`,pars_sub_wo_incidence$`97.5%`))),ylim=c(0,nrow(pars_sub)),axes=F,ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub)), function(x) paste(strwrap(x, 20), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,1),las=2)
axis(1,cex.axis=cex.axis,padj=-1.5,at=log(pretty(c(exp(c(pars_sub$`2.5%`,pars_sub_wo_incidence$`2.5%`,pars_sub$`97.5%`,pars_sub_wo_incidence$`97.5%`))))),label=pretty(exp(c(pars_sub$`2.5%`,pars_sub_wo_incidence$`2.5%`,pars_sub$`97.5%`,pars_sub_wo_incidence$`97.5%`))))
lines(c(0,0),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.3,cex=cex.lab)
box(bty="l")
arrows(x0=(pars_sub$`2.5%`),x1=(pars_sub$`97.5%`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col="navy")
points((pars_sub$mean),seq(nrow(pars_sub)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
arrows(x0=(pars_sub_wo_incidence$`2.5%`),x1=(pars_sub_wo_incidence$`97.5%`),y0=seq(nrow(pars_sub)-0.7,1-0.7,-1)[2:nrow(pars_sub)],y1=seq(nrow(pars_sub)-0.7,1-0.7,-1)[2:nrow(pars_sub)],length=0,lwd=2,col="lightblue")
points((pars_sub_wo_incidence$mean),seq(nrow(pars_sub)-0.7,1-0.7,-1)[2:nrow(pars_sub)],col="pink",pch=20,cex=1.3)
legend(0.6,4.4,legend="C",text.font = 2,bty="n")
legend(log(0.3),3,legend=c("With prior incidence","Without prior incidence"),col=c("navy","lightblue"),lwd=2,cex=0.7,pt.cex =1,text.col = "white",bg="white")
legend(log(0.315),3,legend=c("With prior incidence","Without prior incidence"),col=c("red","pink"),lwd=2,cex=0.7,pt.cex =1,lty=0,pch=20,bty="n",)
legend("topright",legend="A",text.font = 2,bty="n")

par(fig=c(0,1,0,0.3),new=T)
par(mar=c(2.2,2,0.5,1))
p <- pars_standardise["p",]
p_wo_incidence <- pars_wo_incidence_standardise["p",]
plot(NA,xlim=c((min(p$`2.5%`,p_wo_incidence$`2.5%`)),(max(p$`97.5%`,p_wo_incidence$`97.5%`))),ylim=c(0,nrow(p)),axes=F,ylab="",xlab="")
# axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(p)), function(x) paste(strwrap(x, 20), collapse = "\n")),at=seq(0.5,nrow(p)-0.5,1),las=2,font.axis = 3)
axis(1,cex.axis=cex.axis,padj=-1.5)
lines(c(1,1),c(0,nrow(p)),lty=3)
mtext("p",font=3,side=1,line=1.2,cex=cex.lab)
box(bty="l")
arrows(x0=(p$`2.5%`),x1=(p$`97.5%`),y0=seq(nrow(p)-0.5,0.5,-1),y1=seq(nrow(p)-0.5,0.5,-1),length=0,lwd=2,col="navy")
points((p$mean),seq(nrow(p)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
arrows(x0=(p_wo_incidence$`2.5%`),x1=(p_wo_incidence$`97.5%`),y0=1-0.7,y1=1-0.7,length=0,lwd=2,col="lightblue")
points((p_wo_incidence$mean),1-0.7,col="pink",pch=20,cex=1.3)
legend(5,1.25,legend="B",text.font = 2,bty="n")

dev.off()




# Plot fit to data
#___________________

# Plot predictions and data over time
par(mar=c(2.5,2.5,2,1))
# par(fig=c(0.5,1,0,0.5),new=T)
samples <- posterior_samples(model, pars = c("Intercept","beta","p","phi"))
samples <- samples[,-which(colnames(samples)=="beta[1]")]
preds_mat <- matrix(NA,nrow=nrow(data_dist),ncol=nrow(samples))
X <- t(as.matrix(cbind(1,data_dist[,c("log_case_rate_last2monthMean","log_dog_density")],"power_mean_last2MonthMean"=NA,log(data_dist$dogs))))
for(i in 1:nrow(samples)){
  power_mean <- sapply(1:ncol(vax_vill),function(x) powMean(x=1-vax_vill[,x],p=samples[i,"p"],wts = dogs[,x]))
  X["power_mean_last2MonthMean",3:nrow(data_dist)] <- sapply(3:nrow(data_dist),function(x) mean(power_mean[(x-c(2:1))]))
  preds_mat[,i] <- rnbinom(nrow(data_dist),mu=exp(colSums(X*c(as.numeric(samples[i,c(1:(nrow(X)-1))]),1))),size=samples[i,"phi"])
}
preds_lower <- apply(preds_mat,1,quantile,0.025,na.rm=T)[-c(1:2)]
preds_upper <- apply(preds_mat,1,quantile,0.975,na.rm=T)[-c(1:2)]
plot(NA,ylim=c(0,max(preds_upper,data_dist$cases)),xlim=c(1,max(data_dist$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_dist$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col="skyblue",border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases~data_dist$month,col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend("topright",c("data","model 95PI"),col=c("navy","skyblue"),pch=c(20,15),cex=0.75,bty="n",pt.cex = c(cex.pt,1.5))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #4%
# legend("topleft",legend="D",text.font = 2,bty="n")


