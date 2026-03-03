rm(list=ls())


# Read in data & models
#___________________

library(rstan) 
library(brms)
library(viridis)
library(rgeos)
library(rgdal)
library(raster)

model <- readRDS("output/stan_models/incidence_from_vax_model_village_stan.rds")
model_wo_distant_cases <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_distant_cases.rds")
model_wo_cases <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_cases.rds")
model_standardise <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_standardised.rds")
model_wo_distant_cases_standardise <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_distant_cases_standardised.rds")
model_wo_cases_standardise <- readRDS("output/stan_models/incidence_from_vax_model_village_stan_wo_cases_standardised.rds")

data_vill <- read.csv("Output/incidence_coverage_model_data_village.csv")
data_dist <- read.csv("Output/incidence_coverage_model_data_district.csv")
data_vill$susc_last2monthMean <- 1-data_vill$vax_last2monthMean
dogs <- as.matrix(read.csv("Output/dogPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))
vax_vill <- as.matrix(read.csv("Output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) 

data_stan <- readRDS("Output/power_mean_model_village_data.rds")
colnames(data_stan$X)

# Serengeti village shapefile
SD_vill <- readOGR("Output/SD_vill","SD_vill") 
SD_vill<-SD_vill[order(SD_vill$Vill_2012),]
SD_outline <- gUnaryUnion(SD_vill)# get district outline
SD_outline<-gBuffer(SD_outline,width=1) # get rid of a few tiny holes
SD_vill_gridded <- readOGR("data/GIS","SD_Villages_2012_From_HHS_250m_Lattice_UTM") 

load("Output/neighbour_notNeighbour_susceptibilities.Rdata")

source("R/Functions/Lmeans.R")

simulate <- F



# Extract model parameters
#___________________

model_summary <- as.data.frame(summary(model,pars = c("beta","p","phi","Intercept","sigma_village"),probs=c(0.025,0.5,0.975))$summary)
pars <- round(model_summary[c("Intercept","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[2]","beta[8]","beta[9]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars) <- c("Intercept","Log cases/dog over last 2 months in village","Log cases/dog at borders over last 2 months","Log cases/dog in non-bordering villages over last 2 months",
                    "Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village","Power mean susceptibility at borders over last 2 months",
                    "Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")
model_standardise_summary <- as.data.frame(summary(model_standardise,pars = c("beta","p","sigma_village","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_standardise <- round(model_standardise_summary[c("Intercept","beta[3]","beta[4]","beta[5]","beta[6]","beta[7]","beta[2]","beta[8]","beta[9]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_standardise) <- c("Intercept","Log cases/dog over last 2 months in village","Log cases/dog at borders over last 2 months","Log cases/dog in non-bordering villages over last 2 months",
                                "Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village","Power mean susceptibility at borders over last 2 months",
                                "Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")
model_wo_distant_cases_summary <- as.data.frame(summary(model_wo_distant_cases,pars = c("beta","p","sigma_village","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_distant_cases <- round(model_wo_distant_cases_summary[c("Intercept","beta[3]","beta[4]","beta[5]","beta[2]","beta[6]","beta[7]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_distant_cases) <- c("Intercept","Log cases/dog over last 2 months in village","Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village",
                                     "Power mean susceptibility at borders over last 2 months","Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")
model_wo_distant_cases_standardise_summary <- as.data.frame(summary <- summary(model_wo_distant_cases_standardise,pars = c("beta","p","sigma_village","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_distant_cases_standardise <- round(model_wo_distant_cases_standardise_summary[c("Intercept","beta[3]","beta[4]","beta[5]","beta[2]","beta[6]","beta[7]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_distant_cases_standardise) <- c("Intercept","Log cases/dog over last 2 months in village","Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village",
                                                 "Power mean susceptibility at borders over last 2 months","Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")
model_wo_cases_summary <- as.data.frame(summary(model_wo_cases,pars = c("beta","p","sigma_village","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_cases <- round(model_wo_cases_summary[c("Intercept","beta[3]","beta[4]","beta[2]","beta[5]","beta[6]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_cases) <- c("Intercept","Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village",
                             "Power mean susceptibility at borders over last 2 months","Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")
model_wo_cases_standardise_summary <- as.data.frame(summary <- summary(model_wo_cases_standardise,pars = c("beta","p","sigma_village","phi","Intercept"),probs=c(0.025,0.5,0.975))$summary)
pars_wo_cases_standardise <- round(model_wo_cases_standardise_summary[c("Intercept","beta[3]","beta[4]","beta[2]","beta[5]","beta[6]","p","sigma_village","phi"),c("mean","sd","2.5%","97.5%")],2)
rownames(pars_wo_cases_standardise) <- c("Intercept","Log dogs/km2","Human:dog ratio","Susceptibility over last 2 months in village",
                                         "Power mean susceptibility at borders over last 2 months","Power mean susceptibility in non-bordering villages over last 2 months","p","Village RE SD","size")


# Prepare table for SI
pars_order <- c("Intercept",
                "vax","vax_neighbours","vax_notNeighbours",
                "Susceptibility over last 2 months in village","Power mean susceptibility at borders over last 2 months","Power mean susceptibility in non-bordering villages over last 2 months",
                "Log cases/dog over last 2 months in village","Log cases/dog at borders over last 2 months","Log cases/dog in non-bordering villages over last 2 months",
                "Log dogs/km2","Human:dog ratio","Village RE SD","size","p")                               
pars_prep <- data.frame(matrix("", nrow=length(pars_order), ncol=3,dimnames = list(pars_order,c("Full model","Without distant prior cases","Without prior cases"))))
pars_prep_1 <- paste0(pars$mean," (",pars$`2.5%`,", ",pars$`97.5%`,")")
pars_prep_2 <- paste0(pars_wo_distant_cases$mean," (",pars_wo_distant_cases$`2.5%`,", ",pars_wo_distant_cases$`97.5%`,")")
pars_prep_3 <- paste0(pars_wo_cases$mean," (",pars_wo_cases$`2.5%`,", ",pars_wo_cases$`97.5%`,")")
pars_prep[rownames(pars),1] <- pars_prep_1
pars_prep[rownames(pars_wo_distant_cases),2] <- pars_prep_2
pars_prep[rownames(pars_wo_cases),3] <- pars_prep_3
write.csv(pars_prep,"Output/model_vill_month_pars_all_power_mean_models_MSversion.csv")



# Simulate cases from data using models and record prediction intervals
#___________________

if(simulate==T){
  
  n_samples <-6000
  
  for(model_sim in c("model","model_wo_cases","model_wo_distant_cases")){
    samples_pars <- posterior_samples(get(model_sim), pars = c("Intercept","beta","p","phi"))
    samples <- sample(1:nrow(samples_pars),n_samples)
    samples_pars <- samples_pars[samples,]
    samples_reffs <- posterior_samples(get(model_sim), pars = c("gamma_t"))
    samples_reffs <- samples_reffs[samples,]
    samples_pars <- samples_pars[,-which(colnames(samples_pars)=="beta[1]")]
    preds_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
    if(model_sim=="model"){pars_sim <- c("susc_last2monthMean","log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean","log_dog_density","HDR")
    }else if(model_sim=="model_wo_cases"){pars_sim <- c("susc_last2monthMean","log_dog_density","HDR")
    }else if(model_sim=="model_wo_distant_cases"){pars_sim <- c("susc_last2monthMean","log_case_rate_last2monthMean","log_dog_density","HDR")}
    X <- t(as.matrix(cbind(1,data_vill[,pars_sim],
                           "power_mean_neighbours_last2MonthMean"=NA,"power_mean_notNeighbours_last2MonthMean"=NA,"village_re"=NA,log(data_vill$dogs))))
    power_mean_neighbours <- power_mean_notNeighbours <- matrix(NA, nrow=nrow(vax_vill),ncol=ncol(vax_vill))
    for(i in 1:nrow(samples_pars)){
      for(v in 1:nrow(vax_vill)){
        power_mean_neighbours[v,] <- sapply(1:ncol(vax_vill),function(x) powMean(x=S_n[[v]][,x],p=samples_pars[i,"p"],wts = W_n[[v]][,x]))
        power_mean_notNeighbours[v,] <- sapply(1:ncol(vax_vill),function(x) powMean(x=S_nn[[v]][,x],p=samples_pars[i,"p"],wts = W_nn[[v]][,x]))
      }
      X["power_mean_neighbours_last2MonthMean",] <- (c(cbind(NA,power_mean_neighbours[,-ncol(vax_vill)]))+c(cbind(NA,NA,power_mean_neighbours[,-((ncol(vax_vill)-1):ncol(vax_vill))])))/2
      X["power_mean_notNeighbours_last2MonthMean",] <- (c(cbind(NA,power_mean_notNeighbours[,-ncol(vax_vill)]))+c(cbind(NA,NA,power_mean_notNeighbours[,-((ncol(vax_vill)-1):ncol(vax_vill))])))/2
      X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
      preds_mat[,i] <- rnbinom(nrow(data_vill),mu=exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(length(pars_sim)+3))]),1,1))),size=samples_pars[i,"phi"])
    }
    preds_mat_dist <- rowsum((preds_mat),data_vill$month)
    preds_lower <- apply(preds_mat_dist,1,quantile,0.025,na.rm=T)[-c(1:2)]
    preds_upper <- apply(preds_mat_dist,1,quantile,0.975,na.rm=T)[-c(1:2)]
    
    save(preds_lower,preds_upper,
         file=paste0("Output/preds_PI_stan_village_",model_sim,".Rdata"))
  }
  
}



# Some outputs for manuscript text
#___________________

p_lower <- summary(model_wo_distant_cases)$summary["p","2.5%"]
p_upper <- summary(model_wo_distant_cases)$summary["p","97.5%"]
p_mean <- summary(model_wo_distant_cases)$summary["p","mean"]
samples <- posterior_samples(model_wo_distant_cases)

load("Output/preds_PI_stan_village_model_wo_distant_cases.Rdata")

# coverage of 40%. Either homogeneous or 50-50 between 20% and 60% (both at border and non-borders)
mean_homo <- 1-0.4
ps <- samples[,"p"]
mean_hetero <- sapply(1:6000,function(x){powMean(1-c(0.2,0.6),p = ps[x])})
mean((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)))
quantile((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)),c(0.025,0.975))

# coverage of homogeneous 40% vs heterogeneous 40%: 50-50 between 10% and 70% (both at border and non-borders)
mean_homo <- 1-0.4
mean_hetero <- sapply(1:6000,function(x){powMean(1-c(0.1,0.7),p = ps[x])})
mean((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)))
quantile((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)),c(0.025,0.975))

# coverage of homogeneous 40% vs heterogeneous 40%: 50-50 between 30% and 50% (both at border and non-borders)
mean_homo <- 1-0.4
mean_hetero <- sapply(1:6000,function(x){powMean(1-c(0.3,0.5),p = ps[x])})
mean((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)))
quantile((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)),c(0.025,0.975))

# coverage of 20%. Either homogeneous or 50-50 between 0% and 40% (both at border and non-borders)
mean_homo <- 1-0.2
mean_hetero <- sapply(1:6000,function(x){powMean(1-c(0,0.4),p = ps[x])})
mean((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)))
quantile((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)),c(0.025,0.975))

# coverage of homogeneous 20% vs heterogeneous 40%: 50-50 between 20% and 60% (both at border and non-borders)
mean_homo <- 1-0.2
mean_hetero <- sapply(1:6000,function(x){powMean(1-c(0.2,0.6),p = ps[x])})
mean((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)))
quantile((exp((samples[,"beta[6]"])*mean_hetero)*exp((samples[,"beta[7]"])*mean_hetero))/(exp((samples[,"beta[6]"])*mean_homo)*exp((samples[,"beta[7]"])*mean_homo)),c(0.025,0.975))



# Plot standardised parameters
#___________________

pdf("Figs/MonthlyModelCases&PowerMeanSuscVill.pdf",width=7, height=7)
cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5

par(fig=c(0,1,0.49,1))
par(mar=c(3,10,0.25,1))
pars_sub <- pars_standardise[2:(nrow(pars_standardise)-3),]
pars_sub_wo_distant_cases <- pars_wo_distant_cases_standardise[2:(nrow(pars_wo_distant_cases_standardise)-3),]
plot(NA,xlim=c((min(pars_sub$`2.5%`,pars_sub_wo_distant_cases$`2.5%`)),(max(pars_sub$`97.5%`,pars_sub_wo_distant_cases$`97.5%`))),ylim=c(0.2,nrow(pars_sub)-0.15),axes=F,ylab="",xlab="")
# axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub)), function(x) paste(strwrap(x, 30), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,1),las=2)
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(1,nrow(pars_sub),2)], function(x) paste(strwrap(x, 39), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("grey40"))
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(2,nrow(pars_sub),2)], function(x) paste(strwrap(x, 39), collapse = "\n")),at=seq(1.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("black"))
axis(1,cex.axis=cex.axis,padj=-1.5,at=log(pretty(c(exp(c(pars_sub$`2.5%`,pars_sub_wo_distant_cases$`2.5%`,pars_sub$`97.5%`,pars_sub_wo_distant_cases$`97.5%`))))),label=pretty(exp(c(pars_sub$`2.5%`,pars_sub_wo_distant_cases$`2.5%`,pars_sub$`97.5%`,pars_sub_wo_distant_cases$`97.5%`))))
lines(c(0,0),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.3,cex=cex.lab)
box(bty="l")
arrows(x0=(pars_sub$`2.5%`),x1=(pars_sub$`97.5%`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col=scales::alpha("dodgerblue",0.4))
points((pars_sub$mean),seq(nrow(pars_sub)-0.5,0.5,-1),col="navy",pch=20,cex=1.3)
arrows(x0=(pars_sub_wo_distant_cases$`2.5%`),x1=(pars_sub_wo_distant_cases$`97.5%`),y0=seq(nrow(pars_sub)-0.8,1-0.8,-1)[c(1,4:nrow(pars_sub))],y1=seq(nrow(pars_sub)-0.8,1-0.8,-1)[c(1,4:nrow(pars_sub))],length=0,lwd=2,col=scales::alpha("orange",0.3))
points((pars_sub_wo_distant_cases$mean),seq(nrow(pars_sub)-0.8,1-0.8,-1)[c(1,4:nrow(pars_sub))],col="darkorange2",pch=20,cex=1.3)
# legend(0.6,4.4,legend="C",text.font = 2,bty="n")
legend(log(0.43),nrow(pars_sub)+0.2,legend=c("With prior cases\nbeyond focal village ","Without prior cases\nbeyond focal village   "),col=c(scales::alpha("dodgerblue",0.4),scales::alpha("orange",0.3)),lwd=2,cex=cex.axis,pt.cex =1,text.col = "white",y.intersp = 1.7,bg="white",xpd=T)
legend(log(0.44),nrow(pars_sub)+0.2,legend=c("With prior cases\nbeyond focal village ","Without prior cases\nbeyond focal village   "),col=c("navy","darkorange2"),lwd=2,cex=cex.axis,pt.cex =1,lty=0,pch=20,bty="n",y.intersp = 1.7,xpd=T)
legend("topright",legend="A",text.font = 2,bty="n")

par(fig=c(0,1,0.38,0.5),new=T)
par(mar=c(2,2,0.7,1))
p <- pars_standardise["p",]
p_wo_distant_cases <- pars_wo_distant_cases_standardise["p",]
plot(NA,xlim=c((min(p$`2.5%`,p_wo_distant_cases$`2.5%`)),(max(p$`97.5%`,p_wo_distant_cases$`97.5%`))),ylim=c(0,nrow(p)),axes=F,ylab="",xlab="",new=T)
# axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(p)), function(x) paste(strwrap(x, 20), collapse = "\n")),at=seq(0.5,nrow(p)-0.5,1),las=2,font.axis = 3)
axis(1,cex.axis=cex.axis,padj=-1.5)
lines(c(1,1),c(0,nrow(p)),lty=3)
mtext("p",font=3,side=1,line=1,cex=cex.lab)
box(bty="l")
arrows(x0=(p$`2.5%`),x1=(p$`97.5%`),y0=seq(nrow(p)-0.5,0.5,-1),y1=seq(nrow(p)-0.5,0.5,-1),length=0,lwd=2,col=scales::alpha("dodgerblue",0.4))
points((p$mean),seq(nrow(p)-0.5,0.5,-1),col="navy",pch=20,cex=1.3)
arrows(x0=(p_wo_distant_cases$`2.5%`),x1=(p_wo_distant_cases$`97.5%`),y0=1-0.7,y1=1-0.7,length=0,lwd=2,col=scales::alpha("orange",0.3))
points((p_wo_distant_cases$mean),1-0.7,col="darkorange2",pch=20,cex=1.3)
legend(12.5,1.6,legend="B",text.font = 2,bty="n",xpd=T,inset=0)



# Plot power mean over district
#___________________

S <- 1-vax_vill
data_dist$arithmeticMean <- sapply(1:ncol(vax_vill),function(x) powMean(x=S[,x],p=1,wts = dogs[,x]))
data_dist$max_village_S <- sapply(1:ncol(vax_vill),function(x) max(S[,x]))
data_dist$min_village_S <- sapply(1:ncol(vax_vill),function(x) min(S[,x]))

par(fig=c(0,1,0,0.35),new=T)
par(mar=c(3,3.5,0.1,1))
p_lower <- summary(model)$summary["p","2.5%"]
p_upper <- summary(model)$summary["p","97.5%"]
p_mean <- summary(model)$summary["p","mean"]
plot(data_dist$arithmeticMean,type="l",ylim=c(0.4,0.93),bty="l",lwd=1,axes=F,xlab="",ylab="",lty=2)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Power mean susceptibility",side=2,line=1.7,cex=cex.lab)
mtext("Date",side=1,line=1.6,cex=cex.lab)
box(bty="l")
polygon(c(1:ncol(S),rev(1:ncol(S))),
        c(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_lower,wts = dogs[,x])),
          rev(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_upper,wts = dogs[,x])))),
        col=scales::alpha("orange",0.3),border = NA)
lines(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_mean,wts = dogs[,x])),col="darkorange2",lwd=2)

p_lower <- summary(model_wo_distant_cases)$summary["p","2.5%"]
p_upper <- summary(model_wo_distant_cases)$summary["p","97.5%"]
p_mean <- summary(model_wo_distant_cases)$summary["p","mean"]
polygon(c(1:ncol(S),rev(1:ncol(S))),
        c(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_lower,wts = dogs[,x])),
          rev(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_upper,wts = dogs[,x])))),
        col=scales::alpha("dodgerblue",0.4),border = NA)
lines(sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_mean,wts = dogs[,x])),col="navy",lwd=2)
lines(data_dist$arithmeticMean,lwd=1,lty=2)
legend("topright",legend="C",text.font = 2,bty="n")

legend(-3.8,0.62,legend=c("Arithmetic mean","With prior cases beyond focal village","Without prior cases beyond focal village"),col=c(NA,scales::alpha("orange",0.3),scales::alpha("dodgerblue",0.4)),pch=15,cex=cex.axis,pt.cex =2.5,text.col = "white",box.col=NA,y.intersp = 1.6)
legend(-8,0.62,legend=c("Arithmetic mean","With prior cases beyond focal village","Without prior cases beyond focal village"),col=c(1,"darkorange2","navy"),lwd=c(1,2,2),lty=c(2,1,1),cex=cex.axis,pt.cex =1,bty="n",y.intersp = 1.6)

dev.off()



# Plot model predictions at different susceptibilities and case rates
#___________________

pdf("Figs/MonthlyModelCases&PowerMeanSuscVill_supplement.pdf",width=7, height=5.5)

data_vill_case_rate_adjust_dist <- 0.5*min(data_vill$case_rate_last2monthMean_dist[which(data_vill$case_rate_last2monthMean_dist>0)])
par(mar=c(2,4.5,1,0.35))
par(fig=c(0,0.5,0.5,1))
range(data_vill$case_rate_last2monthMean_dist,na.rm = T)
case_rate <- seq(0,0.001,length.out=5)
susceptibility <- seq(0,1,length.out=50)
cols=viridis(length(case_rate))
samples_pars <- posterior_samples(model, pars = c("Intercept","beta","p","phi"))
samples_pars <- samples_pars[,-which(colnames(samples_pars)=="beta[1]")]
preds_mat <- matrix(NA,nrow=(length(susceptibility)*length(case_rate)),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       "susc_last2monthMean"=rep(susceptibility,length(case_rate)),
                       "log_case_rate_last2monthMean"=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(susceptibility)),
                       "log_case_rate_neighbours_last2monthMean"=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(susceptibility)),
                       "log_case_rate_notNeighbours_last2monthMean"=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(susceptibility)),
                       "log_dog_density"=log(mean(data_vill$dog_density)),
                       "HDR"=mean(data_vill$HDR), 
                       "power_mean_neighbours_last2MonthMean"=rep(susceptibility,length(case_rate)),
                       "power_mean_notNeighbours_last2MonthMean"=rep(susceptibility,length(case_rate)))))
for(i in 1:nrow(samples_pars)){
  preds_mat[,i] <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:nrow(X))]))))*1000
}
preds_lower <- matrix(apply(preds_mat,1,quantile,0.025,na.rm=T),nrow=length(susceptibility),ncol=length(case_rate))
preds_upper <- matrix(apply(preds_mat,1,quantile,0.975,na.rm=T),nrow=length(susceptibility),ncol=length(case_rate))
preds_mat <- matrix(apply(preds_mat, 1, mean),nrow=length(susceptibility),ncol=length(case_rate))
col_pal <- viridis(length(case_rate))
ylim=c(0,max(preds_upper))
plot(NA,ylim=ylim,xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("with prior cases\nbeyond focal village",side=2,line=1.3,cex=cex.lab)
# mtext("Susceptibility at all scales\nover prior two months",side=1,line=2,cex=cex.lab)
for(i in 1:length(case_rate)){
  polygon(c(susceptibility,rev(susceptibility)),c(preds_lower[,i],rev(preds_upper[,i])),col=scales::alpha(cols[i],0.25),border=NA)
}
for(i in 1:length(case_rate)){
  lines((preds_mat[,i])~susceptibility,col=cols[i],lwd=2,lty=1)
}
graphics::legend(0.07,max((preds_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[1:3],pch=15,
                 col=scales::alpha(cols,0.25)[1:3],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all\n scales over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.345,max((preds_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[4:5],pch=15,
                 col=scales::alpha(cols,0.25)[c(4:5)],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all\n scales over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.21,max((preds_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T)),
                 col=cols,y.intersp = 1.42,lty=1,lwd=2,title.col = "white",
                 title="Mean cases/1,000 dogs at all\n scales over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=2)
graphics::legend(0.12,max((preds_upper))*1.05,legend="Mean cases/1,000 dogs at all\n scales over prior 2 months:",bty="n",cex=0.75)
graphics::legend(-0.09,1.9,legend="A",text.font = 2,bty="n")

par(fig=c(0,0.5,0,1),new=T)
text(-0.38,1,"Cases/1,000 dogs in village from model:",srt=90,xpd=T,cex=cex.lab)


par(fig=c(0,0.5,0,0.5),new=T)
par(mar=c(3,4.5,0,0.35))
samples_pars <- posterior_samples(model_wo_distant_cases, pars = c("Intercept","beta","p","phi"))
samples_pars <- samples_pars[,-which(colnames(samples_pars)=="beta[1]")]
preds_mat <- matrix(NA,nrow=(length(susceptibility)*length(case_rate)),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       "susc_last2monthMean"=rep(susceptibility,length(case_rate)),
                       "log_case_rate_last2monthMean"=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(susceptibility)),
                       "log_dog_density"=log(mean(data_vill$dog_density)),
                       "HDR"=mean(data_vill$HDR), 
                       "power_mean_neighbours_last2MonthMean"=rep(susceptibility,length(case_rate)),
                       "power_mean_notNeighbours_last2MonthMean"=rep(susceptibility,length(case_rate)))))
for(i in 1:nrow(samples_pars)){
  preds_mat[,i] <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:nrow(X))]))))*1000
}
preds_lower <- matrix(apply(preds_mat,1,quantile,0.025,na.rm=T),nrow=length(susceptibility),ncol=length(case_rate))
preds_upper <- matrix(apply(preds_mat,1,quantile,0.975,na.rm=T),nrow=length(susceptibility),ncol=length(case_rate))
preds_mat <- matrix(apply(preds_mat, 1, mean),nrow=length(susceptibility),ncol=length(case_rate))
col_pal <- viridis(length(case_rate))
plot(NA,ylim=ylim,xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("without prior cases\nbeyond focal village",side=2,line=1.3,cex=cex.lab)
mtext("Susceptibility at all scales\nover prior two months",side=1,line=2,cex=cex.lab)
for(i in 1:length(case_rate)){
  polygon(c(susceptibility,rev(susceptibility)),c(preds_lower[,i],rev(preds_upper[,i])),col=scales::alpha(cols[i],0.25),border=NA)
}
for(i in 1:length(case_rate)){
  lines((preds_mat[,i])~susceptibility,col=cols[i],lwd=2,lty=1)
}
# graphics::legend(0.23,max((preds_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T)),lty=1,col=cols,
#                  title="Mean cases/1,000 dogs in village\n  over prior 2 months:",
#                  title.adj =-0.1,title.cex=0.7,cex=0.7,bty="n",ncol=2,lwd=2)
graphics::legend(-0.07,1.9,legend="B",text.font = 2,bty="n")



# Plot fit to data
#___________________

# Plot predictions and data over time
load("Output/preds_PI_stan_village_model.Rdata")
par(mar=c(1.8,2.5,0.7,1))
par(fig=c(0.5,1,0.5,1),new=T)
ylim <- c(0,max(preds_upper,data_vill$cases))
plot(NA,ylim=ylim,xlim=c(1,max(data_vill$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1,at=seq(0,100,50))
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
# mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
# mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_vill$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col=scales::alpha("dodgerblue",0.4),border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases~data_dist$month,col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend(130,120,c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c("skyblue","navy","red"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #4%
legend(-20,115,legend="C",text.font = 2,bty="n")
par(fig=c(0.5,1,0,0.57),new=T)
text(-45,60,"Dog cases in district",srt=90,cex=cex.lab,xpd=T)


# Plot predictions and data over time
load("Output/preds_PI_stan_village_model_wo_distant_cases.Rdata")
par(mar=c(2.5,2.5,0,1))
par(fig=c(0.5,1,0,0.5),new=T)
# ylim <- c(0,max(preds_upper,data_vill$cases))
plot(NA,ylim=ylim,xlim=c(1,max(data_vill$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1,at=seq(0,100,50))
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
# mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_vill$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col=scales::alpha("orange",0.3),border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases~data_dist$month,col="darkorange",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red2",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend(130,120,c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c(scales::alpha("orange",0.5),"darkorange","red2"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #4%
legend(-20,115,legend="D",text.font = 2,bty="n")

dev.off()



# Difference between power mean and arithmetic mean
#___________________

diff_arithmetic <- sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_mean,wts = dogs[,x]))-data_dist$arithmeticMean
diff_arithmetic_upper <- sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_upper,wts = dogs[,x]))-data_dist$arithmeticMean
diff_arithmetic_lower <- sapply(1:ncol(S),function(x) powMean(x=S[,x],p=p_lower,wts = dogs[,x]))-data_dist$arithmeticMean
diff_arithmetic_annual <- diff_arithmetic_lower_annual <- diff_arithmetic_upper_annual <- rep(NA,length(2002:2022))
for(i in 1:length(2002:2022)){
  diff_arithmetic_annual[i] <- mean(diff_arithmetic[(1:12) + (i-1)*12])
  diff_arithmetic_upper_annual[i] <- mean(diff_arithmetic_upper[(1:12) + (i-1)*12])
  diff_arithmetic_lower_annual[i] <- mean(diff_arithmetic_lower[(1:12) + (i-1)*12])
}

pdf("Figs/diff_arithmetic_power_means.pdf",width=7, height=3.2)

par(mar=c(2.5,3,1,1))
par(fig=c(0,0.5,0,1))
plot(diff_arithmetic,ylim=c(0,0.2),type="l",lty=2,col="grey",axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Power mean - arithmetic mean susceptibility",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 1:max(data_vill$month)
polygon(c(months_plot,rev(months_plot)),c(diff_arithmetic_lower,rev(diff_arithmetic_upper)),col="skyblue",border=NA)
lines(diff_arithmetic,col="navy",lwd=2)

par(fig=c(0.5,1,0,1),new=T)
plot(diff_arithmetic_annual,ylim=c(0,0.2),type="l",lty=2,col="grey",axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,at=seq(1,length(diff_arithmetic_annual),2),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Power mean - arithmetic mean susceptibility\n(12 month mean)",side=2,line=1.5,cex=cex.lab)
mtext("Year",side=1,line=1.5,cex=cex.lab)
months_plot <- 1:length(diff_arithmetic_annual)
polygon(c(months_plot,rev(months_plot)),c(diff_arithmetic_lower_annual,rev(diff_arithmetic_upper_annual)),col="skyblue",border=NA)
lines(diff_arithmetic_annual,col="navy",lwd=2)

dev.off()

names(diff_arithmetic_annual) <- 2002:2022
sort(diff_arithmetic_annual)

# save(diff_arithmetic,diff_arithmetic_lower,diff_arithmetic_upper,
#      diff_arithmetic_annual,diff_arithmetic_lower_annual,diff_arithmetic_upper_annual,
#      file="Output/difference_arithmetic_mean.Rdata")



# Random effects
#___________________

par(mar=c(0,1,1.5,4))
ranefs <- colMeans(posterior_samples(model, pars = c("gamma_t")))
range(ranefs)
breaks=c(seq(min(ranefs),0,length.out=51),
         seq(0,max(ranefs),length.out=50)[-1])
colours=colorRampPalette(c("dodgerblue","white","red"))(length(breaks)-1)
plot(SD_vill,col=colours[findInterval(ranefs,breaks,all.inside=T)],
     cex.main=0.8,lwd=0.5,border="grey")
plot(SD_outline,add=T)
grid <- raster(extent(SD_vill),crs=SD_vill@proj4string);res(grid) <- 1000;grid[]<-1
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=colours,
     legend.args=list(text="exp(Village random effect)", side=4, line=2, cex=cex.lab),
     axis.args=list(at=log(c(0.25,0.5,1,2,4)),labels=c(0.25,0.50,1.00,2.00,4.00),cex.axis=cex.axis,hadj=0.5),
     smallplot=c(0.7,0.72, .25,.75))

par(mar=c(0,1,1.5,4))
ranefs <- colMeans(posterior_samples(model_wo_distant_cases, pars = c("gamma_t")))
breaks=c(seq(min(ranefs),0,length.out=51),
         seq(0,max(ranefs),length.out=50)[-1])
colours=colorRampPalette(c("dodgerblue","white","red"))(length(breaks)-1)
plot(SD_vill,col=colours[findInterval(ranefs,breaks,all.inside=T)],
     cex.main=0.8,lwd=0.5,border="grey")
plot(SD_outline,add=T)
grid <- raster(extent(SD_vill),crs=SD_vill@proj4string);res(grid) <- 1000;grid[]<-1
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=colours,
     legend.args=list(text="exp(Village random effect)", side=4, line=2, cex=cex.lab),
     axis.args=list(at=log(c(0.25,0.5,1,2,4)),labels=c(0.25,0.50,1.00,2.00,4.00),cex.axis=cex.axis,hadj=0.5),
     smallplot=c(0.7,0.72, .25,.75))



