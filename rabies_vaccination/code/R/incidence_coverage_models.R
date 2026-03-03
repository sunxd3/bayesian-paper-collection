
rm(list=ls())

library(MASS)
library(DHARMa)
library(raster)
library(viridis)
library(RColorBrewer)
library(brms)
library(loo)
library(rgeos)
library(rgdal)
library(stringr)
library(scales)



## Load data
#______________________

# Model data
data_vill <- read.csv("output/incidence_coverage_model_data_village.csv")
data_dist <- read.csv("output/incidence_coverage_model_data_district.csv")

# Dog population
dogs <- as.matrix(read.csv("output/dogPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))

# Human population
humans <- as.matrix(read.csv("output/humanPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))

# Vaccination estimates
vax_vill <- as.matrix(read.csv("output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) # village level
immune_vill <- as.matrix(read.csv("output/immunityByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) # village level
vax_dist <- as.matrix(read.csv("output/districtVaccinationCoverage_Jan2002_Dec2022.csv",header = T)) # district level
immune_dist <- vax_dist[,2]
vax_dist <- vax_dist[,1]
vax_year <- read.csv("output/vcDistByYear.csv",header = F)[-1,1] #discard 2002 (no campaign)
mean(vax_year)
max(vax_year)

# Case numbers
cases_dist <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_2002-01-01_2022-12-31.csv",header=F))
cases_vill <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_village_2002-01-01_2022-12-31.csv",header=F,row.names = 1))

# Rabies cases
ct_data <- read.csv("output/serengeti_rabid_dogs.csv")

# Human bites per dog info
load("output/human_bites_per_dog_outputs.RData")

# Human bites by dogs by month
monthlyBites <- read.csv("output/Serengeti_monthly_human_exposures_dogs_2002-01-01_2022-12-31.csv",header=F)$V1

# Serengeti village shapefile
SD_vill <- readOGR("output/SD_vill","SD_vill") 
SD_vill<-SD_vill[order(SD_vill$Vill_2012),]
SD_outline <- gUnaryUnion(SD_vill)# get district outline
SD_outline<-gBuffer(SD_outline,width=1) # get rid of a few tiny holes
SD_vill_gridded <- readOGR("data/GIS","SD_Villages_2012_From_HHS_250m_Lattice_UTM") 

# Run models or just load from memory for plotting?
run_models<-F

# Protected areas
PAs <- readOGR("data/GIS/ProtectedAreas","TZprotected_areas", p4s = ("+proj=utm +zone=37 +south +ellps=clrk80 +towgs84=-160,-6,-302,0,0,0,0 +units=m +no_defs"))
PAs <- spTransform(PAs,SD_outline@proj4string)
Serengeti <- PAs[which(PAs$SP_ID=="2"),]
Serengeti <- Serengeti - SD_vill_gridded

# Plot functions
source("http://www.math.mcmaster.ca/bolker/R/misc/legendx.R")

# Some summaries for MS
dog_densities <- dogs/(gArea(SD_vill,byid = T)/1e6)
range(dog_densities[,ncol(dogs)])
length(which(dog_densities[,ncol(dogs)]>50))
length(which(dog_densities[,ncol(dogs)]>100))
dog_densities2 <- dogs/SD_vill$cells_occ
range(dog_densities2[,ncol(dogs)])
length(which(dog_densities2[,ncol(dogs)]>50))
length(which(dog_densities2[,ncol(dogs)]>100))
length(which(dog_densities2[,ncol(dogs)]>100))/nrow(SD_vill)



# District monthly model
#______________________

# Fit models
#--------------

prior1 <- c(set_prior("normal(0,100000)", class = "Intercept"),
            set_prior("normal(0,100000)", class = "b"))


if(run_models==T){
  
  data_dist2<-data_dist
  data_dist2[,c("vax_last2monthMean","log_case_rate_last2monthMean","log_dog_density")]<-
    scale(data_dist2[,c("vax_last2monthMean","log_case_rate_last2monthMean","log_dog_density")])
  
  if(!dir.exists("output/brms_models")){dir.create("output/brms_models")}
  
  model_dist_month_brms_unlogged <- brm(formula = cases~vax_last2monthMean + case_rate_last2monthMean + log_dog_density + offset(log(dogs)),
                                        data = data_dist[-c(1:2),], family = negbinomial(), 
                                        warmup = 1500, iter = 3000, chains = 4, prior = prior1,
                                        control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  model_dist_month_brms_unlogged <- add_criterion(model_dist_month_brms_unlogged, c("waic","loo"))
  saveRDS(model_dist_month_brms_unlogged,"output/brms_models/incidence_from_vax_model_district_unlogged.rds")
  
  model_dist_month_brms <- brm(formula = cases~vax_last2monthMean + log_case_rate_last2monthMean + log_dog_density + offset(log(dogs)),
                               data = data_dist[-c(1:2),], family = negbinomial(), 
                               warmup = 1500, iter = 3000, chains = 4, prior = prior1,
                               control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  model_dist_month_brms <- add_criterion(model_dist_month_brms, c("waic","loo"))
  saveRDS(model_dist_month_brms,"output/brms_models/incidence_from_vax_model_district.rds")

  # model_dist_month_brms <- brm(formula = cases~vax_last2monthMean*log_case_rate_last2monthMean + log_dog_density + offset(log(dogs)),
  #                              data = data_dist[-c(1:2),], family = negbinomial(), 
  #                              warmup = 1500, iter = 3000, chains = 4, prior = prior1,
  #                              control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  # model_dist_month_brms <- add_criterion(model_dist_month_brms, c("waic","loo"))
  
  
  model_dist_month_brms_standardise <- brm(formula = cases~vax_last2monthMean + log_case_rate_last2monthMean + log_dog_density + offset(log(dogs)),
                                           data = data_dist2[-c(1:2),], family = negbinomial(), 
                                           warmup = 1500, iter = 3000, chains = 4, prior = prior1,
                                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  saveRDS(model_dist_month_brms_standardise,"output/brms_models/incidence_from_vax_model_district_standardise.rds")
  
  model_dist_month_brms_woPriorCases <- brm(formula = cases~vax_last2monthMean + log_dog_density + offset(log(dogs)),
                                            data = data_dist[-c(1:2),], family = negbinomial(), 
                                            warmup = 1500, iter = 3000, chains = 4, prior = prior1,
                                            control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  model_dist_month_brms_woPriorCases <- add_criterion(model_dist_month_brms_woPriorCases, c("waic","loo"))
  saveRDS(model_dist_month_brms_woPriorCases,"output/brms_models/incidence_from_vax_model_district_woPriorCases.rds")

  model_dist_month_brms_woPriorCases_standardise <- brm(formula = cases~vax_last2monthMean + log_dog_density + offset(log(dogs)),
                                            data = data_dist2[-c(1:2),], family = negbinomial(), 
                                            warmup = 1500, iter = 3000, chains = 4, prior = prior1,
                                            control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  model_dist_month_brms_woPriorCases_standardise <- add_criterion(model_dist_month_brms_woPriorCases_standardise, c("waic","loo"))
  saveRDS(model_dist_month_brms_woPriorCases_standardise,"output/brms_models/incidence_from_vax_model_district_woPriorCases_standardise.rds")
  
}

model_dist_month_brms_unlogged <- readRDS("output/brms_models/incidence_from_vax_model_district_unlogged.rds")
model_dist_month_brms <- readRDS("output/brms_models/incidence_from_vax_model_district.rds")
model_dist_month_brms_standardise <- readRDS("output/brms_models/incidence_from_vax_model_district_standardise.rds")
model_dist_month_brms_woPriorCases <- readRDS("output/brms_models/incidence_from_vax_model_district_woPriorCases.rds")
model_dist_month_brms_woPriorCases_standardise <- readRDS("output/brms_models/incidence_from_vax_model_district_woPriorCases_standardise.rds")
model_dist_month_brms_unlogged$criteria$waic
model_dist_month_brms$criteria$waic # logged better
model_dist_month_brms_woPriorCases$criteria$waic



# Model diagnostics
#--------------

plot(model_dist_month_brms,ask=F)
head(predict(model_dist_month_brms))
model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_month_brms)),
  observedResponse = data_dist$cases[-c(1:2)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_month_brms)), 1, median),
  integerResponse = TRUE)
pdf("Figs/model_dist_month_brms_diagnostics.pdf",width=10, height=10)
par(mfrow=c(3,3))
plotQQunif(model.check,pch=4,col="navy")
testZeroInflation(model.check)
plotResiduals(model.check,quantreg=T,pch=4,col="navy")
# testDispersion(model.check)
plotResiduals(model.check,data_dist$case_rate_last2monthMean[3:nrow(data_dist)],quantreg = T, xlab="Mean cases/dog over previous 2 months",pch=4,col="navy") 
plotResiduals(model.check,data_dist$vax_dist_last2monthMean[3:nrow(data_dist)],quantreg = T,xlab="Mean vaccination over previous 2 months",pch=4,col="navy") 
plotResiduals(model.check,data_dist$human_density[3:nrow(data_dist)],quantreg = T,xlab="Humans/km2",pch=4,col="navy") 
plotResiduals(model.check,data_dist$dog_density[3:nrow(data_dist)],quantreg = T,xlab="Dogs/km2",pch=4,col="navy") 
# pp_check(model_dist_month_brms, ndraws = 300) 
dev.off()
# qqplot fine, quantile plots could be better, but not bad

plot(model_dist_month_brms_woPriorCases,ask=F)
head(predict(model_dist_month_brms_woPriorCases))
model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_month_brms_woPriorCases)),
  observedResponse = data_dist$cases[-c(1:2)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_month_brms_woPriorCases)), 1, median),
  integerResponse = TRUE)
pdf("Figs/model_dist_month_brms_diagnostics_woPriorCases.pdf",width=10, height=10)
par(mfrow=c(3,3))
plotQQunif(model.check,pch=4,col="navy")
testZeroInflation(model.check)
plotResiduals(model.check,quantreg=T,pch=4,col="navy")
# testDispersion(model.check)
plotResiduals(model.check,data_dist$case_rate_last2monthMean[3:nrow(data_dist)],quantreg = T, xlab="Mean cases/dog over previous 2 months",pch=4,col="navy") 
plotResiduals(model.check,data_dist$vax_dist_last2monthMean[3:nrow(data_dist)],quantreg = T,xlab="Mean vaccination over previous 2 months",pch=4,col="navy") 
plotResiduals(model.check,data_dist$human_density[3:nrow(data_dist)],quantreg = T,xlab="Humans/km2",pch=4,col="navy") 
plotResiduals(model.check,data_dist$dog_density[3:nrow(data_dist)],quantreg = T,xlab="Dogs/km2",pch=4,col="navy") 
# pp_check(model_dist_month_brms, ndraws = 300) 
dev.off()
# not good




#Explore coefficients
#--------------

# parameters and standardised parameters
model_summary <- summary(model_dist_month_brms)
pars_district <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,1:4]
rownames(pars_district)[2:nrow(pars_district)] <- c("Vaccination over last 2 months","Log cases/dog over last 2 months","Log dogs/km2","shape")
model_summary <- summary(model_dist_month_brms_standardise)
pars_district_standardise <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,1:4]
rownames(pars_district_standardise)[2:nrow(pars_district_standardise)] <- c("Vaccination over last 2 months","Log cases/dog over last 2 months","Log dogs/km2","shape")

model_summary <- summary(model_dist_month_brms_woPriorCases)
pars_district_woPriorCases <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,1:4]
rownames(pars_district_woPriorCases)[2:nrow(pars_district_woPriorCases)] <- c("Vaccination over last 2 months","Log dogs/km2","shape")
model_summary <- summary(model_dist_month_brms_woPriorCases_standardise)
pars_district_woPriorCases_standardise <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,1:4]
rownames(pars_district_woPriorCases_standardise)[2:nrow(pars_district_woPriorCases_standardise)] <- c("Vaccination over last 2 months","Log dogs/km2","shape")

# Prep for MS table
pars_district_exp <- round(exp(summary(model_dist_month_brms)$fixed),2)[,c(1,3,4)]
pars_district_prep <- data.frame(matrix(NA, nrow=nrow(pars_district), ncol=2,dimnames = list(rownames(pars_district),c("Estimate","exp(Estimate)"))))
pars_district_prep[,1] <- paste0(pars_district$Estimate," (",pars_district$`l-95% CI`,", ",pars_district$`u-95% CI`,")")
pars_district_prep[1:nrow(pars_district_exp),2] <- paste0(pars_district_exp$Estimate," (",pars_district_exp$`l-95% CI`,", ",pars_district_exp$`u-95% CI`,")")
write.csv(pars_district_prep,"output/model_dist_month_brms_pars_MSversion.csv")
pars_district_woPriorCases_exp <- round(exp(summary(model_dist_month_brms_woPriorCases)$fixed),2)[,c(1,3,4)]
pars_district_woPriorCases_prep <- data.frame(matrix(NA, nrow=nrow(pars_district_woPriorCases), ncol=2,dimnames = list(rownames(pars_district_woPriorCases),c("Estimate","exp(Estimate)"))))
pars_district_woPriorCases_prep[,1] <- paste0(pars_district_woPriorCases$Estimate," (",pars_district_woPriorCases$`l-95% CI`,", ",pars_district_woPriorCases$`u-95% CI`,")")
pars_district_woPriorCases_prep[1:nrow(pars_district_woPriorCases_exp),2] <- paste0(pars_district_woPriorCases_exp$Estimate," (",pars_district_woPriorCases_exp$`l-95% CI`,", ",pars_district_woPriorCases_exp$`u-95% CI`,")")
write.csv(pars_district_woPriorCases_prep,"output/model_dist_month_brms_pars_woPriorCases_MSversion.csv")


# some summaries for the manuscript
(1-exp(pars_district["Vaccination over last 2 months",c(1,3,4)]*0.1))*100 # % change with 10% change in village vaccination
100*((exp(pars_district["Log cases/dog over last 2 months",c(1,3,4)]*log(2)))/(exp(pars_district["Log cases/dog over last 2 months",c(1,3,4)]*log(1)))-1) 
100*(1-(exp(pars_district["Log dogs/km2",c(1,3,4)]*log(2)))/(exp(pars_district["Log dogs/km2",c(1,3,4)]*log(1)))) 



# Plot model
#--------------

# Plot predictions (lines,projected)
pdf("Figs/MonthlyModelCases&VaxDist.pdf",width=7, height=5)
cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5
par(mar=c(2,2.5,1,1))
par(fig=c(0,0.85,0.5,1))
coverage <- seq(0,1,length.out=100)
case_rate <- seq(0,0.001,0.00025)
breaks = seq(0,max(data_dist$case_rate_last2monthMean,na.rm=T),length.out=100)*1000
cols=viridis(100)
data_dist_case_rate_adjust <-  0.5*min(data_dist$case_rate_last2monthMean[which(data_dist$case_rate_last2monthMean>0)])
new_data_extrap <- data.frame(dogs=1,
                              vax_last2monthMean=rep(coverage,length(case_rate)), 
                              log_case_rate_last2monthMean=rep(log(case_rate + data_dist_case_rate_adjust),each=length(coverage)),
                              log_dog_density = log(mean(data_dist$dog_density))) 
preds_mat_extrap <- posterior_epred(model_dist_month_brms, newdata = new_data_extrap, re_formula = NA)*1000
preds_mat_extrap_upper <- matrix(apply(preds_mat_extrap, 2, quantile,probs=0.975),nrow=length(coverage),ncol=length(case_rate))
preds_mat_extrap_lower <- matrix(apply(preds_mat_extrap, 2, quantile,probs=0.025),nrow=length(coverage),ncol=length(case_rate))
preds_mat_extrap <- matrix(apply(preds_mat_extrap, 2, mean),nrow=length(coverage),ncol=length(case_rate))
col_pal <- viridis(length(case_rate))
plot(NA,ylim=c(0,max(preds_mat_extrap_upper,data_dist$incidence)),xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases per 1,000 dogs in district",side=2,line=1.5,cex=cex.lab)
mtext("Mean rolling vaccination over previous two months",side=1,line=1.5,cex=cex.lab)
for(i in 1:length(case_rate)){
  polygon(c(coverage,rev(coverage)),c(preds_mat_extrap_lower[,i],rev(preds_mat_extrap_upper[,i])),col=scales::alpha(cols[as.numeric(findInterval(case_rate[i]*1000,breaks))],0.25),border=NA)
}
for(i in 1:length(case_rate)){
  lines(preds_mat_extrap[,i]~coverage,col=cols[as.numeric(findInterval(case_rate[i]*1000,breaks))],lwd=2)
}
points((data_dist$incidence*1000)~data_dist$vax_last2monthMean,
       col=cols[as.numeric(findInterval(data_dist$case_rate_last2monthMean*1000,breaks))],
       pch=20,cex=cex.pt)
graphics::legend("topleft",legend="A",text.font = 2,bty="n")
grid <- raster(nrows=10, ncols=10);grid[]<-0.0001
plot(grid,
     breaks=breaks,legend.only=T, add=T,col=cols,
     legend.args=list(text="Mean cases per 1,000 dogs over\nprevious 2 months", side=4, line=3.2, cex=0.8),
     axis.args=list(at=pretty(c(0,data_dist$case_rate_last2monthMean*1000)),hadj=0.3,cex.axis=0.75),
     smallplot=c(0.98,0.99, .2,.85))


# Plot predictions and data over time
par(mar=c(2.5,2.5,2,1))
par(fig=c(0.5,1,0,0.5),new=T)
preds_mat <- predict(model_dist_month_brms, re_formula=NULL, ndraws=5000, summary=F) 
preds_mat_dist <- rowsum(t(preds_mat),data_dist$month[-c(1:2)])
preds_lower <- apply(preds_mat_dist,1,quantile,0.025)
preds_upper <- apply(preds_mat_dist,1,quantile,0.975)
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
points(data_dist$cases[months_plot]~data_dist$month[months_plot],col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend("topright",c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c("skyblue","navy","red"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #9%
legend("topleft",legend="C",text.font = 2,bty="n")

# standardised coefficients
par(fig=c(0,0.5,0,0.5),new=T)
par(mar=c(2.5,6,2,1))
pars_sub <- pars_district_standardise[2:(nrow(pars_district_standardise)-1),]
pars_sub_woPriorCases <- pars_district_woPriorCases_standardise[2:(nrow(pars_district_woPriorCases_standardise)-1),]
xlim=c((min(pars_sub$`l-95% CI`,pars_sub_woPriorCases$`l-95% CI`)),(max(pars_sub$`u-95% CI`,pars_sub_woPriorCases$`u-95% CI`)))
plot(NA,xlim=xlim,ylim=c(0,nrow(pars_sub)+1),axes=F,ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub)), function(x) paste(strwrap(x, 20), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,1),las=2)
at=log(pretty(c(exp(c(pars_sub$`l-95% CI`,pars_sub_woPriorCases$`l-95% CI`,pars_sub$`u-95% CI`,pars_sub_woPriorCases$`u-95% CI`)))))
label=pretty(exp(c(pars_sub$`l-95% CI`,pars_sub_woPriorCases$`l-95% CI`,pars_sub$`u-95% CI`,pars_sub_woPriorCases$`u-95% CI`)))
axis(1,cex.axis=cex.axis,padj=-1.5,at=at,label=label)
lines(c(0,0),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.5,cex=cex.lab)
box(bty="l")
arrows(x0=(pars_sub$`l-95% CI`),x1=(pars_sub$`u-95% CI`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col="navy")
points((pars_sub$Estimate),seq(nrow(pars_sub)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
arrows(x0=(pars_sub_woPriorCases$`l-95% CI`),x1=(pars_sub_woPriorCases$`u-95% CI`),y0=seq(nrow(pars_sub)-0.7,0.3,-1)[c(1,3)],y1=seq(nrow(pars_sub)-0.7,0.3,-1)[c(1,3)],length=0,lwd=2,col="lightblue")
points((pars_sub_woPriorCases$Estimate),seq(nrow(pars_sub)-0.7,0.3,-1)[c(1,3)],col="pink",pch=20,cex=1.3)
legend("topleft",legend="B",text.font = 2,bty="n")
legend("topright",legend=c("With prior cases","Without prior cases"),col=c("navy","lightblue"),lwd=2,cex=0.6,pt.cex =1,text.col = "white")
legend("topright",legend=c("With prior cases","Without prior cases"),col=c("red","pink"),lwd=2,cex=0.6,pt.cex =1,lty=0,pch=20,bty="n",)

dev.off()




# Village monthly model
#______________________

# Fit models
#--------------

prior2 <- c(set_prior("normal(0,100000)", class = "Intercept"),
            set_prior("normal(0,100000)", class = "b"),
            set_prior("exponential(0.001)", class = "sd"))


data_vill2<-data_vill
data_vill2[,c("vax_last2monthMean","vax_neighbours_last2monthMean","vax_notNeighbours_last2monthMean","vax_last2monthMean_dist",
              "immune_last2monthMean","immune_neighbours_last2monthMean","immune_notNeighbours_last2monthMean","immune_last2monthMean_dist",
              "log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean","log_case_rate_last2monthMean_dist",
              "log_human_density","log_dog_density","HDR")]<-
  scale(data_vill2[,c("vax_last2monthMean","vax_neighbours_last2monthMean","vax_notNeighbours_last2monthMean","vax_last2monthMean_dist",
                      "immune_last2monthMean","immune_neighbours_last2monthMean","immune_notNeighbours_last2monthMean","immune_last2monthMean_dist",
                      "log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean","log_case_rate_notNeighbours_last2monthMean","log_case_rate_last2monthMean_dist",
                      "log_human_density","log_dog_density","HDR")])
if(run_models==T){
  model_vill_month_brms_unlogged <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                          case_rate_last2monthMean + case_rate_neighbours_last2monthMean + case_rate_notNeighbours_last2monthMean +
                                          log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                        data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                        warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                                        control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4)
  model_vill_month_brms_unlogged <- add_criterion(model_vill_month_brms_unlogged, c("waic"))
  saveRDS(model_vill_month_brms_unlogged,"output/brms_models/incidence_from_vax_model_village_unlogged.rds")
  
  model_vill_month_brms <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                 log_case_rate_last2monthMean + log_case_rate_neighbours_last2monthMean + log_case_rate_notNeighbours_last2monthMean + 
                                 log_dog_density + HDR + (1|village) + offset(log(dogs)),
                               data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                               warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                               control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_vill_month_brms <- add_criterion(model_vill_month_brms, c("waic"))
  saveRDS(model_vill_month_brms,"output/brms_models/incidence_from_vax_model_village.rds")
  
  # model_vill_month_brms <- brm(formula = cases~vax_last2monthMean*log_case_rate_last2monthMean + 
  #                                vax_neighbours_last2monthMean*log_case_rate_neighbours_last2monthMean + 
  #                                vax_notNeighbours_last2monthMean*log_case_rate_notNeighbours_last2monthMean +
  #                                log_dog_density + HDR + (1|village) + offset(log(dogs)),
  #                              data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
  #                              warmup = 1500, iter = 3000, chains = 4, prior = prior2,
  #                              control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  # model_vill_month_brms <- add_criterion(model_vill_month_brms, c("waic"))
  # 
  
  model_vill_month_brms_standardise <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                             log_case_rate_last2monthMean + log_case_rate_neighbours_last2monthMean + log_case_rate_notNeighbours_last2monthMean + 
                                             log_dog_density + HDR + (1|village) + offset(log(dogs)),
                               data = data_vill2[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                               warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                               control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  saveRDS(model_vill_month_brms_standardise,"output/brms_models/incidence_from_vax_model_village_standardise.rds")
  
  model_vill_month_brms_woDistantPriorCases <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                                     log_case_rate_last2monthMean + log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                            data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                            warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                                            control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_vill_month_brms_woDistantPriorCases <- add_criterion(model_vill_month_brms_woDistantPriorCases, c("waic"))
  saveRDS(model_vill_month_brms_woDistantPriorCases,"output/brms_models/incidence_from_vax_model_village_woDistantPriorCases.rds")
  
  model_vill_month_brms_standardise_woDistantPriorCases <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                                          log_case_rate_last2monthMean + log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                                        data = data_vill2[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                                        warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                                                        control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  saveRDS(model_vill_month_brms_standardise_woDistantPriorCases,"output/brms_models/incidence_from_vax_model_village_woDistantPriorCases_standardise.rds")
  
  model_vill_month_brms_woPriorCases <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                                      log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                                   data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                                   warmup = 3000, iter = 6000, chains = 4, prior = prior2,
                                                   control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_vill_month_brms_woPriorCases <- add_criterion(model_vill_month_brms_woPriorCases, c("waic"))
  saveRDS(model_vill_month_brms_woPriorCases,"output/brms_models/incidence_from_vax_model_village_woPriorCases.rds")

  model_vill_month_brms_standardise_woPriorCases <- brm(formula = cases~vax_last2monthMean + vax_neighbours_last2monthMean + vax_notNeighbours_last2monthMean +
                                              log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                            data = data_vill2[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                            warmup = 3000, iter = 6000, chains = 4, prior = prior2,
                                            control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_vill_month_brms_standardise_woPriorCases <- add_criterion(model_vill_month_brms_standardise_woPriorCases, c("waic"))
  saveRDS(model_vill_month_brms_standardise_woPriorCases,"output/brms_models/incidence_from_vax_model_village_woPriorCases_standardise.rds")
  
  model_immunity_vill_month_brms_unlogged <- brm(formula = cases~immune_last2monthMean + immune_neighbours_last2monthMean + immune_notNeighbours_last2monthMean +
                                                   case_rate_last2monthMean + case_rate_neighbours_last2monthMean + case_rate_notNeighbours_last2monthMean + 
                                                   log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                                 data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                                 warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                                                 control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_immunity_vill_month_brms_unlogged <- add_criterion(model_immunity_vill_month_brms_unlogged, c("waic"))
  saveRDS(model_immunity_vill_month_brms_unlogged,"output/brms_models/incidence_from_immunity_model_village_unlogged.rds")
  
  model_immunity_vill_month_brms <- brm(formula = cases~immune_last2monthMean + immune_neighbours_last2monthMean + immune_notNeighbours_last2monthMean +
                                          log_case_rate_last2monthMean + log_case_rate_neighbours_last2monthMean + log_case_rate_notNeighbours_last2monthMean + 
                                          log_dog_density + HDR + (1|village) + offset(log(dogs)),
                                        data = data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(), 
                                        warmup = 1500, iter = 3000, chains = 4, prior = prior2,
                                        control = list(adapt_delta = 0.95,max_treedepth = 15), cores=4)
  model_immunity_vill_month_brms <- add_criterion(model_immunity_vill_month_brms, c("waic"))
  saveRDS(model_immunity_vill_month_brms,"output/brms_models/incidence_from_immunity_model_village.rds")
}
model_vill_month_brms_unlogged<-readRDS("output/brms_models/incidence_from_vax_model_village_unlogged.rds")
model_vill_month_brms <- readRDS("output/brms_models/incidence_from_vax_model_village.rds")
model_vill_month_brms_standardise <- readRDS("output/brms_models/incidence_from_vax_model_village_standardise.rds")
model_vill_month_brms_woDistantPriorCases <- readRDS("output/brms_models/incidence_from_vax_model_village_woDistantPriorCases.rds")
model_vill_month_brms_standardise_woDistantPriorCases <- readRDS("output/brms_models/incidence_from_vax_model_village_woDistantPriorCases_standardise.rds")
model_vill_month_brms_woPriorCases <- readRDS("output/brms_models/incidence_from_vax_model_village_woPriorCases.rds")
model_vill_month_brms_standardise_woPriorCases <- readRDS("output/brms_models/incidence_from_vax_model_village_woPriorCases_standardise.rds")
model_immunity_vill_month_brms_unlogged<-readRDS("output/brms_models/incidence_from_immunity_model_village_unlogged.rds")
model_immunity_vill_month_brms <- readRDS("output/brms_models/incidence_from_immunity_model_village.rds")



# WAIC
#--------------

model_sim<-"model_immunity_vill_month_brms"
samples_pars <- posterior_samples(get(model_sim), pars = c("b_Intercept", "b_immune_last2monthMean", "b_immune_neighbours_last2monthMean", "b_immune_notNeighbours_last2monthMean", "b_log_case_rate_last2monthMean", "b_log_case_rate_neighbours_last2monthMean", "b_log_case_rate_notNeighbours_last2monthMean", "b_log_dog_density", "b_HDR","shape"))
samples_reffs <- posterior_samples(get(model_sim), pars = c("r_village"))
log_lik_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       data_vill$immune_last2monthMean,data_vill$immune_neighbours_last2monthMean,data_vill$immune_notNeighbours_last2monthMean,
                       data_vill$log_case_rate_last2monthMean,data_vill$log_case_rate_neighbours_last2monthMean,data_vill$log_case_rate_notNeighbours_last2monthMean,
                       data_vill$log_dog_density,data_vill$HDR,
                       "village_re"=NA,log(data_vill$dogs))))
for(i in 1:nrow(samples_pars)){
  X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
  mu <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(ncol(samples_pars)-1))]),1,1)))
  log_lik_mat[,i] <- dnbinom(data_vill$cases,mu=mu,size=samples_pars[i,"shape"],log = T)
}
log_lik_mat <- t(log_lik_mat[(88*2+1):nrow(log_lik_mat),])
waic_full_immunity <- waic(log_lik_mat)
loo_full_immunity <- loo(log_lik_mat)

model_sim<-"model_vill_month_brms_unlogged"
samples_pars <- posterior_samples(get(model_sim), pars = c("b_Intercept", "b_vax_last2monthMean", "b_vax_neighbours_last2monthMean", "b_vax_notNeighbours_last2monthMean", "b_case_rate_last2monthMean", "b_case_rate_neighbours_last2monthMean", "b_case_rate_notNeighbours_last2monthMean", "b_log_dog_density", "b_HDR","shape"))
samples_reffs <- posterior_samples(get(model_sim), pars = c("r_village"))
log_lik_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       data_vill$vax_last2monthMean,data_vill$vax_neighbours_last2monthMean,data_vill$vax_notNeighbours_last2monthMean,
                       data_vill$case_rate_last2monthMean,data_vill$case_rate_neighbours_last2monthMean,data_vill$case_rate_notNeighbours_last2monthMean,
                       data_vill$log_dog_density,data_vill$HDR,
                       "village_re"=NA,log(data_vill$dogs))))
for(i in 1:nrow(samples_pars)){
  X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
  mu <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(ncol(samples_pars)-1))]),1,1)))
  log_lik_mat[,i] <- dnbinom(data_vill$cases,mu=mu,size=samples_pars[i,"shape"],log = T)
}
log_lik_mat <- t(log_lik_mat[(88*2+1):nrow(log_lik_mat),])
waic_full_unlogged <- waic(log_lik_mat)
loo_full_unlogged <- loo(log_lik_mat)

model_sim<-"model_vill_month_brms"
samples_pars <- posterior_samples(get(model_sim), pars = c("b_Intercept", "b_vax_last2monthMean", "b_vax_neighbours_last2monthMean", "b_vax_notNeighbours_last2monthMean", "b_log_case_rate_last2monthMean", "b_log_case_rate_neighbours_last2monthMean", "b_log_case_rate_notNeighbours_last2monthMean", "b_log_dog_density", "b_HDR","shape"))
samples_reffs <- posterior_samples(get(model_sim), pars = c("r_village"))
log_lik_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       data_vill$vax_last2monthMean,data_vill$vax_neighbours_last2monthMean,data_vill$vax_notNeighbours_last2monthMean,
                       data_vill$log_case_rate_last2monthMean,data_vill$log_case_rate_neighbours_last2monthMean,data_vill$log_case_rate_notNeighbours_last2monthMean,
                       data_vill$log_dog_density,data_vill$HDR,
                       "village_re"=NA,log(data_vill$dogs))))
for(i in 1:nrow(samples_pars)){
  X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
  mu <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(ncol(samples_pars)-1))]),1,1)))
  log_lik_mat[,i] <- dnbinom(data_vill$cases,mu=mu,size=samples_pars[i,"shape"],log = T)
}
log_lik_mat <- t(log_lik_mat[(88*2+1):nrow(log_lik_mat),])
waic_full <- waic(log_lik_mat)
loo_full <- loo(log_lik_mat)


model_sim<-"model_vill_month_brms_woPriorCases"
samples_pars <- posterior_samples(get(model_sim), pars = c("b_Intercept", "b_vax_last2monthMean", "b_vax_neighbours_last2monthMean", "b_vax_notNeighbours_last2monthMean", "b_log_dog_density", "b_HDR","shape"))
samples_reffs <- posterior_samples(get(model_sim), pars = c("r_village"))
log_lik_mat <- matrix(NA,nrow=nrow(data_vill),ncol=nrow(samples_pars))
X <- t(as.matrix(cbind(1,
                       data_vill$vax_last2monthMean,data_vill$vax_neighbours_last2monthMean,data_vill$vax_notNeighbours_last2monthMean,
                       data_vill$log_dog_density,data_vill$HDR,
                       "village_re"=NA,log(data_vill$dogs))))
for(i in 1:nrow(samples_pars)){
  X["village_re",] <- rep(as.numeric(samples_reffs[i,]),ncol(vax_vill))
  mu <- exp(colSums(X*c(as.numeric(samples_pars[i,c(1:(ncol(samples_pars)-1))]),1,1)))
  log_lik_mat[,i] <- dnbinom(data_vill$cases,mu=mu,size=samples_pars[i,"shape"],log = T)
}
log_lik_mat <- t(log_lik_mat[(88*2+1):nrow(log_lik_mat),])
waic_woPriorCases <- waic(log_lik_mat)
loo_woPriorCases <- loo(log_lik_mat)

waic_full
waic_full_unlogged
loo_full
loo_full_unlogged
# logged a little better

waic_full
waic_woPriorCases
loo_full
loo_woPriorCases
# better with prior cases

waic_full
waic_full_immunity
loo_full
loo_full_immunity
#vaccination model basically identical to immunity model so stick with simpler one with no immunity loss



# Model diagnostics
#--------------

if(run_models==T){ #slow!!
  
  model.check <- createDHARMa(
    simulatedResponse = t(posterior_predict(model_vill_month_brms)),
    observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
    fittedPredictedResponse = apply(t(posterior_epred(model_vill_month_brms)), 1, median),
    integerResponse = TRUE)
  
  pdf("Figs/model_vill_month_brms_diagnostics.pdf",width=14, height=10.5)
  par(mfrow=c(3,4))
  plotQQunif(model.check,pch=4,col="navy")
  testZeroInflation(model.check)
  plotResiduals(model.check,quantreg=T,pch=4,col=scales::alpha("navy",0.2),smoothScatter=FALSE)
  # testDispersion(model.check)# not really an issue for negative binomial models
  plotResiduals(model.check,data_vill$vax_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in village over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage at borders over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$case_rate_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in village over previous 2 months") # could be better but not terrible
  plotResiduals(model.check,data_vill$case_rate_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog at borders over previous 2 months") # ok
  plotResiduals(model.check,data_vill$case_rate_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Human:dog ratio") # ok
  plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Dog density") # ok
  plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*2))]),col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Village",text.cex=0.9,ylab="DHARMa residual") # ok
  dev.off()
  
  model.check <- createDHARMa(
    simulatedResponse = t(posterior_predict(model_vill_month_brms_woDistantPriorCases)),
    observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
    fittedPredictedResponse = apply(t(posterior_epred(model_vill_month_brms_woDistantPriorCases)), 1, median),
    integerResponse = TRUE)
  
  pdf("Figs/model_vill_month_brms_woDistantPriorCases_diagnostics.pdf",width=14, height=10.5)
  par(mfrow=c(3,4))
  plotQQunif(model.check,pch=4,col="navy")
  testZeroInflation(model.check)
  plotResiduals(model.check,quantreg=T,pch=4,col=scales::alpha("navy",0.2),smoothScatter=FALSE)
  plotResiduals(model.check,data_vill$vax_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in village over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage at borders over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$case_rate_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in village over previous 2 months") # not great
  plotResiduals(model.check,data_vill$case_rate_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog at borders over previous 2 months") # could be better
  plotResiduals(model.check,data_vill$case_rate_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Human:dog ratio") # ok
  plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Dog density") # ok
  plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*2))]),col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Village",text.cex=0.9,ylab="DHARMa residual") # ok
  dev.off()
  
  model.check <- createDHARMa(
    simulatedResponse = t(posterior_predict(model_vill_month_brms_woPriorCases)),
    observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
    fittedPredictedResponse = apply(t(posterior_epred(model_vill_month_brms_woPriorCases)), 1, median),
    integerResponse = TRUE)
  
  pdf("Figs/model_vill_month_brms_woPriorCases_diagnostics.pdf",width=14, height=10.5)
  par(mfrow=c(3,4))
  plotQQunif(model.check,pch=4,col="navy")
  testZeroInflation(model.check)
  plotResiduals(model.check,quantreg=T,pch=4,col=scales::alpha("navy",0.2),smoothScatter=FALSE)
  plotResiduals(model.check,data_vill$vax_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in village over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage at borders over previous 2 months") # ok
  plotResiduals(model.check,data_vill$vax_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Mean coverage in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$case_rate_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in village over previous 2 months") # not good
  plotResiduals(model.check,data_vill$case_rate_neighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog at borders over previous 2 months") # not great
  plotResiduals(model.check,data_vill$case_rate_notNeighbours_last2monthMean[-c(1:(nrow(cases_vill)*2))],col=scales::alpha("navy",0.2),pch=4,quantreg = T,smoothScatter=FALSE, xlab="Mean cases/dog in non-neighbouring villages over previous 2 months") # ok
  plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Human:dog ratio") # ok
  plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*2))],quantreg = T,col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Dog density") # ok
  plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*2))]),col=scales::alpha("navy",0.2),pch=4,smoothScatter=FALSE,xlab="Village",text.cex=0.9,ylab="DHARMa residual") # ok
  dev.off()
}



#Explore coefficients
#--------------

model_summary <- summary(model_vill_month_brms)
pars <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,1:4]
rownames(pars)[2:nrow(pars)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                  "Log cases/dog over last 2 months in village","Log cases/dog at borders over last 2 months","Log cases/dog in non-bordering villages over last 2 months",
                                  "Log dogs/km2","Human:dog ratio","Village RE SD","shape")

model_summary_woDistantPriorCases <- summary(model_vill_month_brms_woDistantPriorCases)
pars_woDistantPriorCases <- round(rbind(model_summary_woDistantPriorCases$fixed,model_summary_woDistantPriorCases$random$village,model_summary_woDistantPriorCases$spec_pars),2)[,1:4]
rownames(pars_woDistantPriorCases)[2:nrow(pars_woDistantPriorCases)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                                                          "Log cases/dog over last 2 months in village","Log dogs/km2","Human:dog ratio","Village RE SD","shape")

model_summary_woPriorCases <- summary(model_vill_month_brms_woPriorCases)
pars_woPriorCases <- round(rbind(model_summary_woPriorCases$fixed,model_summary_woPriorCases$random$village,model_summary_woPriorCases$spec_pars),2)[,1:4]
rownames(pars_woPriorCases)[2:nrow(pars_woPriorCases)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                                            "Log dogs/km2","Human:dog ratio","Village RE SD","shape")

# Prep for MS table
pars_exp <- round(exp(model_summary$fixed),2)[,c(1,3,4)]
pars_prep <- data.frame(matrix(NA, nrow=nrow(pars), ncol=2,dimnames = list(rownames(pars),c("Estimate","exp(Estimate)"))))
pars_prep[,1] <- paste0(pars$Estimate," (",pars$`l-95% CI`,", ",pars$`u-95% CI`,")")
pars_prep[1:nrow(pars_exp),2] <- paste0(pars_exp$Estimate," (",pars_exp$`l-95% CI`,", ",pars_exp$`u-95% CI`,")")
pars_prep2 <- cbind(pars_prep[,1],NA,NA);rownames(pars_prep2)<-rownames(pars_prep);colnames(pars_prep2) <- c("Full model","Without prior cases/dog beyond village","Without prior cases/dog at any scale")
pars_exp <- round(exp(model_summary_woPriorCases$fixed),2)[,c(1,3,4)]
pars_prep <- data.frame(matrix(NA, nrow=nrow(pars_woPriorCases), ncol=2,dimnames = list(rownames(pars_woPriorCases),c("Estimate","exp(Estimate)"))))
pars_prep[,1] <- paste0(pars_woPriorCases$Estimate," (",pars_woPriorCases$`l-95% CI`,", ",pars_woPriorCases$`u-95% CI`,")")
pars_prep[1:nrow(pars_exp),2] <- paste0(pars_exp$Estimate," (",pars_exp$`l-95% CI`,", ",pars_exp$`u-95% CI`,")")
pars_prep2[rownames(pars_prep),3] <- pars_prep[,1]
pars_exp <- round(exp(model_summary_woDistantPriorCases$fixed),2)[,c(1,3,4)]
pars_prep <- data.frame(matrix(NA, nrow=nrow(pars_woDistantPriorCases), ncol=2,dimnames = list(rownames(pars_woDistantPriorCases),c("Estimate","exp(Estimate)"))))
pars_prep[,1] <- paste0(pars_woDistantPriorCases$Estimate," (",pars_woDistantPriorCases$`l-95% CI`,", ",pars_woDistantPriorCases$`u-95% CI`,")")
pars_prep[1:nrow(pars_exp),2] <- paste0(pars_exp$Estimate," (",pars_exp$`l-95% CI`,", ",pars_exp$`u-95% CI`,")")
pars_prep2[rownames(pars_prep),2] <- pars_prep[,1]
write.csv(pars_prep2,"output/model_vill_month_pars_all_MSversion.csv")


model_summary_standardise <- summary(model_vill_month_brms_standardise)
pars_standardise <- round(rbind(model_summary_standardise$fixed,model_summary_standardise$random$village,model_summary_standardise$spec_pars),2)[,1:4]
rownames(pars_standardise)[2:nrow(pars_standardise)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                                          "Log cases/dog over last 2 months in village","Log cases/dog at borders over last 2 months","Log cases/dog in non-bordering villages over last 2 months",
                                                          "Log dogs/km2","Human:dog ratio","Village RE SD","shape")

model_summary_standardise_woDistantPriorCases <- summary(model_vill_month_brms_standardise_woDistantPriorCases)
pars_standardise_woDistantPriorCases <- round(rbind(model_summary_standardise_woDistantPriorCases$fixed,model_summary_standardise_woDistantPriorCases$random$village,model_summary_standardise_woDistantPriorCases$spec_pars),2)[,1:4]
rownames(pars_standardise_woDistantPriorCases)[2:nrow(pars_standardise_woDistantPriorCases)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                                                                                  "Log cases/dog over last 2 months in village","Log dogs/km2","Human:dog ratio","Village RE SD","shape")

model_summary_standardise_woPriorCases <- summary(model_vill_month_brms_standardise_woPriorCases)
pars_standardise_woPriorCases <- round(rbind(model_summary_standardise_woPriorCases$fixed,model_summary_standardise_woPriorCases$random$village,model_summary_standardise_woPriorCases$spec_pars),2)[,1:4]
rownames(pars_standardise_woPriorCases)[2:nrow(pars_standardise_woPriorCases)] <- c("Vaccination over last 2 months in village","Vaccination at borders over last 2 months","Vaccination in non-bordering villages over last 2 months",
                                                                                    "Log dogs/km2","Human:dog ratio","Village RE SD","shape")

# some summaries for the manuscript (full model)
(1-exp(pars["Vaccination over last 2 months in village",c(1,3,4)]*0.1))*100 # % change with 10% change in village Vaccination
(1-exp(pars["Vaccination over last 2 months in village",c(1,3,4)]*0.45))*100 # % change with 45% change in village vax
(1-exp(pars["Vaccination at borders over last 2 months",c(1,3,4)]*0.1))*100 # 
(1-exp(pars["Vaccination in non-bordering villages over last 2 months",c(1,3,4)]*0.1))*100
(1-(exp(pars["Vaccination over last 2 months in village",c(1,3,4)]*0.1)*exp(pars["Vaccination at borders over last 2 months",c(1,3,4)]*0.1)*exp(pars["Vaccination in non-bordering villages over last 2 months",c(1,3,4)]*0.1)))*100
mean((1-(exp(posterior_samples(model_vill_month_brms)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100)
quantile((1-(exp(posterior_samples(model_vill_month_brms)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100,c(0.025,0.975))
mean((1-(exp(posterior_samples(model_vill_month_brms)[,"b_vax_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100)
quantile((1-(exp(posterior_samples(model_vill_month_brms)[,"b_vax_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100,c(0.025,0.975))
length(which(posterior_samples(model_vill_month_brms)[,"b_vax_neighbours_last2monthMean"]>0&posterior_samples(model_vill_month_brms)[,"b_vax_notNeighbours_last2monthMean"]>0))/nrow(posterior_samples(model_vill_month_brms))
100*((exp(pars["Log cases/dog at borders over last 2 months",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog at borders over last 2 months",c(1,3,4)]*log(1)))-1) 
100*((exp(pars["Log cases/dog in non-bordering villages over last 2 months",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog in non-bordering villages over last 2 months",c(1,3,4)]*log(1)))-1) 
100*((exp(pars["Log cases/dog over last 2 months in village",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog over last 2 months in village",c(1,3,4)]*log(1)))-1) 
100*(((exp(pars["Log cases/dog in non-bordering villages over last 2 months",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog in non-bordering villages over last 2 months",c(1,3,4)]*log(1))))*
       ((exp(pars["Log cases/dog at borders over last 2 months",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog at borders over last 2 months",c(1,3,4)]*log(1))))*
       ((exp(pars["Log cases/dog over last 2 months in village",c(1,3,4)]*log(2)))/(exp(pars["Log cases/dog over last 2 months in village",c(1,3,4)]*log(1))))
     -1) 
mean(100*(((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_notNeighbours_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_notNeighbours_last2monthMean"]*log(1))))*
       ((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_neighbours_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_neighbours_last2monthMean"]*log(1))))*
       ((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_last2monthMean"]*log(1))))
     -1) )
quantile(100*(((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_notNeighbours_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_notNeighbours_last2monthMean"]*log(1))))*
            ((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_neighbours_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_neighbours_last2monthMean"]*log(1))))*
            ((exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_last2monthMean"]*log(2)))/(exp(posterior_samples(model_vill_month_brms)[,"b_log_case_rate_last2monthMean"]*log(1))))
          -1),c(0.025,0.975) )

100*(1-(exp(pars["Log dogs/km2",c(1,3,4)]*log(2)))/(exp(pars["Log dogs/km2",c(1,3,4)]*log(1))))

range_HDR <- range(data_vill$HDR)
100*((exp(pars["Human:dog ratio",c(1,3,4)]*1))-1) # % change when increasing HDR by 1
100*((exp(pars["Human:dog ratio",c(1,3,4)]*range_HDR[2]))/(exp(pars["Human:dog ratio",c(1,3,4)]*range_HDR[1]))-1) # % change over range of HDR

range(exp(ranef(model_vill_month_brms)$village[,"Estimate",1]))

corrplot::corrplot(cor(posterior_samples(model_vill_month_brms)[,c("b_vax_last2monthMean","b_vax_neighbours_last2monthMean","b_vax_notNeighbours_last2monthMean")]))

# some summaries for the manuscript (model without prior cases)
(1-exp(pars_woPriorCases["Vaccination over last 2 months in village",c(1,3,4)]*0.1))*100 # % change with 10% change in village Vaccination
mean((1-(exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100)
quantile((1-(exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_neighbours_last2monthMean"]*0.1)*exp(posterior_samples(model_vill_month_brms_woPriorCases)[,"b_vax_notNeighbours_last2monthMean"]*0.1)))*100,c(0.025,0.975))
(1-exp(pars_woPriorCases["Vaccination at borders over last 2 months",c(1,3,4)]*0.1))*100 # 




#Plot full model
#--------------

pdf("Figs/MonthlyModelCases&VaxVill.pdf",width=7, height=7)
cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5

# Plot model predictions at different coverages and case rates
data_vill_case_rate_adjust_dist <- 0.5*min(data_vill$case_rate_last2monthMean_dist[which(data_vill$case_rate_last2monthMean_dist>0)])
par(mar=c(2.5,2.5,0.25,0.1))
par(fig=c(0.57,1,0.62,1))
range(data_vill$case_rate_last2monthMean_dist,na.rm = T)
case_rate <- seq(0,0.001,length.out=5)
coverage <- seq(0,1,length.out=100)
cols=viridis(length(case_rate))
new_data <- data.frame(dogs=1,
                       log_dog_density=log(mean(data_vill$dog_density)),
                       vax_neighbours_last2monthMean=rep(coverage,length(case_rate)), 
                       vax_notNeighbours_last2monthMean=rep(coverage,length(case_rate)), 
                       vax_last2monthMean=rep(coverage,length(case_rate)), 
                       log_case_rate_last2monthMean=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(coverage)),
                       log_case_rate_neighbours_last2monthMean=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(coverage)),
                       log_case_rate_notNeighbours_last2monthMean=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(coverage)),
                       HDR=mean(data_vill$HDR)) 
preds_mat <- posterior_epred(model_vill_month_brms, newdata = new_data, re_formula = NA)*1000
preds_mat_upper <- matrix(apply(preds_mat, 2, quantile,probs=0.975),nrow=length(coverage),ncol=length(case_rate))
preds_mat_lower <- matrix(apply(preds_mat, 2, quantile,probs=0.025),nrow=length(coverage),ncol=length(case_rate))
preds_mat <- matrix(apply(preds_mat, 2, mean),nrow=length(coverage),ncol=length(case_rate))
col_pal <- viridis(length(case_rate))
plot(NA,ylim=c(0,max(preds_mat_upper)),xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in village\n(mean & 95% CrI)",side=2,line=1.5,cex=cex.lab)
mtext("Mean rolling vaccination coverage at\nall scales over prior two months",side=1,line=2,cex=cex.lab)
for(i in 1:length(case_rate)){
  polygon(c(coverage,rev(coverage)),c(preds_mat_lower[,i],rev(preds_mat_upper[,i])),col=scales::alpha(cols[i],0.25),border=NA)
}
for(i in 1:length(case_rate)){
  lines((preds_mat[,i])~coverage,col=cols[i],lwd=2,lty=1)
}
graphics::legend(0.18,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[1:3],pch=15,
                 col=scales::alpha(cols,0.25)[1:3],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all scales\n  over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.455,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[4:5],pch=15,
                 col=scales::alpha(cols,0.25)[c(4:5)],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all scales\n  over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.32,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T)),
                 col=cols,y.intersp = 1.42,lty=1,lwd=2,title.col = "white",
                 title="Mean cases/1,000 dogs at all scales\n over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=2)
graphics::legend(0.18,max((preds_mat_upper))*1.05,legend="Mean cases/1,000 dogs at all\nscales over prior 2 months:",bty="n",cex=0.75)
graphics::legend("topleft",legend="B",text.font = 2,bty="n")

# standardised exponentiated
par(fig=c(0,0.56,0.34,1),new=T)
par(mar=c(2,9,0.25,0))
pars_sub <- pars_standardise[2:(nrow(pars)-2),]
pars_sub_woPriorCases <- pars_standardise_woPriorCases[2:(nrow(pars_standardise_woPriorCases)-2),]
xlim=c((min(pars_sub$`l-95% CI`)),(max(pars_sub$`u-95% CI`)))
xlim=log(c(0.5,2))
# xlim=c((min(pars_sub$`l-95% CI`,pars_sub_woPriorCases$`l-95% CI`)),(max(pars_sub$`u-95% CI`,pars_sub_woPriorCases$`u-95% CI`)))
plot(NA,xlim=xlim,ylim=c(0.5,nrow(pars_sub)-0.5),axes=F,ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(1,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("grey40"))
axis(2,cex.axis=cex.axis,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(2,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(1.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("black"))
at=log(c(0.5,1,2))
label=c(0.5,1,2)
axis(1,cex.axis=cex.axis,padj=-1.5,at=at,label=label)
lines(c(0,0),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.25,cex=cex.lab)
box(bty="l")
arrows(x0=(pars_sub$`l-95% CI`),x1=(pars_sub$`u-95% CI`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col="navy")
# arrows(x0=(pars_sub_woPriorCases$`l-95% CI`),x1=(pars_sub_woPriorCases$`u-95% CI`),y0=seq(nrow(pars_sub)-0.7,1-0.7,-1)[c(1:3,7:nrow(pars))],y1=seq(nrow(pars_sub)-0.7,1-0.7,-1)[c(1:3,7:nrow(pars))],length=0,lwd=2,col="lightblue")
points((pars_sub$Estimate),seq(nrow(pars_sub)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
# points((pars_sub_woPriorCases$Estimate),seq(nrow(pars_sub)-0.7,1-0.7,-1)[c(1:3,7:nrow(pars_sub))],col="pink",pch=20,cex=1.3)
legend("topleft",legend="A",text.font = 2,bty="n")
# legend("topleft",legend=c("With prior cases","Without prior cases"),col=c("navy","lightblue"),lwd=2,cex=0.7,pt.cex =1,text.col = "white")
# legend("topleft",legend=c("With prior cases","Without prior cases"),col=c("red","pink"),lwd=2,cex=0.7,pt.cex =1,lty=0,pch=20,bty="n",)

# random effect
par(fig=c(0.5,1,0.31,0.63),new=T)
par(mar=c(0,3,1.5,2))
ranefs <- ranef(model_vill_month_brms)$village[,"Estimate",1]
breaks=c(seq(min(ranefs),0,length.out=51),
         seq(0,max(ranefs),length.out=50)[-1])
colours=colorRampPalette(c("dodgerblue","white","red"))(length(breaks)-1)
plot(SD_vill,col=colours[findInterval(ranefs,breaks,all.inside=T)],
     cex.main=0.8,lwd=0.5,border="grey")
plot(SD_outline,add=T)
legend("topleft",legend="C",text.font = 2,bty="n")
grid <- raster(extent(SD_vill),crs=SD_vill@proj4string);res(grid) <- 1000;grid[]<-1
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=colours,
     legend.args=list(text="exp(Village random effect)", side=4, line=1.5, cex=cex.lab),
     axis.args=list(at=log(c(0.25,0.5,1,2,4)),labels=c(0.25,0.50,1.00,2.00,4.00),cex.axis=cex.axis,hadj=0.5),
     smallplot=c(0.8,0.82, .25,.75))


# Plot case predictions and data over time
par(mar=c(2.5,2.5,1.5,1))
par(fig=c(0,0.5,0,0.33),new=T)
preds_mat <- predict(model_vill_month_brms, re_formula=NULL, ndraws=5000, summary=F) 
preds_mat_dist <- rowsum(t(preds_mat),data_vill$month[-c(1:(nrow(cases_vill)*2))])
preds_lower <- apply(preds_mat_dist,1,quantile,0.025)
preds_upper <- apply(preds_mat_dist,1,quantile,0.975)
plot(NA,ylim=c(0,max(preds_upper,data_dist$cases)),xlim=c(1,max(data_dist$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1,at=c(0,50,100))
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_dist$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col="skyblue",border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases[months_plot]~data_dist$month[months_plot],col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend("topright",c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c("skyblue","navy","red"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #9%
legend("topleft",legend="D",text.font = 2,bty="n")

# Plot human bite predictions and data over time
par(mar=c(2.5,2.5,1.5,1))
par(fig=c(0.5,1,0,0.33),new=T)
bite_preds_mat_dist <- preds_mat_dist
bite_preds_mat_dist[] <- sapply(preds_mat_dist,function(x){sum(rnbinom(x,mu=human_bites_fit$estimate["mu"],size=human_bites_fit$estimate["size"]))})
bite_preds_lower <- apply(bite_preds_mat_dist,1,quantile,0.025)
bite_preds_upper <- apply(bite_preds_mat_dist,1,quantile,0.975)
plot(NA,ylim=c(0,max(bite_preds_upper,monthlyBites)),xlim=c(1,max(data_dist$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Human bites by rabid\ndogs in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_dist$month)
polygon(c(months_plot,rev(months_plot)),c(bite_preds_lower,rev(bite_preds_upper)),col=scales::alpha("orange",0.5),border=NA)
out_PI <- which(monthlyBites[months_plot]<bite_preds_lower|monthlyBites[months_plot]>bite_preds_upper)+2
points(monthlyBites[months_plot]~data_dist$month[months_plot],col="darkorange",pch=20,cex=cex.pt)
points(monthlyBites[out_PI]~data_dist$month[out_PI],col="red3",pch=20,cex=cex.pt)
legend("topright",c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c(scales::alpha("orange",0.5),"darkorange","red3"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #9%
legend("topleft",legend="E",text.font = 2,bty="n")

dev.off()



#Plot model without distant prior cases
#--------------

pdf("Figs/MonthlyModelVax&LocalCasesVill.pdf",width=7, height=4.7)
cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5

# Plot model predictions at different coverages and case rates
par(mar=c(2.5,2.5,0.5,1))
par(fig=c(0.5,1,0.5,1))
range(data_vill$case_rate_last2monthMean_dist,na.rm = T)
case_rate <- seq(0,0.001,length.out=5)
coverage <- seq(0,1,length.out=100)
cols=viridis(length(case_rate))
new_data <- data.frame(dogs=1,
                       log_dog_density=log(mean(data_vill$dog_density)),
                       vax_neighbours_last2monthMean=rep(coverage,length(case_rate)), 
                       vax_notNeighbours_last2monthMean=rep(coverage,length(case_rate)), 
                       vax_last2monthMean=rep(coverage,length(case_rate)), 
                       log_case_rate_last2monthMean=rep(log(case_rate+data_vill_case_rate_adjust_dist),each=length(coverage)),
                       HDR=mean(data_vill$HDR)) 
preds_mat <- posterior_epred(model_vill_month_brms_woDistantPriorCases, newdata = new_data, re_formula = NA)*1000
preds_mat_upper <- matrix(apply(preds_mat, 2, quantile,probs=0.975),nrow=length(coverage),ncol=length(case_rate))
preds_mat_lower <- matrix(apply(preds_mat, 2, quantile,probs=0.025),nrow=length(coverage),ncol=length(case_rate))
preds_mat <- matrix(apply(preds_mat, 2, mean),nrow=length(coverage),ncol=length(case_rate))
col_pal <- viridis(length(case_rate))
plot(NA,ylim=c(0,max(preds_mat_upper)),xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in village\n(mean & 95% CrI)",side=2,line=1.5,cex=cex.lab)
mtext("Mean rolling vaccination coverage at\nall scales over prior two months",side=1,line=2,cex=cex.lab)
for(i in 1:length(case_rate)){
  polygon(c(coverage,rev(coverage)),c(preds_mat_lower[,i],rev(preds_mat_upper[,i])),col=scales::alpha(cols[i],0.25),border=NA)
}
for(i in 1:length(case_rate)){
  lines((preds_mat[,i])~coverage,col=cols[i],lwd=2,lty=1)
}
graphics::legend(0.2,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[1:3],pch=15,
                 col=scales::alpha(cols,0.25)[1:3],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all scales\n  over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.4,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T))[4:5],pch=15,
                 col=scales::alpha(cols,0.25)[c(4:5)],pt.cex =2.5,y.intersp = 1.42,text.col="white",
                 title="Mean cases/1,000 dogs at all scales\n  over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=1)
graphics::legend(0.32,max((preds_mat_upper)),paste0(format(case_rate*1000,scientific=F,drop0trailing=T)),
                 col=cols,y.intersp = 1.42,lty=1,lwd=2,title.col = "white",
                 title="Mean cases/1,000 dogs at all scales\n over prior 2 months:",
                 title.adj =-0.1,title.cex=0.75,cex=0.7,bty="n",ncol=2)
graphics::legend(0.2,max((preds_mat_upper))*1.05,legend="Mean cases/1,000 dogs at all\nscales over prior 2 months:",bty="n",cex=0.75)

graphics::legend("topleft",legend="B",text.font = 2,bty="n")


# standardised exponentiated
par(fig=c(0,0.5,0.5,1),new=T)
par(mar=c(2.5,7.5,0.25,2))
pars_sub <- pars_standardise_woDistantPriorCases[2:(nrow(pars_standardise_woDistantPriorCases)-2),]
plot(NA,xlim=c(min(exp(pars_sub$`l-95% CI`)),max(exp(pars_sub$`u-95% CI`))),ylim=c(0.5,nrow(pars_sub)-0.5),axes=F,ylab="",xlab="")
axis(2,cex.axis=cex.axis-0.1,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(1,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("grey40"))
axis(2,cex.axis=cex.axis-0.1,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(2,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(1.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("black"))
axis(1,cex.axis=cex.axis,padj=-1.5)
lines(c(1,1),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.5,cex=cex.lab)
box(bty="l")
arrows(x0=exp(pars_sub$`l-95% CI`),x1=exp(pars_sub$`u-95% CI`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col="navy")
points(exp(pars_sub$Estimate),seq(nrow(pars_sub)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
legend("topleft",legend="A",text.font = 2,bty="n")

# random effect
par(fig=c(0,0.5,0,0.5),new=T)
par(mar=c(0,1,1.5,2))
ranefs <- ranef(model_vill_month_brms_woDistantPriorCases)$village[,"Estimate",1]
breaks=c(seq(min(ranefs),0,length.out=51),
         seq(0,max(ranefs),length.out=50)[-1])
colours=colorRampPalette(c("dodgerblue","white","red"))(length(breaks)-1)
plot(SD_vill,col=colours[findInterval(ranefs,breaks,all.inside=T)],
     cex.main=0.8,lwd=0.5,border="grey")
plot(SD_outline,add=T)
legend("topleft",legend="C",text.font = 2,bty="n")
grid <- raster(extent(SD_vill),crs=SD_vill@proj4string);res(grid) <- 1000;grid[]<-1
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=colours,
     legend.args=list(text="exp(Village random effect)", side=4, line=2, cex=cex.lab),
     axis.args=list(at=log(c(0.125,0.25,0.5,1,2,4,8)),labels=c(0.125,0.250,0.500,1.000,2.000,4.000,8.000),cex.axis=cex.axis,hadj=0.2),
     smallplot=c(0.71,0.73, .25,.75))


# Plot case predictions and data over time
par(mar=c(2.5,2.5,1.5,1))
par(fig=c(0.5,1,0,0.5),new=T)
preds_mat <- predict(model_vill_month_brms_woDistantPriorCases, re_formula=NULL, ndraws=5000, summary=F) 
preds_mat_dist <- rowsum(t(preds_mat),data_vill$month[-c(1:(nrow(cases_vill)*2))])
preds_lower <- apply(preds_mat_dist,1,quantile,0.025)
preds_upper <- apply(preds_mat_dist,1,quantile,0.975)
plot(NA,ylim=c(0,max(preds_upper,data_dist$cases)),xlim=c(1,max(data_dist$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1,at=c(0,50,100))
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_dist$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col="skyblue",border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases[months_plot]~data_dist$month[months_plot],col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
# points(data_dist$cases[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)]~data_dist$month[-c(1:2)][which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper)],col="red",pch=20,cex=cex.pt)
legend("topright",c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c("skyblue","navy","red"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #9%
legend("topleft",legend="D",text.font = 2,bty="n")

dev.off()



#Plot model without prior cases
#--------------

pdf("Figs/MonthlyModelVaxVill.pdf",width=7, height=4.7)
cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5

# Plot model predictions at different coverages 
par(mar=c(2.5,3,0.5,0.5))
par(fig=c(0.5,1,0.5,1))
range(data_vill$case_rate_last2monthMean_dist,na.rm = T)
coverage <- seq(0,1,length.out=100)
cols=viridis(length(case_rate))
new_data <- data.frame(dogs=1,
                       log_dog_density=log(mean(data_vill$dog_density)),
                       vax_neighbours_last2monthMean=coverage, 
                       vax_notNeighbours_last2monthMean=coverage, 
                       vax_last2monthMean=coverage, 
                       HDR=mean(data_vill$HDR)) 
preds_mat <- posterior_epred(model_vill_month_brms_woPriorCases, newdata = new_data, re_formula = NA)*1000
preds_mat_upper <- matrix(apply(preds_mat, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_mat_lower <- matrix(apply(preds_mat, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds_mat <- matrix(apply(preds_mat, 2, mean),nrow=length(coverage),ncol=1)
plot(NA,ylim=c(0,max(preds_mat_upper)),xlim=c(0,1),bty="l",cex=cex.pt,axes=F,
     ylab="",xlab="")
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in village",side=2,line=1.5,cex=cex.lab)
mtext("Mean rolling vaccination coverage at\nall scales over prior two months",side=1,line=2,cex=cex.lab)
polygon(c(coverage,rev(coverage)),c(preds_mat_lower[,1],rev(preds_mat_upper[,1])),col="lightblue",border=NA)
lines((preds_mat[,1])~coverage,col="navy",lwd=2,lty=1)
graphics::legend("topleft",legend="B",text.font = 2,bty="n")


# standardised exponentiated
par(fig=c(0,0.5,0.5,1),new=T)
par(mar=c(2.5,7.8,0.25,1))
pars_sub <- pars_standardise_woPriorCases[2:(nrow(pars_standardise_woPriorCases)-2),]
xlim=c((min(pars_sub$`l-95% CI`)),(max(pars_sub$`u-95% CI`)))
plot(NA,xlim=xlim,ylim=c(0.5,nrow(pars_sub)-0.5),axes=F,ylab="",xlab="")
axis(2,cex.axis=cex.axis-0.1,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(1,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(0.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("grey40"))
axis(2,cex.axis=cex.axis-0.1,padj=0.5,labels=sapply(rev(rownames(pars_sub))[seq(2,nrow(pars_sub),2)], function(x) paste(strwrap(x, 32), collapse = "\n")),at=seq(1.5,nrow(pars_sub)-0.5,2),las=2,col.axis=c("black"))
at=log(c(0.25,0.5,1,2))
label=c(0.25,0.5,1,2)
axis(1,cex.axis=cex.axis,padj=-1.5,at=at,label=label)
lines(c(0,0),c(0,nrow(pars_sub)),lty=3)
mtext("exp(Standardised coefficient)",side=1,line=1.5,cex=cex.lab)
box(bty="l")
arrows(x0=(pars_sub$`l-95% CI`),x1=(pars_sub$`u-95% CI`),y0=seq(nrow(pars_sub)-0.5,0.5,-1),y1=seq(nrow(pars_sub)-0.5,0.5,-1),length=0,lwd=2,col="navy")
points((pars_sub$Estimate),seq(nrow(pars_sub)-0.5,0.5,-1),col="red",pch=20,cex=1.3)
legend("topleft",legend="A",text.font = 2,bty="n")


# random effect
par(fig=c(0,0.45,0,0.5),new=T)
par(mar=c(0,0,1.5,2))
ranefs <- ranef(model_vill_month_brms_woPriorCases)$village[,"Estimate",1]
breaks=c(seq(min(ranefs),0,length.out=51),
         seq(0,max(ranefs),length.out=50)[-1])
colours=colorRampPalette(c("dodgerblue","white","red"))(length(breaks)-1)
plot(SD_vill,col=colours[findInterval(ranefs,breaks,all.inside=T)],
     cex.main=0.8,lwd=0.5,border="grey")
plot(SD_outline,add=T)
legend("topleft",legend="C",text.font = 2,bty="n")
grid <- raster(extent(SD_vill),crs=SD_vill@proj4string);res(grid) <- 1000;grid[]<-1
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=colours,
     legend.args=list(text="exp(Village random effect)", side=4, line=2.5, cex=cex.lab),
     axis.args=list(at=log(c(0.0625,0.25,1,4,16)),labels=c(0.0625,0.25,1,4,16),cex.axis=cex.axis,hadj=0.15),
     smallplot=c(0.71,0.73, .25,.75))


# Plot case predictions and data over time
par(mar=c(2.5,3,1.5,0.5))
par(fig=c(0.5,1,0,0.5),new=T)
preds_mat <- predict(model_vill_month_brms_woPriorCases, re_formula=NULL, ndraws=5000, summary=F) 
preds_mat_dist <- rowsum(t(preds_mat),data_vill$month[-c(1:(nrow(cases_vill)*2))])
preds_lower <- apply(preds_mat_dist,1,quantile,0.025)
preds_upper <- apply(preds_mat_dist,1,quantile,0.975)
plot(NA,ylim=c(0,max(preds_upper,data_dist$cases)),xlim=c(1,max(data_dist$month)),bty="l",
     ylab="",xlab="",cex.lab=cex.lab,axes=F)
axis(2,cex.axis=cex.axis,padj=1,at=c(0,50,100))
axis(1,at=seq(1,length(2002:2022)*12,24),labels=paste(seq(2002,2022,2)),cex.axis=cex.axis,padj=-1.5)
mtext("Dog cases in district",side=2,line=1.5,cex=cex.lab)
mtext("Date",side=1,line=1.5,cex=cex.lab)
box(bty="l")
months_plot <- 3:max(data_dist$month)
polygon(c(months_plot,rev(months_plot)),c(preds_lower,rev(preds_upper)),col="skyblue",border=NA)
out_PI <- which(data_dist$cases[months_plot]<preds_lower|data_dist$cases[months_plot]>preds_upper)+2
points(data_dist$cases[months_plot]~data_dist$month[months_plot],col="navy",pch=20,cex=cex.pt)
points(data_dist$cases[out_PI]~data_dist$month[out_PI],col="red",pch=20,cex=cex.pt)
legend("topright",c("model 95% PI","data within 95% PI","data outside 95% PI"),col=c("skyblue","navy","red"),pch=c(15,20,20),cex=0.75,bty="n",pt.cex = c(1.5,cex.pt,cex.pt))
length(which(data_dist$cases[-c(1:2)]<preds_lower|data_dist$cases[-c(1:2)]>preds_upper))/length(preds_lower) #9%
legend("topleft",legend="D",text.font = 2,bty="n")


dev.off()


