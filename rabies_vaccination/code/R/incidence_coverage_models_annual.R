
rm(list=ls())

library(MASS)
library(DHARMa)
library(raster)
library(viridis)
library(RColorBrewer)
library(scales)
library(brms)
library(rgdal)
library(rgeos)
library(stringr)

set.seed(0)

fit <- F


## Load data
#________________

# Model data
data_vill <- read.csv("output/incidence_coverage_model_data_village_annual.csv")
data_dist <- read.csv("output/incidence_coverage_model_data_district_annual.csv")

# Vaccination estimates
vax_vill <- as.matrix(read.csv("output/vaccinationCoverageByVillageMonth_Jan2002_Dec2022.csv",header = F,row.names = 1)) # village level
vax_vill_annual <- as.matrix(read.csv("output/vcVillByYear.csv",header = F)) # annual district campaign coverages
vax_dist <- as.matrix(read.csv("output/districtVaccinationCoverage_Jan2002_Dec2022.csv",header = T)) # district level
vax_dist_annual <- as.matrix(read.csv("output/vcDistByYear.csv",header = F)) # annual district campaign coverages

# Case numbers
cases_dist <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_2002-01-01_2022-12-31.csv",header=F))
cases_vill <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_village_2002-01-01_2022-12-31.csv",header=F,row.names = 1))

# Dog population
dogs <- as.matrix(read.csv("output/dogPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))

# Human population
humans <- as.matrix(read.csv("output/humanPopulationByVillageMonth_Jan2002_Dec2022.csv",row.names = 1,header=F))

# Serengeti village shapefile
SD_vill <- readOGR("output/SD_vill","SD_vill") 
SD_vill<-SD_vill[order(SD_vill$Vill_2012),]
SD_outline <- gUnaryUnion(SD_vill)# get district outline
SD_outline<-gBuffer(SD_outline,width=1) # get rid of a few tiny holes
SD_vill_gridded <- readOGR("data/GIS","SD_Villages_2012_From_HHS_250m_Lattice_UTM") 

# Protected areas
PAs <- readOGR("data/GIS/ProtectedAreas","TZprotected_areas", p4s = ("+proj=utm +zone=37 +south +ellps=clrk80 +towgs84=-160,-6,-302,0,0,0,0 +units=m +no_defs"))
PAs <- spTransform(PAs,SD_outline@proj4string)
Serengeti <- PAs[which(PAs$SP_ID=="2"),]
Serengeti <- Serengeti - SD_vill_gridded



# District annual model
#________________

if(fit==T){
  model_dist_year_1 <- brm(formula=cases ~ vax_last_year + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_2 <- brm(formula=cases ~ vax_last2yearMean + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1:2),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_3 <- brm(formula=cases ~ vax_last3yearMean + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1:3),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_4 <- brm(formula=cases ~ vax_last_year+incidence_last_year  + log_dog_density + offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_5 <- brm(formula=cases ~ vax_last_year+log_incidence_last_year  + log_dog_density + offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_6 <- brm(formula=cases ~ vax_last_year_campaign + log_dog_density +  offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_7 <- brm(formula=cases ~ vax_last2yearMean_campaign + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1:2),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_8 <- brm(formula=cases ~ vax_last3yearMean_campaign + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1:3),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_9 <- brm(formula=cases ~ vax_last_year_campaign + incidence_last_year + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_dist_year_10 <- brm(formula=cases ~ vax_last_year_campaign + log_incidence_last_year + log_dog_density  + offset(log(dogs)),data=data_dist[-c(1),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                            control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  
  save(model_dist_year_1,model_dist_year_2,model_dist_year_3,model_dist_year_4,model_dist_year_5,
       model_dist_year_6,model_dist_year_7,model_dist_year_8,model_dist_year_9,model_dist_year_10,
       file="output/brms_models/annual_district_models.Rdata")
}
load("output/brms_models/annual_district_models.Rdata")


loo_compare(loo(model_dist_year_1),loo(model_dist_year_4),loo(model_dist_year_5),loo(model_dist_year_6),loo(model_dist_year_9),loo(model_dist_year_10))
# model with log cases on top, unlogged next, whether average or campaign coverage is better varies and difference is small (just go for campaign - it's better for village-level below)
loo_compare(loo(model_dist_year_2),loo(model_dist_year_7)) 
loo_compare(loo(model_dist_year_3),loo(model_dist_year_8)) 

# Check residuals
model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_1)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_1)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
testZeroInflation(model.check)
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_2)),
  observedResponse = data_dist$cases[-c(1:2)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_2)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last2yearMean[3:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[3:nrow(data_dist)],quantreg = T) 
testZeroInflation(model.check)
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_3)),
  observedResponse = data_dist$cases[-c(1:3)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_3)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last3yearMean[4:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[4:nrow(data_dist)],quantreg = T) 
testZeroInflation(model.check)
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_4)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_4)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$incidence_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
testZeroInflation(model.check)
# better

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_5)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_5)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$incidence_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
testZeroInflation(model.check)
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_6)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_6)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year_campaign[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_7)),
  observedResponse = data_dist$cases[-c(1:2)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_7)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last2yearMean_campaign[3:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[3:nrow(data_dist)],quantreg = T) 
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_8)),
  observedResponse = data_dist$cases[-c(1:3)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_8)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last3yearMean_campaign[4:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[4:nrow(data_dist)],quantreg = T) 
# quantiles not great qq fine

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_9)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_9)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year_campaign[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$log_cases_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
# bit better

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_dist_year_10)),
  observedResponse = data_dist$cases[-c(1)],
  fittedPredictedResponse = apply(t(posterior_epred(model_dist_year_10)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_dist$vax_last_year_campaign[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$log_cases_last_year[2:nrow(data_dist)],quantreg = T) 
plotResiduals(model.check,data_dist$dog_density[2:nrow(data_dist)],quantreg = T) 
# not terrible

# qq plots generally good

# issues with some quantile plots, particularly when don't include prior cases
# but so little data it's hard to be sure what's a real pattern


#Parameter tables
#---------

pars <- data.frame("Intercept"=rep("",4),
                   "District last year"=rep("",4),
                   "District mean last 2 years"=rep("",4),
                   "District mean last 3 years"=rep("",4),
                   "District ln(Cases/dog) last year"=rep("",4),
                   "ln(dog density)"=rep("",4),
                   "Shape"=rep("",4),
                   check.names = F)
model_summary <- summary(model_dist_year_1)
pars1 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars1 <- paste0(pars1[,1]," (",pars1[,2],"-",pars1[,3],")")
pars[1, c("Intercept","District last year","ln(dog density)" ,"Shape")] <- pars1
model_summary <- summary(model_dist_year_2)
pars2 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars2 <- paste0(pars2[,1]," (",pars2[,2],"-",pars2[,3],")")
pars[2, c("Intercept","District mean last 2 years" ,"ln(dog density)","Shape")] <- pars2
model_summary <- summary(model_dist_year_3)
pars3 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars3 <- paste0(pars3[,1]," (",pars3[,2],"-",pars3[,3],")")
pars[3, c("Intercept","District mean last 3 years" ,"ln(dog density)","Shape")] <- pars3
model_summary <- summary(model_dist_year_5)
pars4 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars4 <- paste0(pars4[,1]," (",pars4[,2],"-",pars4[,3],")")
pars[4, c("Intercept","District last year", "District ln(Cases/dog) last year","ln(dog density)" ,"Shape")] <- pars4

write.csv(pars,"output/annual_district_mean_annual_vax_model_pars.csv",row.names=F)


pars <- data.frame("Intercept"=rep("",4),
                   "District last year"=rep("",4),
                   "District mean last 2 years"=rep("",4),
                   "District mean last 3 years"=rep("",4),
                   "District ln(Cases/dog) last year"=rep("",4),
                   "ln(dog density)"=rep("",4),
                   "Shape"=rep("",4),
                   check.names = F)
model_summary <- summary(model_dist_year_6)
pars1 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars1 <- paste0(pars1[,1]," (",pars1[,2],"-",pars1[,3],")")
pars[1, c("Intercept","District last year","ln(dog density)" ,"Shape")] <- pars1
model_summary <- summary(model_dist_year_7)
pars2 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars2 <- paste0(pars2[,1]," (",pars2[,2],"-",pars2[,3],")")
pars[2, c("Intercept","District mean last 2 years" ,"ln(dog density)","Shape")] <- pars2
model_summary <- summary(model_dist_year_8)
pars3 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars3 <- paste0(pars3[,1]," (",pars3[,2],"-",pars3[,3],")")
pars[3, c("Intercept","District mean last 3 years" ,"ln(dog density)","Shape")] <- pars3
model_summary <- summary(model_dist_year_10)
pars4 <- round(rbind(model_summary$fixed,model_summary$spec_pars),2)[,c(1,3:4)]
pars4 <- paste0(pars4[,1]," (",pars4[,2],"-",pars4[,3],")")
pars[4, c("Intercept","District last year", "District ln(Cases/dog) last year","ln(dog density)" ,"Shape")] <- pars4

write.csv(pars,"output/annual_district_campaign_vax_model_pars.csv",row.names=F)



#Plots 
#---------

cex.axis <- 0.7
cex.lab <- 0.8
cex.pt <- 0.5


# Plot predictions (vaccination only models)
#---------------------

pdf("Figs/AnnualDistrictModelsVax.pdf",width=3.5, height=3)

par(fig=c(0,1,0,1))

par(mar=c(3.5,3.5,1,1))
coverage <- seq(0,1,length.out=100)
new_data_extrap <- data.frame(dogs=1,vax_last_year_campaign=coverage,vax_last2yearMean_campaign=coverage,vax_last3yearMean_campaign=coverage,
                              log_dog_density=log(mean(data_dist$dog_density)))
preds_extrap <- posterior_epred(model_dist_year_6, newdata = new_data_extrap, re_formula = NA)*1000
preds_extrap_upper1 <- matrix(apply(preds_extrap, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_extrap_lower1 <- matrix(apply(preds_extrap, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds_extrap1 <- apply(preds_extrap, 2, mean)
preds_extrap <- posterior_epred(model_dist_year_7, newdata = new_data_extrap, re_formula = NA)*1000
preds_extrap_upper2 <- matrix(apply(preds_extrap, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_extrap_lower2 <- matrix(apply(preds_extrap, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds_extrap2 <- apply(preds_extrap, 2, mean)
preds_extrap <- posterior_epred(model_dist_year_8, newdata = new_data_extrap, re_formula = NA)*1000
preds_extrap_upper3 <- matrix(apply(preds_extrap, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_extrap_lower3 <- matrix(apply(preds_extrap, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds_extrap3 <- apply(preds_extrap, 2, mean)

col_pal <- viridis(3)
plot(NA, ylim=c(0,100),xlim=c(0,1),bty="l",ylab="",xlab="",cex.lab=1.1,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in district this year",side=2,line=1.5,cex=cex.lab)
mtext("Campaign coverage in\ndistrict over previous years",side=1,line=2,cex=cex.lab)
polygon(c(coverage,rev(coverage)),c(preds_extrap_lower1[,1],rev(preds_extrap_upper1[,1])),col=alpha(col_pal[1],0.1),border=NA)
polygon(c(coverage,rev(coverage)),c(preds_extrap_lower2[,1],rev(preds_extrap_upper2[,1])),col=alpha(col_pal[2],0.1),border=NA)
polygon(c(coverage,rev(coverage)),c(preds_extrap_lower3[,1],rev(preds_extrap_upper3[,1])),col=alpha(col_pal[3],0.1),border=NA)
points((data_dist$incidence*1000)~data_dist$vax_last_year, col=alpha(col_pal[1],0.5),pch=20)
points((data_dist$incidence*1000)~data_dist$vax_last2yearMean, col=alpha(col_pal[2],0.5),pch=20)
points((data_dist$incidence*1000)~data_dist$vax_last3yearMean, col=alpha(col_pal[3],0.5),pch=20)
lines(preds_extrap1~coverage,col=col_pal[1],lwd=2,lty=1)
lines(preds_extrap2~coverage,col=col_pal[2],lwd=2,lty=1)
lines(preds_extrap3~coverage,col=col_pal[3],lwd=2,lty=1)

legend("topleft",c("year","2 years","3 years"),lty=1,col=col_pal,
       title="Campaign coverage\nover previous:",
       title.cex=0.8,cex=0.8,bty="n",lwd=2,yjust=1,title.adj = -0.01,inset=0.05)

dev.off()



# Plot predictions  (campaign coverage and cases models)
#---------------------

pdf("Figs/AnnualDistrictModelsCases&CampaignVax.pdf",width=4, height=3)

par(mar=c(3.5,2.5,1,4.5))
par(fig=c(0,1,0,1))
coverage <- seq(0,1,length.out=100)
incidence <- c(0,0.003,0.006,0.009)
breaks = seq(0,max(data_dist$incidence_last_year,na.rm=T),length.out=100)
data_dist_incidence_adjust <- 0.5*min(data_dist$incidence_last_year[which(data_dist$incidence_last_year>0)])
new_data_extrap <- data.frame(dogs=1,
                              vax_last_year_campaign=rep(coverage,length(incidence)), 
                              log_incidence_last_year=rep(log(incidence+data_dist_incidence_adjust),each=length(coverage)),
                              log_dog_density=log(mean(data_dist$dog_density)))
preds_extrap <- posterior_epred(model_dist_year_10, newdata = new_data_extrap, re_formula = NA)*1000
preds_mat_extrap_upper <- matrix(apply(preds_extrap, 2, quantile,probs=0.975),nrow=length(coverage),ncol=length(incidence))
preds_mat_extrap_lower <- matrix(apply(preds_extrap, 2, quantile,probs=0.025),nrow=length(coverage),ncol=length(incidence))
preds_extrap_mat <- matrix(apply(preds_extrap, 2, mean),nrow=length(coverage),ncol=length(incidence))
cols <- viridis(100)
plot((data_dist$incidence*1000)~data_dist$vax_last_year,
     col=cols[as.numeric(findInterval(data_dist$incidence_last_year,breaks))],
     pch=20,ylim=c(0,max(preds_mat_extrap_upper)),xlim=c(0,1),bty="l",
     ylab="",xlab="",cex.lab=1.1,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in district this year",side=2,line=1.5,cex=cex.lab)
mtext("Campaign coverage in\ndistrict last year",side=1,line=2,cex=cex.lab)
for(i in 1:length(incidence)){
  polygon(c(coverage,rev(coverage)),c(preds_mat_extrap_lower[,i],rev(preds_mat_extrap_upper[,i])),col=scales::alpha(cols[as.numeric(findInterval(incidence[i],breaks))],0.1),border=NA)}
for(i in 1:length(incidence)){
  lines(preds_extrap_mat[,i]~coverage,col=cols[as.numeric(findInterval(incidence[i],breaks))],lwd=2,lty=1)}
points((data_dist$incidence*1000)~data_dist$vax_last_year,
       col=cols[as.numeric(findInterval(data_dist$incidence_last_year,breaks))],
       pch=20)

grid <- raster(nrows=10, ncols=10);grid[]<-0.0001
plot(grid, 
     breaks=breaks,legend.only=T, add=T,col=cols,
     legend.args=list(text="Cases/1,000 dogs in district\nin the previous year", side=4, line=2.8, cex=0.8),
     axis.args=list(at=pretty(c(0,data_dist$incidence_last_year)),labels=pretty(c(0,data_dist$incidence_last_year))*1000,hadj=0.3,cex.axis=0.75),
     smallplot=c(0.78,0.8, .33,.88))

dev.off()




# Village annual model
#________________

if(fit==T){
  model_vill_year_1 <- brm(formula=cases ~ vax_last_year + vax_last_year_neighbours + vax_last_year_notNeighbours +  log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_2 <- brm(formula=cases ~ vax_last2yearMean + vax_last2yearMean_neighbours + vax_last2yearMean_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_3 <- brm(formula=cases ~ vax_last3yearMean + vax_last3yearMean_neighbours + vax_last3yearMean_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill)*3)),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_4 <- brm(formula=cases ~ vax_last_year + vax_last_year_neighbours + vax_last_year_notNeighbours+incidence_last_year + incidence_last_year_neighbours + incidence_last_year_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_5 <- brm(formula=cases ~ vax_last_year + vax_last_year_neighbours + vax_last_year_notNeighbours+log_incidence_last_year + log_incidence_last_year_neighbours + log_incidence_last_year_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_6 <- brm(formula=cases ~ vax_last_year_campaign + vax_last_year_neighbours_campaign + vax_last_year_notNeighbours_campaign + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_7 <- brm(formula=cases ~ vax_last2yearMean_campaign + vax_last2yearMean_neighbours_campaign + vax_last2yearMean_notNeighbours_campaign + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill)*2)),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_8 <- brm(formula=cases ~ vax_last3yearMean_campaign + vax_last3yearMean_neighbours_campaign + vax_last3yearMean_notNeighbours_campaign + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill)*3)),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_9 <- brm(formula=cases ~ vax_last_year_campaign + vax_last_year_neighbours_campaign + vax_last_year_notNeighbours_campaign+incidence_last_year + incidence_last_year_neighbours + incidence_last_year_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                           control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  model_vill_year_10 <- brm(formula=cases ~ vax_last_year_campaign + vax_last_year_neighbours_campaign + vax_last_year_notNeighbours_campaign +log_incidence_last_year + log_incidence_last_year_neighbours + log_incidence_last_year_notNeighbours + log_dog_density + HDR  + (1|village) + offset(log(dogs)),data=data_vill[-c(1:(nrow(cases_vill))),], family = negbinomial(),warmup = 1500, iter = 3000, chains = 4,
                            control = list(adapt_delta = 0.99,max_treedepth = 15), cores=4,save_pars = save_pars(all = TRUE))
  
  save(model_vill_year_1,model_vill_year_2,model_vill_year_3,model_vill_year_4,model_vill_year_5,
       file="output/brms_models/annual_village_models_average_coverage.Rdata")
  save(model_vill_year_6,model_vill_year_7,model_vill_year_8,model_vill_year_9,model_vill_year_10,
       file="output/brms_models/annual_village_models_campaign_coverage.Rdata")
}
load("output/brms_models/annual_village_models_average_coverage.Rdata")
load("output/brms_models/annual_village_models_campaign_coverage.Rdata")

# loo_compare(loo(model_vill_year_1,moment_match = TRUE),loo(model_vill_year_4,moment_match = TRUE),loo(model_vill_year_5,moment_match = TRUE),
#             loo(model_vill_year_6,moment_match = TRUE),loo(model_vill_year_9,moment_match = TRUE),loo(model_vill_year_10,moment_match = TRUE))
# # campaign coverage models better
# # with logged cases best, then unlogged cases
# loo_compare(loo(model_vill_year_2,moment_match = TRUE),loo(model_vill_year_7,moment_match = TRUE))
# loo_compare(loo(model_vill_year_3,moment_match = TRUE),loo(model_vill_year_8,moment_match = TRUE))

# Check residuals
model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_1)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_1)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
par(mfrow=c(4,3))
plotResiduals(model.check,data_vill$vax_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
testZeroInflation(model.check)
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
# good

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_2)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_2)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last2yearMean[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last2yearMean_neighbours[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last2yearMean_notNeighbours[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*2))])) 
testZeroInflation(model.check)
# good

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_3)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*3))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_3)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last3yearMean[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last3yearMean_neighbours[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last3yearMean_notNeighbours[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
testZeroInflation(model.check)
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*3))])) 
# good

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_4)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_4)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
testZeroInflation(model.check)
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_5)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_5)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
testZeroInflation(model.check)
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_6)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_6)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last_year_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
testZeroInflation(model.check)
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_7)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_7)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last2yearMean_campaign[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last2yearMean_neighbours_campaign[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last2yearMean_notNeighbours_campaign[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*2))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*2))])) 
testZeroInflation(model.check)
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_8)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)*3))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_8)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last3yearMean_campaign[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last3yearMean_neighbours_campaign[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last3yearMean_notNeighbours_campaign[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)*3))],quantreg = T) 
testZeroInflation(model.check)
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)*3))])) 
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_9)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_9)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last_year_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$incidence_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
testZeroInflation(model.check)
# ok

model.check <- createDHARMa(
  simulatedResponse = t(posterior_predict(model_vill_year_10)),
  observedResponse = data_vill$cases[-c(1:(nrow(cases_vill)))],
  fittedPredictedResponse = apply(t(posterior_epred(model_vill_year_10)), 1, median),
  integerResponse = TRUE)
plot(model.check,quantreg=T)
plotResiduals(model.check,data_vill$vax_last_year_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_neighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$vax_last_year_notNeighbours_campaign[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year_neighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$log_incidence_last_year_notNeighbours[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$dog_density[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,data_vill$HDR[-c(1:(nrow(cases_vill)))],quantreg = T) 
plotResiduals(model.check,as.factor(data_vill$village[-c(1:(nrow(cases_vill)))])) 
testZeroInflation(model.check)
# ok




#Parameter tables
#---------

pars <- data.frame("Intercept"=rep("",4),
                   "Village last year"=rep("",4),
                   "Borders last year"=rep("",4),
                   "Non-bordering villages last year"=rep("",4),
                   "Village mean last 2 years"=rep("",4),
                   "Borders mean last 2 years"=rep("",4),
                   "Non-bordering villages mean last 2 years"=rep("",4),
                   "Village mean last 3 years"=rep("",4),
                   "Borders mean last 3 years"=rep("",4),
                   "Non-bordering villages mean last 3 years"=rep("",4),
                   "Village ln(Cases/dog) last year"=rep("",4),
                   "Borders ln(Cases/dog) last year"=rep("",4),
                   "Non-bordering villages ln(Cases/dog) last year"=rep("",4),
                   "Village ln(dog density)"=rep("",4),
                   "Human:dog ratio"=rep("",4),
                   "Village random effect standard deviation"=rep("",4),
                   "Shape"=rep("",4),
                   check.names = F)
model_summary <- summary(model_vill_year_1)
pars1 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars1 <- paste0(pars1[,1]," (",pars1[,2]," - ",pars1[,3],")")
pars[1, c("Intercept","Village last year","Borders last year","Non-bordering villages last year","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars1
model_summary <- summary(model_vill_year_2)
pars2 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars2 <- paste0(pars2[,1]," (",pars2[,2]," - ",pars2[,3],")")
pars[2, c("Intercept","Village mean last 2 years","Borders mean last 2 years","Non-bordering villages mean last 2 years","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars2
model_summary <- summary(model_vill_year_3)
pars3 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars3 <- paste0(pars3[,1]," (",pars3[,2]," - ",pars3[,3],")")
pars[3, c("Intercept","Village mean last 3 years","Borders mean last 3 years","Non-bordering villages mean last 3 years","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars3
model_summary <- summary(model_vill_year_5)
pars4 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars4 <- paste0(pars4[,1]," (",pars4[,2]," - ",pars4[,3],")")
pars[4, c("Intercept","Village last year","Borders last year","Non-bordering villages last year",
          "Village ln(Cases/dog) last year","Borders ln(Cases/dog) last year","Non-bordering villages ln(Cases/dog) last year",
          "Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars4

write.csv(pars,"output/annual_village_mean_annual_vax_model_pars.csv",row.names=F)




model_summary <- summary(model_vill_year_6)
pars5 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars5 <- paste0(pars5[,1]," (",pars5[,2]," - ",pars5[,3],")")
pars[1, c("Intercept","Village last year","Borders last year","Non-bordering villages last year","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars5
model_summary <- summary(model_vill_year_7)
pars6 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars6 <- paste0(pars6[,1]," (",pars6[,2]," - ",pars6[,3],")")
pars[2, c("Intercept","Village mean last 2 years","Borders mean last 2 years","Non-bordering villages mean last 2 years","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars6
model_summary <- summary(model_vill_year_8)
pars7 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars7 <- paste0(pars7[,1]," (",pars7[,2]," - ",pars7[,3],")")
pars[3, c("Intercept","Village mean last 3 years","Borders mean last 3 years","Non-bordering villages mean last 3 years","Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars7
model_summary <- summary(model_vill_year_10)
pars8 <- round(rbind(model_summary$fixed,model_summary$random$village,model_summary$spec_pars),2)[,c(1,3:4)]
pars8 <- paste0(pars8[,1]," (",pars8[,2]," - ",pars8[,3],")")
pars[4, c("Intercept","Village last year","Borders last year","Non-bordering villages last year",
          "Village ln(Cases/dog) last year","Borders ln(Cases/dog) last year","Non-bordering villages ln(Cases/dog) last year",
          "Village ln(dog density)","Human:dog ratio","Village random effect standard deviation","Shape")] <- pars8

write.csv(pars,"output/annual_village_campaign_vax_model_pars.csv",row.names=F)




#Plot
#---------

range(data_vill$incidence_last_year,na.rm=T) # max 0.105
range(data_dist$incidence_last_year,na.rm=T) # max 0.009
incidences <- seq(0,0.009,0.003)

range(data_vill$vax_last_year,na.rm=T) # 0-1
range(data_vill$vax_last_year_neighbours,na.rm=T) # 0-0.37
range(data_vill$vax_last_year_notNeighbours,na.rm=T) # 0-0.37



# Plot predictions (vaccination only models)
#---------------------

pdf("Figs/AnnualVillageModelsVax.pdf",width=3.5, height=3)
par(mar=c(3.5,3.5,1,1))

coverage <- seq(0,1,length.out=100)
new_data <- data.frame(dogs=1,
                       vax_last_year_campaign=coverage, vax_last_year_neighbours_campaign=coverage, vax_last_year_notNeighbours_campaign=coverage,
                       vax_last2yearMean_campaign=coverage, vax_last2yearMean_neighbours_campaign=coverage, vax_last2yearMean_notNeighbours_campaign=coverage,
                       vax_last3yearMean_campaign=coverage, vax_last3yearMean_neighbours_campaign=coverage, vax_last3yearMean_notNeighbours_campaign=coverage,
                       log_dog_density=log(mean(data_vill$dog_density)), HDR=mean(data_vill$HDR))
preds <- posterior_epred(model_vill_year_6, newdata = new_data, re_formula = NA)*1000
preds_upper1 <- matrix(apply(preds, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_lower1 <- matrix(apply(preds, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds1 <- apply(preds, 2, mean)
preds <- posterior_epred(model_vill_year_7, newdata = new_data, re_formula = NA)*1000
preds_upper2 <- matrix(apply(preds, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_lower2 <- matrix(apply(preds, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds2 <- apply(preds, 2, mean)
preds <- posterior_epred(model_vill_year_8, newdata = new_data, re_formula = NA)*1000
preds_upper3 <- matrix(apply(preds, 2, quantile,probs=0.975),nrow=length(coverage),ncol=1)
preds_lower3 <- matrix(apply(preds, 2, quantile,probs=0.025),nrow=length(coverage),ncol=1)
preds3 <- apply(preds, 2, mean)

par(fig=c(0,1,0,1))

col_pal <- viridis(3)
plot(NA,ylim=c(0,max(preds_upper1,preds_upper2,preds_upper3)),xlim=c(0,1),bty="l",
     ylab="",xlab="",cex.lab=1.1,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in village this year",side=2,line=1.5,cex=cex.lab)
mtext("Campaign coverage at all scales\nover previous years",side=1,line=2,cex=cex.lab)
polygon(c(coverage,rev(coverage)),c(preds_lower1[,1],rev(preds_upper1[,1])),col=alpha(col_pal[1],0.1),border=NA)
polygon(c(coverage,rev(coverage)),c(preds_lower2[,1],rev(preds_upper2[,1])),col=alpha(col_pal[2],0.1),border=NA)
polygon(c(coverage,rev(coverage)),c(preds_lower3[,1],rev(preds_upper3[,1])),col=alpha(col_pal[3],0.1),border=NA)
lines(preds1~coverage,col=col_pal[1],lwd=2,lty=1)
lines(preds2~coverage,col=col_pal[2],lwd=2,lty=1)
lines(preds3~coverage,col=col_pal[3],lwd=2,lty=1)

legend("topright",c("year","2 years","3 years"),lty=1,col=col_pal,
       title="Mean campaign coverage\nover previous:",
       title.cex=0.8,cex=0.8,bty="n",lwd=2,yjust=1,title.adj = -0.01)

dev.off()



# Plot predictions  (campaign coverage and cases models)
#---------------------

pdf("Figs/AnnualVillageModelsCases&Vax.pdf",width=4, height=3)

par(fig=c(0,1,0,1))
par(mar=c(3.5,3.5,1,1))
coverage <- seq(0,1,length.out=100)
incidence <- c(0,0.003,0.006,0.009)
data_vill_incidence_adjust_dist <- 0.5*min(data_dist$incidence_last_year[which(data_dist$incidence_last_year>0)])
new_data <- data.frame(dogs=1,
                       vax_last_year_campaign=rep(coverage,length(incidence)), 
                       vax_last_year_neighbours_campaign=rep(coverage,length(incidence)), 
                       vax_last_year_notNeighbours_campaign=rep(coverage,length(incidence)), 
                       log_incidence_last_year=rep(log(incidence+data_vill_incidence_adjust_dist),each=length(coverage)),
                       log_incidence_last_year_neighbours=rep(log(incidence+data_vill_incidence_adjust_dist),each=length(coverage)),
                       log_incidence_last_year_notNeighbours=rep(log(incidence+data_vill_incidence_adjust_dist),each=length(coverage)),
                       log_dog_density=log(mean(data_vill$dog_density)),
                       HDR = mean(data_vill$HDR))
preds <- posterior_epred(model_vill_year_10, newdata = new_data, re_formula = NA)*1000
preds_upper <- matrix(apply(preds, 2, quantile,probs=0.975),nrow=length(coverage),ncol=length(incidence))
preds_lower <- matrix(apply(preds, 2, quantile,probs=0.025),nrow=length(coverage),ncol=length(incidence))
preds <- matrix(apply(preds, 2, mean),nrow=length(coverage),ncol=length(incidence))
cols <- viridis(length(incidence))
plot(NA,ylim=c(0,max(preds_upper)),xlim=c(0,1),bty="l",
     ylab="",xlab="",cex.lab=1.1,axes=F)
axis(2,cex.axis=cex.axis,padj=1)
axis(1,cex.axis=cex.axis,padj=-1.5)
box(bty="l")
mtext("Cases/1,000 dogs in village this year",side=2,line=1.5,cex=cex.lab)
mtext("Campaign coverage at all scales\nlast year",side=1,line=2,cex=cex.lab)
for(i in 1:length(incidence)){
  polygon(c(coverage,rev(coverage)),c(preds_lower[,i],rev(preds_upper[,i])),col=alpha(cols[i],0.1),border=NA)
}
for(i in 1:length(incidence)){lines(preds[,i]~coverage,col=cols[i],lwd=2,lty=1)}

legend("topright",legend=incidences*1000,lty=1,col=cols,
       title="Cases/1,000 dogs at all scales\nover previous year:",
       title.cex=0.8,cex=0.8,bty="n",lwd=2,yjust=1,title.adj = -0.01)

dev.off()


