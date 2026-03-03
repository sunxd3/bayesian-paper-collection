
rm(list=ls())

library(viridis)
library(rgdal)
library(rgeos)
library(raster)
library(stringr)

set.seed(0)


## Load data
#________________

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




# Prep data for models
#________________

# Alternative meaned annual vax estimate
vax_dist_annual_alt <- rep(NA,length(2002:2022))
vax_vill_annual_alt <- matrix(NA,nrow=nrow(vax_vill),ncol=length(2002:2022))
for(i in 1:length(2002:2022)){
  vax_dist_annual_alt[i] <- mean(vax_dist[(1:12) + (i-1)*12])
  vax_vill_annual_alt[,i] <- rowMeans(vax_vill[,(1:12) + (i-1)*12])
}

# Annual cases in district/village
cases_dist_annual <- rep(NA,length(2002:2022))
cases_vill_annual <- matrix(NA,nrow=nrow(cases_vill),ncol=length(2002:2022))
for(i in 1:length(2002:2022)){
  cases_dist_annual[i] <- sum(cases_dist[(1:12) + (i-1)*12])
  cases_vill_annual[,i] <- rowSums(cases_vill[,(1:12) + (i-1)*12])
}

# Annual dogs & humans
dogs_dist <- colSums(dogs)
humans_dist <- colSums(humans)
dogs_dist_annual <- humans_dist_annual <- rep(NA,length(2002:2022))
dogs_vill_annual <- humans_vill_annual<- matrix(NA,nrow=nrow(cases_vill),ncol=length(2002:2022))
for(i in 1:length(2002:2022)){
  dogs_dist_annual[i] <- round(mean(dogs_dist[(1:12) + (i-1)*12]))
  dogs_vill_annual[,i] <- rowMeans(dogs[,(1:12) + (i-1)*12])
  humans_dist_annual[i] <- round(mean(humans_dist[(1:12) + (i-1)*12]))
  humans_vill_annual[,i] <- rowMeans(humans[,(1:12) + (i-1)*12])
}

# Get shared borders with other villages, the park and Mara
load("output/village_borders.Rdata")

# Cases last year at distant spatial scales
# Assume average of neighbour case rates on park border
# Assume 0.01 cases/dog on Mara border
str(cases_vill_annual)
case_rate_vill_neighbours_lastYear <- case_rate_vill_notNeighbours_lastYear <- matrix(NA,nrow=nrow(cases_vill_annual),ncol=ncol(cases_vill_annual))
case_rate_vill <- cases_vill_annual/dogs_vill_annual
for(i in 2:length(cases_dist_annual)){
  case_rate_vill_notNeighbours_lastYear[,i] <- rowSums(not_bordering*matrix(cases_vill_annual[,i-1],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering)))/rowSums(not_bordering*matrix(dogs[,i-1],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering)))
  for(vill in 1:nrow(SD_vill)){
    neighbours <- which(bordering[vill,])
    case_rates <- case_rate_vill[,(i-1)] 
    case_rates_w_park_mara <- c(case_rates,mean(case_rates[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))]]),0.01/12) 
    neighbour_case_rates_last_year <-  case_rates_w_park_mara[neighbours]
    weights_neighbours <- borders_prop[vill,neighbours] #weight by border length 
    case_rate_vill_neighbours_lastYear[vill,i] <- sum(neighbour_case_rates_last_year*weights_neighbours)
  }
}

#Previous vaccination
vax_dist_last3yearMean <- vax_dist_last2yearMean <- vax_dist_last3yearMean_campaign <- vax_dist_last2yearMean_campaign <- rep(NA,length(vax_dist_annual))
vax_vill_neighbours_lastYear <- vax_vill_notNeighbours_lastYear <- 
  vax_vill_last3yearMean <- vax_vill_neighbours_last3yearMean <- vax_vill_notNeighbours_last3yearMean <-
  vax_vill_last2yearMean <- vax_vill_neighbours_last2yearMean <- vax_vill_notNeighbours_last2yearMean <-
  vax_vill_neighbours_lastYear_campaign <- vax_vill_notNeighbours_lastYear_campaign <- 
  vax_vill_last3yearMean_campaign <- vax_vill_neighbours_last3yearMean_campaign <- vax_vill_notNeighbours_last3yearMean_campaign <-
  vax_vill_last2yearMean_campaign <- vax_vill_neighbours_last2yearMean_campaign <- vax_vill_notNeighbours_last2yearMean_campaign <-
  matrix(NA,nrow=nrow(vax_vill_annual),ncol=ncol(vax_vill_annual))
for(i in 2:length(vax_dist_annual)){
  
  for(vill in 1:nrow(SD_vill)){
    # neighbours 
    neighbours <- which(bordering[vill,])
    notNeighbours <- which(not_bordering[vill,])
    
    #coverages for subpopulations
    coverages <- vax_vill_annual_alt[,(i-1)] 
    coverages_w_park_mara <- c(coverages,mean(coverages[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))]]),0.09) # assume average of neighbour coverages in park and 9% coverage in Mara 
    neighbour_coverages <-  coverages_w_park_mara[neighbours]
    notNeighbour_coverages <- coverages[notNeighbours]
    coverages_campaign <- vax_vill_annual[,(i-1)] 
    coverages_w_park_mara_campaign <- c(coverages_campaign,mean(coverages_campaign[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))]]),0.05) # assume average of neighbour coverages in park and 5% coverage in Mara (not 9% because not including carry-over)
    neighbour_coverages_campaign <-  coverages_w_park_mara_campaign[neighbours]
    notNeighbour_coverages_campaign <- coverages_campaign[notNeighbours]
    
    # Weights
    weights_neighbours <-  borders_prop[vill,neighbours] #weight by border length 
    notNeighbour_dogs <- dogs_vill_annual[notNeighbours,(i-1)]
    weights_notNeighbours <- notNeighbour_dogs/sum(notNeighbour_dogs) #weight by population
    
    vax_vill_neighbours_lastYear[vill,i] <- sum(weights_neighbours*neighbour_coverages) 
    vax_vill_notNeighbours_lastYear[vill,i] <- sum(weights_notNeighbours*notNeighbour_coverages)
    vax_vill_neighbours_lastYear_campaign[vill,i] <- sum(weights_neighbours*neighbour_coverages_campaign) 
    vax_vill_notNeighbours_lastYear_campaign[vill,i] <- sum(weights_notNeighbours*notNeighbour_coverages_campaign)
  }
  
}
for(i in 3:length(vax_dist_annual)){
  vax_dist_last2yearMean[i] <- mean(vax_dist_annual_alt[(i-2):(i-1)])
  vax_vill_last2yearMean[,i] <- rowMeans(vax_vill_annual_alt[,(i-2):(i-1)])
  vax_dist_last2yearMean_campaign[i] <- mean(vax_dist_annual[(i-2):(i-1)])
  vax_vill_last2yearMean_campaign[,i] <- rowMeans(vax_vill_annual[,(i-2):(i-1)])
  
  for(vill in 1:nrow(SD_vill)){
    # neighbours 
    neighbours <- which(bordering[vill,])
    notNeighbours <- which(not_bordering[vill,])
    
    #coverages for subpopulations
    coverages <- vax_vill_annual_alt[,(i-2):(i-1)] 
    coverages_w_park_mara <- rbind(coverages,colMeans(coverages[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.09) # assume average of neighbour coverages in park and 9% coverage in Mara
    neighbour_coverages <-  coverages_w_park_mara[neighbours,]
    notNeighbour_coverages <- coverages[notNeighbours,]
    coverages_campaign <- vax_vill_annual[,(i-2):(i-1)] 
    coverages_w_park_mara_campaign <- rbind(coverages_campaign,colMeans(coverages_campaign[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.05) # assume average of neighbour coverages in park and 5% coverage in Mara (not 9% because not including carry-over)
    neighbour_coverages_campaign <-  coverages_w_park_mara_campaign[neighbours,]
    notNeighbour_coverages_campaign <- coverages_campaign[notNeighbours,]
    
    # Weights
    weights_neighbours <-  matrix(rep(borders_prop[vill,neighbours],2),ncol=2) #weight by border length 
    notNeighbour_dogs <- dogs_vill_annual[notNeighbours,(i-2):(i-1)]
    weights_notNeighbours <- notNeighbour_dogs/matrix(rep(colSums((notNeighbour_dogs)),each=nrow(notNeighbour_dogs)),ncol=2) #weight by population
    
    vax_vill_neighbours_last2yearMean[vill,i] <- mean(colSums(neighbour_coverages*weights_neighbours))
    vax_vill_notNeighbours_last2yearMean[vill,i] <- mean(colSums(notNeighbour_coverages*weights_notNeighbours))
    vax_vill_neighbours_last2yearMean_campaign[vill,i] <- mean(colSums(neighbour_coverages_campaign*weights_neighbours))
    vax_vill_notNeighbours_last2yearMean_campaign[vill,i] <- mean(colSums(notNeighbour_coverages_campaign*weights_notNeighbours))
    
  }  
}
for(i in 4:length(vax_dist_annual)){
  vax_dist_last3yearMean[i] <- mean(vax_dist_annual_alt[(i-3):(i-1)])
  vax_vill_last3yearMean[,i] <- rowMeans(vax_vill_annual_alt[,(i-3):(i-1)])
  vax_dist_last3yearMean_campaign[i] <- mean(vax_dist_annual[(i-3):(i-1)])
  vax_vill_last3yearMean_campaign[,i] <- rowMeans(vax_vill_annual[,(i-3):(i-1)])
  
  for(vill in 1:nrow(SD_vill)){
    # neighbours 
    neighbours <- which(bordering[vill,])
    notNeighbours <- which(not_bordering[vill,])
    
    #coverages for subpopulations
    coverages <- vax_vill_annual_alt[,(i-3):(i-1)] 
    coverages_w_park_mara <- rbind(coverages,colMeans(coverages[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.09) # assume average of neighbour coverages in park and 9% coverage in Mara
    neighbour_coverages <-  coverages_w_park_mara[neighbours,]
    notNeighbour_coverages <- coverages[notNeighbours,]
    coverages_campaign <- vax_vill_annual[,(i-3):(i-1)] 
    coverages_w_park_mara_campaign <- rbind(coverages_campaign,colMeans(coverages_campaign[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.05) # assume average of neighbour coverages in park and 9% coverage in Mara
    neighbour_coverages_campaign <-  coverages_w_park_mara_campaign[neighbours,]
    notNeighbour_coverages_campaign <- coverages_campaign[notNeighbours,]
    
    # Weights
    weights_neighbours <-  matrix(rep(borders_prop[vill,neighbours],3),ncol=3) #weight by border length 
    notNeighbour_dogs <- dogs_vill_annual[notNeighbours,(i-3):(i-1)]
    weights_notNeighbours <- notNeighbour_dogs/matrix(rep(colSums((notNeighbour_dogs)),each=nrow(notNeighbour_dogs)),ncol=3) #weight by population
    
    vax_vill_neighbours_last3yearMean[vill,i] <- mean(colSums(neighbour_coverages*weights_neighbours))
    vax_vill_notNeighbours_last3yearMean[vill,i] <- mean(colSums(notNeighbour_coverages*weights_notNeighbours))
    vax_vill_neighbours_last3yearMean_campaign[vill,i] <- mean(colSums(neighbour_coverages_campaign*weights_neighbours))
    vax_vill_notNeighbours_last3yearMean_campaign[vill,i] <- mean(colSums(notNeighbour_coverages_campaign*weights_notNeighbours))
    
  }  
}



data_dist <- data.frame(cases = c(cases_dist_annual), 
                        vax = c(vax_dist_annual_alt),
                        vax_campaign = c(vax_dist_annual),
                        vax_last_year = c(NA,vax_dist_annual_alt[1:(length(vax_dist_annual_alt)-1)]),
                        vax_last_year_campaign = c(NA,vax_dist_annual[1:(length(vax_dist_annual)-1)]),
                        vax_last2yearMean = vax_dist_last2yearMean,
                        vax_last3yearMean = vax_dist_last3yearMean,
                        vax_last2yearMean_campaign = vax_dist_last2yearMean_campaign,
                        vax_last3yearMean_campaign = vax_dist_last3yearMean_campaign,
                        cases_last_year = c(NA,cases_dist_annual[1:(length(cases_dist_annual)-1)]),
                        year = 2002:2022,
                        dogs = dogs_dist_annual,
                        dog_density = (dogs_dist_annual/sum(SD_vill$cells_occ)),
                        humans = humans_dist_annual,
                        human_density = (humans_dist_annual/sum(SD_vill$cells_occ)),
                        incidence = c(cases_dist_annual/dogs_dist_annual),
                        incidence_last_year = c(NA,cases_dist_annual[1:(length(cases_dist_annual)-1)]/dogs_dist_annual[1:(length(dogs_dist_annual)-1)]))
data_dist_incidence_adjust <- 0.5*min(data_dist$incidence_last_year[which(data_dist$incidence_last_year>0)])
data_dist$log_incidence_last_year <- log(data_dist$incidence_last_year + data_dist_incidence_adjust)
data_dist$log_human_density <- log(data_dist$human_density)
data_dist$log_human_density_sq<- data_dist$log_human_density^2
data_dist$log_dog_density <- log(data_dist$dog_density)


data_vill <- data.frame(cases = c(cases_vill_annual), 
                        vax = c(vax_vill_annual_alt),
                        vax_campaign = c(vax_vill_annual),
                        vax_last_year = c(rep(NA,nrow(vax_vill)),c(vax_vill_annual_alt[,1:(ncol(vax_vill_annual_alt)-1)])),
                        vax_last_year_campaign = c(rep(NA,nrow(vax_vill)),c(vax_vill_annual[,1:(ncol(vax_vill_annual)-1)])),
                        vax_last_year_notNeighbours = c(vax_vill_notNeighbours_lastYear),
                        vax_last_year_neighbours = c(vax_vill_neighbours_lastYear),
                        vax_last_year_notNeighbours_campaign = c(vax_vill_notNeighbours_lastYear_campaign),
                        vax_last_year_neighbours_campaign = c(vax_vill_neighbours_lastYear_campaign),
                        vax_last2yearMean = c(vax_vill_last2yearMean),
                        vax_last2yearMean_notNeighbours = c(vax_vill_notNeighbours_last2yearMean),
                        vax_last2yearMean_neighbours = c(vax_vill_neighbours_last2yearMean),
                        vax_last3yearMean = c(vax_vill_last3yearMean),
                        vax_last3yearMean_notNeighbours = c(vax_vill_notNeighbours_last3yearMean),
                        vax_last3yearMean_neighbours = c(vax_vill_neighbours_last3yearMean),
                        vax_last2yearMean_campaign = c(vax_vill_last2yearMean_campaign),
                        vax_last2yearMean_notNeighbours_campaign = c(vax_vill_notNeighbours_last2yearMean_campaign),
                        vax_last2yearMean_neighbours_campaign = c(vax_vill_neighbours_last2yearMean_campaign),
                        vax_last3yearMean_campaign = c(vax_vill_last3yearMean_campaign),
                        vax_last3yearMean_notNeighbours_campaign = c(vax_vill_notNeighbours_last3yearMean_campaign),
                        vax_last3yearMean_neighbours_campaign = c(vax_vill_neighbours_last3yearMean_campaign),
                        village = rep(rownames(cases_vill),ncol(cases_vill_annual)),
                        year = rep(2002:2022,each=nrow(vax_vill)),
                        dogs = c(dogs_vill_annual),
                        dog_density = (c(dogs_vill_annual)/rep(SD_vill$cells_occ,ncol(dogs_vill_annual))),
                        humans = c(humans_vill_annual),
                        human_density = (c(humans_vill_annual)/rep(SD_vill$cells_occ,ncol(humans_vill_annual))),
                        incidence = c(cases_vill_annual/dogs_vill_annual),
                        incidence_last_year = c(rep(NA,nrow(vax_vill_annual)),
                                                c(cases_vill_annual[,1:(ncol(cases_vill_annual)-1)]/dogs_vill_annual[,1:(ncol(dogs_vill_annual)-1)])),
                        incidence_last_year_neighbours = c(case_rate_vill_neighbours_lastYear),
                        incidence_last_year_notNeighbours = c(case_rate_vill_notNeighbours_lastYear))
data_vill_incidence_adjust_dist <- 0.5*min(data_dist$incidence_last_year[which(data_dist$incidence_last_year>0)])
data_vill$log_incidence_last_year <- log(data_vill$incidence_last_year + data_vill_incidence_adjust_dist)
data_vill$log_incidence_last_year_neighbours <- log(data_vill$incidence_last_year_neighbours + data_vill_incidence_adjust_dist)
data_vill$log_incidence_last_year_notNeighbours <- log(data_vill$incidence_last_year_notNeighbours + data_vill_incidence_adjust_dist)
data_vill$log_human_density <- log(data_vill$human_density)
data_vill$log_dog_density <- log(data_vill$dog_density)
data_vill$HDR <- data_vill$humans/data_vill$dogs

# output data
write.csv(data_vill,"output/incidence_coverage_model_data_village_annual.csv",row.names = F)
write.csv(data_dist,"output/incidence_coverage_model_data_district_annual.csv",row.names = F)



