
rm(list=ls())

library(raster)
library(rgeos)
library(rgdal)
library(spdep)
library(stringr)



## Load data
#______________________

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

# Case numbers
cases_dist <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_2002-01-01_2022-12-31.csv",header=F))
cases_vill <- as.matrix(read.csv("output/Serengeti_monthly_rabid_dogs_village_2002-01-01_2022-12-31.csv",header=F,row.names = 1))

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




# Prep data for models
#______________________

# Get shared borders with other villages, the park and Mara
borders <- matrix(0,nrow=nrow(SD_vill),ncol=nrow(SD_vill)+2,dimnames = list(SD_vill$Vill_2012,c(SD_vill$Vill_2012,"Park","Mara")))
for(i in 1:nrow(SD_vill)){
  polygon <- SD_vill_gridded[i,]
  polygons <- SD_vill_gridded[-i,]
  par(mar=c(0,0,0,0))
  plot(polygon)
  text(SD_vill_gridded,SD_vill_gridded$Vill_2012,col="red")
  intersections <- gIntersection(polygon,polygons,byid = T)
  plot(intersections,col=2:(length(intersections)+1),lwd=2,add=T)
  lengths <- gLength(intersections,byid = T)
  borders[i,as.numeric(word(names(lengths),2,sep=" "))+1] <- lengths
  intersection <- gIntersection(polygon,Serengeti,byid = T)
  if(!is.null(intersection)){
    plot(intersection,col="darkorange",lwd=2,add=T)
    borders[i,which(colnames(borders)=="Park")] <- gLength(intersection)}
  borders[i,which(colnames(borders)=="Mara")] <- round(gLength(polygon) - sum(borders[i,]))
}
borders[,c("Park","Mara")]

# Bits of park assigned to Mara:Bonchugu, Iharara, Merenga (this village actually has both Mara & park - some park assigned to Mara, but not bad enough to bother with), Motukeri, Robanda, Singisi
# Except for Nyiberekera and Merenga, assign any with both park and Mara to park
borders[which(borders[,"Park"]>0 & borders[,"Mara"]>0 & !rownames(borders)%in%c("Merenga","Nyiberekera")),"Park"] <- borders[which(borders[,"Park"]>0 & borders[,"Mara"]>0 & !rownames(borders)%in%c("Merenga","Nyiberekera")),"Park"] + borders[which(borders[,"Park"]>0 & borders[,"Mara"]>0 & !rownames(borders)%in%c("Merenga","Nyiberekera")),"Mara"]
borders[which(borders[,"Park"]>0 & borders[,"Mara"]>0 & !rownames(borders)%in%c("Merenga","Nyiberekera")),"Mara"] <- 0
borders_prop <- borders/rowSums(borders)
bordering <- borders>0
not_bordering <- !bordering[,-c((ncol(bordering)-1):ncol(bordering))]
not_bordering[cbind(1:nrow(bordering),1:nrow(bordering))] <- FALSE
hist(rowSums(bordering))
mean(rowSums(bordering))
median(rowSums(bordering))
range(rowSums(bordering))
mean(rowSums(!bordering)-1)
range(rowSums(!bordering)-1)

save(borders, borders_prop, bordering, not_bordering,file="output/village_borders.Rdata")

# Average cases in previous x months
# roughly 75% of inc periods in a month, 90% within 2 months, 95% within 3 months (maybe use average over 2 months?)
# Assume average of neighbour case rates on park border
# Assume 0.01/12=0.00083 cases/dog on Mara border
str(cases_vill)
case_rate_dist_last2monthMean <- rep(NA,length(cases_dist))
case_rate_vill_last2monthMean <- case_rate_vill_neighbours_last2monthMean <- case_rate_vill_notNeighbours_last2monthMean <- 
  matrix(NA,nrow=nrow(cases_vill),ncol=ncol(cases_vill))
case_rate_vill <- cases_vill/dogs
case_rate_vill_border <- rbind(cases_vill/dogs, rep(0,ncol(cases_vill)), rep(0.01/12,ncol(cases_vill)))
for(i in 3:length(cases_dist)){
  case_rate_dist_last2monthMean[i] <- mean(cases_dist[(i-2):(i-1)]/colSums(dogs)[(i-2):(i-1)])
  case_rate_vill_last2monthMean[,i] <- rowMeans(cases_vill[,(i-2):(i-1)]/dogs[,(i-2):(i-1)])
  case_rate_vill_notNeighbours_last2monthMean[,i] <- (rowSums(not_bordering*matrix(cases_vill[,i-2],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering)))/rowSums(not_bordering*matrix(dogs[,i-2],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering))) +
                                                        rowSums(not_bordering*matrix(cases_vill[,i-1],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering)))/rowSums(not_bordering*matrix(dogs[,i-1],byrow=T,ncol=ncol(not_bordering),nrow=nrow(not_bordering))))/2
  for(vill in 1:nrow(SD_vill)){
    neighbours <- which(bordering[vill,])
    case_rates <- case_rate_vill[,(i-2):(i-1)] 
    case_rates_w_park_mara <- rbind(case_rates,colMeans(case_rates[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.01/12) 
    neighbour_case_rates_last2months <-  case_rates_w_park_mara[neighbours,]
    weights_neighbours <-  matrix(rep(borders_prop[vill,neighbours],2),ncol=2) #weight by border length 
    case_rate_vill_neighbours_last2monthMean[vill,i] <- mean(colSums(neighbour_case_rates_last2months*weights_neighbours))
  }
}

# Average vaccination coverage in previous x months
vax_dist_last2monthMean <- susc_dist_last2monthHarmMean <- susc_dist_last2monthGeoMean <- var_cov <- sd_cov <- rep(NA,length(vax_dist))
vax_vill_last2monthMean <- vax_vill_neighbours_last2monthMean <- vax_vill_notNeighbours_last2monthMean <- var_cov_neighbours <- var_cov_notNeighbours <- sd_cov_neighbours <- sd_cov_notNeighbours <- matrix(NA,nrow=nrow(vax_vill),ncol=ncol(vax_vill))
immune_dist_last2monthMean <- var_immune <- sd_immune <- rep(NA,length(immune_dist))
immune_vill_last2monthMean <- immune_vill_neighbours_last2monthMean <- immune_vill_notNeighbours_last2monthMean <- var_immune_neighbours <- var_immune_notNeighbours <- sd_immune_neighbours <- sd_immune_notNeighbours <- matrix(NA,nrow=nrow(immune_vill),ncol=ncol(immune_vill))
for(i in 3:length(vax_dist)){
  
  vax_dist_last2monthMean[i] <- mean(vax_dist[(i-2):(i-1)])
  vax_vill_last2monthMean[,i] <- rowMeans(vax_vill[,(i-2):(i-1)])
  immune_dist_last2monthMean[i] <- mean(immune_dist[(i-2):(i-1)])
  immune_vill_last2monthMean[,i] <- rowMeans(immune_vill[,(i-2):(i-1)])
  
  # Weighted mean variance
  village_coverages_last2months <- vax_vill[,(i-2):(i-1)]
  district_coverage_last2months <- matrix(rep(vax_dist[(i-2):(i-1)],each=nrow(SD_vill)),ncol=2)
  weights <- dogs[,(i-2):(i-1)]/matrix(rep(colSums(dogs[,(i-2):(i-1)]),each=nrow(SD_vill)),ncol=2)# Proportion dog pop in each village
  village_immunity_last2months <- immune_vill[,(i-2):(i-1)]
  district_immunity_last2months <- matrix(rep(immune_dist[(i-2):(i-1)],each=nrow(SD_vill)),ncol=2)

  for(vill in 1:nrow(SD_vill)){
    
    # neighbours 
    neighbours <- which(bordering[vill,])
    notNeighbours <- which(not_bordering[vill,])
    
    #coverages/immunity for subpopulations
    coverages <- vax_vill[,(i-2):(i-1)] 
    coverages_w_park_mara <- rbind(coverages,colMeans(coverages[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.09) # assume average of neighbour coverages in park and 9% coverage in Mara
    neighbour_coverages_last2months <-  coverages_w_park_mara[neighbours,]
    notNeighbour_coverages_last2months <- coverages[notNeighbours,]
    immune <- immune_vill[,(i-2):(i-1)] 
    immune_w_park_mara <-  rbind(immune,colMeans(immune[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))],,drop=F]),0.05) # assume average of neighbour immune in park and 5% immunity in Mara
    neighbour_immune_last2months <-  immune_w_park_mara[neighbours,]
    notNeighbour_immune_last2months <- immune[notNeighbours,]
    
    # Weights
    weights_neighbours <-  matrix(rep(borders_prop[vill,neighbours],2),ncol=2) #weight by border length 
    notNeighbour_dogs <- dogs[notNeighbours,(i-2):(i-1)]
    weights_notNeighbours <- notNeighbour_dogs/matrix(rep(colSums((notNeighbour_dogs)),each=nrow(notNeighbour_dogs)),ncol=2) #weight by population
    
    vax_vill_neighbours_last2monthMean[vill,i] <- mean(colSums(neighbour_coverages_last2months*weights_neighbours))
    vax_vill_notNeighbours_last2monthMean[vill,i] <- mean(colSums(notNeighbour_coverages_last2months*weights_notNeighbours))
    immune_vill_neighbours_last2monthMean[vill,i] <- mean(colSums(neighbour_immune_last2months*weights_neighbours))
    immune_vill_notNeighbours_last2monthMean[vill,i] <- mean(colSums(notNeighbour_immune_last2months*weights_notNeighbours))
    
  }
  
}

# Combine district level data
data_dist <- data.frame(cases = c(cases_dist), 
                        vax = c(vax_dist),
                        vax_last_month = c(NA,vax_dist[1:(length(vax_dist)-1)]),
                        vax_last2monthMean = vax_dist_last2monthMean,
                        immune = c(immune_dist),
                        immune_last_month = c(NA,immune_dist[1:(length(immune_dist)-1)]),
                        immune_last2monthMean = immune_dist_last2monthMean,
                        cases_last_month = c(NA,cases_dist[1:(length(cases_dist)-1)]),
                        case_rate_last_month = c(NA,cases_dist[1:(length(cases_dist)-1)]/colSums(dogs)[1:(length(cases_dist)-1)]),
                        case_rate_last2monthMean = case_rate_dist_last2monthMean,
                        month = 1:nrow(cases_dist),
                        dogs = colSums(dogs),
                        humans = colSums(humans),
                        dog_density = colSums(dogs)/sum(SD_vill$cells_occ),
                        human_density = colSums(humans)/sum(SD_vill$cells_occ),
                        incidence = c(cases_dist/colSums(dogs)))
data_dist_case_rate_adjust <-  0.5*min(data_dist$case_rate_last2monthMean[which(data_dist$case_rate_last2monthMean>0)])
data_dist$log_case_rate_last2monthMean <- log(data_dist$case_rate_last2monthMean + data_dist_case_rate_adjust)
data_dist$log_human_density <- log(data_dist$human_density)
data_dist$log_dog_density <- log(data_dist$dog_density)
plot(data_dist$dog_density,data_dist$human_density) # don't include both in models!!


#Combine village level data
data_vill <- data.frame(cases = c(cases_vill),
                        cases_dist = data_dist$cases[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        vax = c(vax_vill),
                        vax_last_month = c(rep(NA,nrow(vax_vill)),c(vax_vill[,1:(ncol(vax_vill)-1)])),
                        vax_last2monthMean = c(vax_vill_last2monthMean),
                        vax_neighbours_last2monthMean = c(vax_vill_neighbours_last2monthMean),
                        vax_notNeighbours_last2monthMean = c(vax_vill_notNeighbours_last2monthMean),
                        vax_last2monthMean_dist = data_dist$vax_last2monthMean[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        immune = c(immune_vill),
                        immune_last_month = c(rep(NA,nrow(immune_vill)),c(immune_vill[,1:(ncol(immune_vill)-1)])),
                        immune_last2monthMean = c(immune_vill_last2monthMean),
                        immune_neighbours_last2monthMean = c(immune_vill_neighbours_last2monthMean),
                        immune_notNeighbours_last2monthMean = c(immune_vill_notNeighbours_last2monthMean),
                        immune_last2monthMean_dist = data_dist$immune_last2monthMean[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        cases_last_month = c(rep(NA,nrow(cases_vill)),c(cases_vill[,1:(ncol(cases_vill)-1)])),
                        case_rate_last_month = c(rep(NA,nrow(cases_vill)),c(cases_vill[,1:(ncol(cases_vill)-1)])/c(dogs[,1:(ncol(cases_vill)-1)])),
                        case_rate_last2monthMean = c(case_rate_vill_last2monthMean),
                        case_rate_neighbours_last2monthMean = c(case_rate_vill_neighbours_last2monthMean),
                        case_rate_notNeighbours_last2monthMean = c(case_rate_vill_notNeighbours_last2monthMean),
                        case_rate_last_month_dist = data_dist$case_rate_last_month[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        case_rate_last2monthMean_dist = data_dist$case_rate_last2monthMean[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        village = rep(rownames(cases_vill),ncol(cases_vill)),
                        month = rep(1:ncol(cases_vill),each=nrow(cases_vill)),
                        dogs = c(dogs),
                        dogs_dist = data_dist$dogs[rep(1:ncol(cases_vill),each=nrow(cases_vill))],
                        humans = c(humans),
                        dog_density = c(dogs)/rep(SD_vill$cells_occ,ncol(cases_vill)),
                        human_density = c(humans)/rep(SD_vill$cells_occ,ncol(cases_vill)),
                        incidence = c(cases_vill/dogs))
data_vill_case_rate_adjust_dist <- 0.5*min(data_vill$case_rate_last2monthMean_dist[which(data_vill$case_rate_last2monthMean_dist>0)])
data_vill$log_case_rate_last2monthMean <- log(data_vill$case_rate_last2monthMean + data_vill_case_rate_adjust_dist)
data_vill$log_case_rate_neighbours_last2monthMean <- log(data_vill$case_rate_neighbours_last2monthMean + data_vill_case_rate_adjust_dist)
data_vill$log_case_rate_notNeighbours_last2monthMean <- log(data_vill$case_rate_notNeighbours_last2monthMean + data_vill_case_rate_adjust_dist)
data_vill$log_case_rate_last2monthMean_dist <- log(data_vill$case_rate_last2monthMean_dist + data_vill_case_rate_adjust_dist)
data_vill$log_human_density <- log(data_vill$human_density)
data_vill$log_dog_density <- log(data_vill$dog_density)
data_vill$HDR <- data_vill$humans/data_vill$dogs

# output data
write.csv(data_vill,"output/incidence_coverage_model_data_village.csv",row.names = F)
write.csv(data_dist,"output/incidence_coverage_model_data_district.csv",row.names = F)


# Vaccination info in form needed by heterogeneity model
n_neighbours <- rowSums(bordering)
n_notNeighbours <- rowSums(not_bordering)
neighbour_vax <- matrix(NA,nrow=sum(n_neighbours),ncol=length(vax_dist))
notNeighbour_vax <- matrix(NA,nrow=sum(n_notNeighbours),ncol=length(vax_dist))
neighbour_weights <- matrix(NA,nrow=sum(n_neighbours),ncol=length(vax_dist))
notNeighbour_weights <- matrix(NA,nrow=sum(n_notNeighbours),ncol=length(vax_dist))
normalisation_neighbours <- normalisation_notNeighbours <- matrix(NA, nrow=nrow(SD_vill), ncol=length(vax_dist))
for(i in 1:length(vax_dist)){
  
  neighbour_vax_i <- c()
  notNeighbour_vax_i <- c()
  neighbour_weights_i <- c()
  notNeighbour_weights_i <- c()
  
  for(vill in 1:nrow(SD_vill)){
    
    # neighbours 
    neighbours <- which(bordering[vill,])
    notNeighbours <- which(not_bordering[vill,])
    
    #coverages for subpopulations
    coverages <- vax_vill[,i] 
    coverages_w_park_mara <- c(coverages,mean(coverages[neighbours[which(!neighbours%in%(nrow(SD_vill)+1:2))]]),0.09) # assume average of neighbour coverages in park and 9% coverage in Mara
    neighbour_vax_i <- c(neighbour_vax_i, coverages_w_park_mara[neighbours])
    notNeighbour_vax_i <- c(notNeighbour_vax_i, coverages[notNeighbours]) 
    
    # Normalisation constants
    normalisation_neighbours[vill,i] <- exp(mean(log(1-neighbour_vax_i)))
    normalisation_notNeighbours[vill,i] <- exp(mean(log(1-notNeighbour_vax_i)))

    # Weights
    neighbour_weights_i  <-  c(neighbour_weights_i, borders_prop[vill,neighbours]) #weight by border length 
    notNeighbour_dogs <- dogs[notNeighbours,i]
    notNeighbour_weights_i <- c(notNeighbour_weights_i, notNeighbour_dogs/sum(notNeighbour_dogs)) #weight by population
  }
  
  neighbour_vax[,i] <- neighbour_vax_i
  notNeighbour_vax[,i] <- notNeighbour_vax_i
  neighbour_weights[,i] <- neighbour_weights_i
  notNeighbour_weights[,i] <- notNeighbour_weights_i
}

data_vill$susc_last2monthMean <- 1-data_vill$vax_last2monthMean
saveRDS(list(N = nrow(data_vill)-nrow(vax_vill)*2, Nv = nrow(vax_vill), Nm =  ncol(vax_vill)-2, Nn = sum(n_neighbours), 
             Nn_v = n_neighbours, Nnn = sum(n_notNeighbours), Nnn_v = n_notNeighbours, 
             n_index = 1:sum(n_neighbours), nn_index = 1:sum(n_notNeighbours),
             Y = data_vill$cases[-c(1:(nrow(cases_vill)*2))],
             S_n = (1-neighbour_vax)/normalisation_neighbours[rep(1:88,n_neighbours),], S_nn = (1-notNeighbour_vax)/normalisation_notNeighbours[rep(1:88,n_notNeighbours),],
             W_n = neighbour_weights, W_nn = notNeighbour_weights,
             norm_n = normalisation_neighbours, norm_nn = normalisation_notNeighbours,
             K = 7,
             X = cbind(1,data_vill[-c(1:(nrow(cases_vill)*2)),c("susc_last2monthMean","log_case_rate_last2monthMean","log_case_rate_neighbours_last2monthMean",
                                             "log_case_rate_notNeighbours_last2monthMean","log_dog_density","HDR")]),
             V = match(data_vill$village[-c(1:(nrow(cases_vill)*2))],SD_vill$Vill_2012),
             offsets = log(data_vill$dogs[-c(1:(nrow(cases_vill)*2))]),
             eps = 1e-5),
        "output/power_mean_model_village_data.rds")

S_n <- split(1:sum(n_neighbours),rep(1:nrow(vax_vill),n_neighbours))
S_n <- sapply(1:nrow(vax_vill),function(x){(1-neighbour_vax)[S_n[[x]],]})
S_nn <- split(1:sum(n_notNeighbours),rep(1:nrow(vax_vill),n_notNeighbours))
S_nn <- sapply(1:nrow(vax_vill),function(x){(1-notNeighbour_vax)[S_nn[[x]],]})
W_n <- split(1:sum(n_neighbours),rep(1:nrow(vax_vill),n_neighbours))
W_n <- sapply(1:nrow(vax_vill),function(x){(neighbour_weights)[W_n[[x]],]})
W_nn <- split(1:sum(n_notNeighbours),rep(1:nrow(vax_vill),n_notNeighbours))
W_nn <- sapply(1:nrow(vax_vill),function(x){(notNeighbour_weights)[W_nn[[x]],]})
save(S_n, S_nn, W_n, W_nn,file="output/neighbour_notNeighbour_susceptibilities.Rdata")


