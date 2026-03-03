rm(list=ls())
library(lubridate); 
library(rgdal)
library(dplyr)
library(raster)
library(rgeos)
set.seed(0)

start.date <- as.Date("2002-01-01")
end.date <- as.Date("2022-12-31")

# Read in CT data
biting_animals <- readRDS("data/animalCT_deid.rds"); nrow(biting_animals) 
humans <- readRDS("data/humanCT_deid.rds")

# Map
SD_vill <- readOGR("data/GIS","SD_Villages_2012_From_HHS_250m_Smoothed_UTM") 
SD_vill <- SD_vill[order(SD_vill$Vill_2012),]
vill_grid <- raster("Output/villGrid_1km.grd")
SD_grid <- raster("Output/cellGrid_1km.grd")



# Process rabid carnivores
#______________________

species_to_exclude <- c("Wildlife: Wildebeest", "Other", "Human", "Unknown")
biting_animals$Carnivore <- ifelse(substr(biting_animals$Species,1,9) != "Livestock" & !biting_animals$Species %in% species_to_exclude, T, F)

# Subset on carnivores (or dogs) considered rabid - add information on whether the species is carnivorous
biting_animals$Carnivore <- ifelse(substr(biting_animals$Species,1,9) != "Livestock" & !biting_animals$Species %in% species_to_exclude, T, F)
rabid_carnivores <- subset(biting_animals, Carnivore & Rabid); nrow(rabid_carnivores) # Subset on carnivores (or dogs) considered rabid
rabid_carnivores$Duplicated <- duplicated(rabid_carnivores$ID)

# Subset on end date, having added step information (to "offspring" cases)
rabid_carnivores <- subset(rabid_carnivores, Symptoms.started <= end.date|Date.bitten <= end.date); length(unique(rabid_carnivores$ID)) 
which(rabid_carnivores$Symptoms.started>end.date)

# Add month and day to dataset
rabid_carnivores$month <- NA
ss <- which(!is.na(rabid_carnivores$Symptoms.started))
db <- which(is.na(rabid_carnivores$Symptoms.started))
mean_inc <- mean(rabid_carnivores$Symptoms.started-rabid_carnivores$Date.bitten,na.rm=T)
rabid_carnivores$month[ss] <- month(rabid_carnivores$Symptoms.started[ss])+(year(rabid_carnivores$Symptoms.started[ss]) - year(start.date))*12
rabid_carnivores$month[db] <- month(rabid_carnivores$Date.bitten[db]+mean_inc)+(year(rabid_carnivores$Date.bitten[db]+mean_inc) - year(start.date))*12
which(is.na(rabid_carnivores$month))
rabid_carnivores$day <- NA
rabid_carnivores$day[ss] <- rabid_carnivores$Symptoms.started[ss] - start.date + 1
rabid_carnivores$day[db] <- rabid_carnivores$Date.bitten[db] + mean_inc - start.date + 1
which(is.na(rabid_carnivores$day))

# Add cell ID to dataset
gps <- rabid_carnivores[which(!is.na(rabid_carnivores$UTM.Easting)),c("UTM.Easting","UTM.Northing")]
cell_IDs <- raster::extract(SD_grid,gps)
gps[which(is.na(cell_IDs)),]
rabid_carnivores$Village[which(!is.na(rabid_carnivores$UTM.Easting))][which(is.na(cell_IDs))]
plot(SD_vill)
plot(SD_grid,add=T)
plot(SD_vill,add=T)
points(gps[which(is.na(cell_IDs)),1],gps[which(is.na(cell_IDs)),2],col="red")
# individual near edge of Machochwe that falls just off the grid - find the closest cell within the grid
toAssign <- which(is.na(cell_IDs))
cell_IDs[toAssign] <- apply(X = as.matrix(gps[toAssign,]), MARGIN = 1, 
                            FUN = function(xy) SD_grid[which.min(replace(distanceFromPoints(SD_grid, xy), is.na(SD_grid), NA))])
rabid_carnivores$cell_ID[which(!is.na(rabid_carnivores$UTM.Easting))] <- cell_IDs 
length(which(is.na(rabid_carnivores$cell_ID))) # 43 missing IDs due to lack of gps

# Save the data
saveRDS(object = rabid_carnivores, file = paste0("output/clean_bite_data.rda"))
rabid_carnivores = readRDS(file = paste0("output/clean_bite_data.rda"))



# Animal time series
#______________________

# Dog cases
#--------------
# Extract time series of dog cases per month 
rabid_dogs <- subset(rabid_carnivores,Species=="Domestic dog")
rabid_dogs <- subset(rabid_dogs,Duplicated==F) # remove duplicates
no_gps <- which(is.na(rabid_dogs$UTM.Easting)); length(no_gps)
matchVillage <- match(rabid_dogs$Village[no_gps],SD_vill$Vill_2012)
vill_coords <- gPointOnSurface(SD_vill, byid = T)@coords
rabid_dogs$UTM.Easting[no_gps] <- vill_coords[matchVillage,1]
rabid_dogs$UTM.Northing[no_gps] <- vill_coords[matchVillage,2]
plot(rabid_dogs$UTM.Easting, rabid_dogs$UTM.Northing, pch=20) 
write.csv(rabid_dogs,file="output/serengeti_rabid_dogs.csv",row.names = F)

ts <- rep(0,12*(year(end.date)-year(start.date)+1))
table_rabid_dogs <- table(rabid_dogs$month)
ts[as.numeric(rownames(table_rabid_dogs))] <- table_rabid_dogs
write.table(ts,paste0("output/Serengeti_monthly_rabid_dogs_",start.date,"_",end.date,".csv"),row.names = F,col.names = F,sep=",")

# And cases per month per village (not saving this in this public version of the
# repo to avoid overwriting the file produced using the exact locations, which
# is already included in the output folder)
ts_village <- matrix(0,ncol=12*(year(end.date)-year(start.date)+1),nrow=nrow(SD_vill))
for(i in 1:nrow(SD_vill)){
  table_rabid_dogs_i <- table(c(rabid_dogs$month[which(SD_vill$Vill_2012[vill_grid[match(rabid_dogs$cell_ID,SD_grid[])]]==SD_vill$Vill_2012[i])],
                                rabid_dogs$month[which(is.na(rabid_dogs$UTM.Easting)&rabid_dogs$Village==SD_vill$Vill_2012[i])]))
  ts_village[i,as.numeric(rownames(table_rabid_dogs_i))] <- table_rabid_dogs_i 
}
# write.table(ts_village,paste0("output/Serengeti_monthly_rabid_dogs_village_",start.date,"_",end.date,".csv"),row.names = SD_vill$Vill_2012,col.names = F,sep=",")


# Animal cases
#--------------
# Extract time series of animal cases per month 
rabid_animals <- biting_animals[which(biting_animals$Rabid==TRUE & biting_animals$Species!="Human"),]
rabid_animals$Duplicated <- duplicated(rabid_animals$ID)

# Add month and day to dataset
rabid_animals$month <- NA
ss <- which(!is.na(rabid_animals$Symptoms.started))
db <- which(is.na(rabid_animals$Symptoms.started))
mean_inc <- mean(rabid_animals$Symptoms.started-rabid_animals$Date.bitten,na.rm=T)
rabid_animals$month[ss] <- month(rabid_animals$Symptoms.started[ss])+(year(rabid_animals$Symptoms.started[ss]) - year(start.date))*12
rabid_animals$month[db] <- month(rabid_animals$Date.bitten[db]+mean_inc)+(year(rabid_animals$Date.bitten[db]+mean_inc) - year(start.date))*12
which(is.na(rabid_animals$month))
rabid_animals$day <- NA
rabid_animals$day[ss] <- rabid_animals$Symptoms.started[ss] - start.date + 1
rabid_animals$day[db] <- rabid_animals$Date.bitten[db] + mean_inc - start.date + 1
which(is.na(rabid_animals$day))
rabid_animals <- subset(rabid_animals, Symptoms.started <= end.date|Date.bitten <= end.date); nrow(rabid_animals) # 3567
rabid_animals <- subset(rabid_animals,Duplicated==F) # remove duplicates
no_gps <- which(is.na(rabid_animals$UTM.Easting)); length(no_gps)
matchVillage <- match(rabid_animals$Village[no_gps],SD_vill$Vill_2012)
vill_coords <- gPointOnSurface(SD_vill, byid = T)@coords
rabid_animals$UTM.Easting[no_gps] <- vill_coords[matchVillage,1]
rabid_animals$UTM.Northing[no_gps] <- vill_coords[matchVillage,2]
plot(rabid_animals$UTM.Easting, rabid_animals$UTM.Northing, pch=20) 
write.csv(rabid_animals,file="output/serengeti_rabid_animals.csv",row.names = F)

ts <- rep(0,12*(year(end.date)-year(start.date)+1))
table_rabid_animals <- table(rabid_animals$month)
ts[as.numeric(rownames(table_rabid_animals))] <- table_rabid_animals
write.table(ts,paste0("output/Serengeti_monthly_rabid_animals_",start.date,"_",end.date,".csv"),row.names = F,col.names = F,sep=",")


# Animal cases by village
#--------------

ts_village <- matrix(0,ncol=12*(year(end.date)-year(start.date)+1),nrow=nrow(SD_vill))
for(i in 1:nrow(SD_vill)){
  table_rabid_animals_i <- table(c(rabid_animals$month[which(SD_vill$Vill_2012[vill_grid[match(rabid_animals$cell_ID,SD_grid[])]]==SD_vill$Vill_2012[i])],
                                   rabid_animals$month[which(is.na(rabid_animals$UTM.Easting)&rabid_animals$Village==SD_vill$Vill_2012[i])]))
  ts_village[i,as.numeric(rownames(table_rabid_animals_i))] <- table_rabid_animals_i 
}
write.table(ts_village,paste0("output/Serengeti_monthly_rabid_animals_village_",start.date,"_",end.date,".csv"),row.names = SD_vill$Vill_2012,col.names = F,sep=",")

# Representation of different species
rabid_animals$Species[which(rabid_animals$Other.species=="bush baby")] <- "Wildlife: Bush baby"
filter(rabid_animals,Species=="Unknown") # no clues as to species
length(which(rabid_animals$Species=="Domestic dog"))/nrow(rabid_animals)
not_dog <- which(!rabid_animals$Species%in%c("Domestic dog", "Unknown"))
n_not_dog <- length(not_dog)
sort(round((table(rabid_animals$Species[not_dog])/n_not_dog)*100,1))
round((sum(grepl("Livestock",rabid_animals$Species))/n_not_dog)*100,1)
round((sum(!(grepl("Livestock",rabid_animals$Species)|rabid_animals$Species%in%c("Domestic dog","Cat","Wildlife: Jackal","Unknown")))/n_not_dog)*100,1)



# Process human exposures
#______________________

length(which(is.na(humans$Date.bitten)))/nrow(humans) #8% have missing date bitten
humans$Date.proxy <- humans$Date.bitten
humans$Date.proxy[which(is.na(humans$Date.proxy))] <- humans$Date.reported[which(is.na(humans$Date.proxy))] - humans$How.many.days.taken.to.seek.treatment[which(is.na(humans$Date.proxy))]
length(which(is.na(humans$Date.proxy)))/nrow(humans) #that didn't fill in any
table(humans$How.many.days.taken.to.seek.treatment) # mode at 3 days
mean(humans$How.many.days.taken.to.seek.treatment,na.rm=T) #mean at 3 days
median(humans$How.many.days.taken.to.seek.treatment,na.rm=T) # median of 3 days
humans$Date.proxy[which(is.na(humans$Date.proxy))] <- humans$Date.reported[which(is.na(humans$Date.proxy))] - median(humans$How.many.days.taken.to.seek.treatment,na.rm=T)
length(which(is.na(humans$date)))/nrow(humans) 

# Subset by start date
humans <- subset(humans, (Date.proxy >= start.date) ); nrow(humans) 

# Save the data
saveRDS(object = humans, file = paste0("output/clean_human_data.rds"))
humans = readRDS(file = paste0("output/clean_human_data.rds"))

# Subset out exposures and deaths
exposures <- subset(humans, Rabid=="Yes")
deaths <- subset(humans,Patient.outcome=="Died" & Cause.of.death=="Rabies")

# Add month and day to exposures dataset
exposures$month <- month(exposures$Date.proxy)+(year(exposures$Date.proxy) - year(start.date))*12
exposures$day <- exposures$Date.proxy - start.date + 1
exposures <- subset(exposures, Date.proxy <= end.date|Date.bitten <= end.date); nrow(exposures) 
no_gps <- which(is.na(exposures$UTM.Easting)); length(no_gps)
matchVillage <- match(exposures$Village[no_gps],SD_vill$Vill_2012)
vill_coords <- gPointOnSurface(SD_vill, byid = T)@coords
exposures$UTM.Easting[no_gps] <- vill_coords[matchVillage,1]
exposures$UTM.Northing[no_gps] <- vill_coords[matchVillage,2]
which(is.na(exposures$UTM.Easting))# still a few
plot(exposures$UTM.Easting, exposures$UTM.Northing, pch=20) 
write.csv(exposures,file="output/serengeti_exposed_humans.csv",row.names = F)

# Add month and day to deaths dataset
which(is.na(deaths$When.died))
deaths$Date.reported[which(is.na(deaths$When.died))]
deaths$Date.proxy[which(is.na(deaths$When.died))]
deaths$When.died.with.imputed <- deaths$When.died
deaths$When.died.with.imputed[which(is.na(deaths$When.died))]<-deaths$Date.proxy[which(is.na(deaths$When.died))]
deaths$month <- month(deaths$When.died.with.imputed)+(year(deaths$When.died.with.imputed) - year(start.date))*12
deaths$day <- deaths$When.died.with.imputed - start.date + 1
deaths <- subset(deaths, When.died <= end.date|Date.bitten <= end.date); nrow(deaths) 
no_gps <- which(is.na(deaths$UTM.Easting)); length(no_gps)
plot(deaths$UTM.Easting, deaths$UTM.Northing, pch=20) 
write.csv(deaths,file="output/serengeti_human_deaths.csv",row.names = F)



# Time series of human exposures and deaths
#______________________

#human exposures through time (all animals and just dogs)
ts <- rep(0,12*(year(end.date)-year(start.date)+1))
table_exposures <- table(exposures$month)
ts[as.numeric(rownames(table_exposures))] <- table_exposures
write.table(ts,paste0("output/Serengeti_monthly_human_exposures_",start.date,"_",end.date,".csv"),row.names = F,col.names = F,sep=",")
exposures_dogs <- filter(exposures, Attacking.species=="Domestic dog")
ts <- rep(0,12*(year(end.date)-year(start.date)+1))
table_exposures <- table(exposures_dogs$month)
ts[as.numeric(rownames(table_exposures))] <- table_exposures
write.table(ts,paste0("output/Serengeti_monthly_human_exposures_dogs_",start.date,"_",end.date,".csv"),row.names = F,col.names = F,sep=",")
nrow(exposures)

# human deaths through time
ts <- rep(0,12*(year(end.date)-year(start.date)+1))
table_deaths <- table(deaths$month)
ts[as.numeric(rownames(table_deaths))] <- table_deaths
write.table(ts,paste0("output/Serengeti_monthly_human_deaths_",start.date,"_",end.date,".csv"),row.names = F,col.names = F,sep=",")

