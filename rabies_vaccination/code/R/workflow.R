
# Create 1km grid of the study area with cells assigned to district, ward and
# village
source("R/Create_SD_grid.R")

# Estimate human and dog populations monthly by village and district
source("R/SerengetiDogsVillage.R")
source("R/SerengetiDogsGrid.R")

# Produce Fig. 1
source("R/Fig.1.R")

# Look at parameter options for bounding vaccination coverage in (0,1). And
# produce Fig. S14
source("R/Bound_coverage.R")

# Obtain vaccination coverage by village and district from various sources of
# vaccination data
source("R/CreateVaccination.R")

# Process CT data for downstream scripts
source("R/process_CT_data.R")

# Figs 2, S1, S2, and S4, Video S1, & Table S1
source("R/DescriptivePlots.R")

# Get distribution of human bites by a rabid dog (& Fig. S3)
source("R/HumanBitesPerDog.R")

# Fit distance kernels for transmission tree
# (Note these look very different to the ones presented in the paper, because of
# the increased distances resulting from randomly jittering the locations.
# However the distribution used in the paper is available in the output folder
# and used for R/Plot_TT_dists.R and R/Run_Transmission_Trees_treerabid.R)
source("R/Make_Distance_Kernel.R")

# Fit incubation and infectious periods, serial intervals and generation
# intervals
source("R/Make_Incubation_Infectious_Periods.R")

# Fig.S13
source("R/Plot_TT_dists.R") 

# Generate transmission trees and estimate and plot incursions
source("R/Run_Transmission_Trees_treerabid.R")

# Fig 5
source("R/incursions_over_time_treerabid.R")

# Prepare input data for all monthly models of incidence
source("R/model_data_prep.R")

# Prepare input data for all annual model of incidence
source("R/model_data_prep_annual.R")

# Models describing impact of prior vaccination and prior incidence on current
# incidence, without considering heterogeneity in coverage
source("R/incidence_coverage_models.R") # Figs 3, S5, and S9; Tables 1 and S2
source("R/incidence_coverage_models_annual.R") # Figs S11, and S12; Tables S3-4

# Explore feasible values of the power parameter p in heterogeneity models (Fig.
# S6)
source("R/explore_powers.R")

# Fit power mean models (commented out as very slow - district not bad but
# village fits take a day each!!) 
# source("R/Power_mean_model_district_stan.R") 
# source("R/Power_mean_model_village_stan.R") 

# Plot power mean models
source("R/Plot_stan_district_monthly.R") # Fig S10
source("R/Plot_stan_village_monthly.R") # Figs 4, S7, S8
