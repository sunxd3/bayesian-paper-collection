#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(cmdstanr)
})

# ---- env info ----
Sys.setenv(CMDSTANR_VERBOSE = "1",
           MAKEFLAGS = paste0("-j", max(1, parallel::detectCores() - 1)))
cat("CmdStan path: ", cmdstanr::cmdstan_path(), "\n", sep = "")
cat("CmdStan version: ", cmdstanr::cmdstan_version(), "\n", sep = "")
cat("Detected cores: ", parallel::detectCores(), "\n", sep = "")

# ---- load & tidy long data ----
# Expected columns: id, region, group, count (counts must be integers for Poisson)
  long <- read_csv(
  "data.csv",
  show_col_types = FALSE,
  col_types = cols(
    id     = col_integer(),
    region = col_character(),
    group  = col_character(),
    count  = col_double()
  )
) |>
  mutate(region = as.character(region),
         group  = as.character(group))

# Sanity: each id has exactly one group
multi_grp <- long |>
  distinct(id, group) |>
  count(id, name = "n_groups") |>
  filter(n_groups > 1)
if (nrow(multi_grp) > 0) {
  stop("Each id must belong to exactly one group. Offending ids: ",
       paste(head(multi_grp$id, 10), collapse = ", "),
       if (nrow(multi_grp) > 10) " ...")
}

# Regions and animals (stable, explicit levels)
region_levels <- sort(unique(long$region))
animals <- long |>
  distinct(id, group) |>
  arrange(id)

# Factor for group (1..G), kept in the animals table
grp_fac <- factor(animals$group)
A <- nrow(animals)
G <- nlevels(grp_fac)
R <- length(region_levels)

# Build sparse indices for each observed triple
ld <- long |>
  mutate(
    a = match(id, animals$id),                        # 1..A
    r = match(region, region_levels)                  # 1..R
  ) |>
  arrange(a, r)

stan_data <- list(
  A = A, R = R, G = G,
  group_idx = as.integer(grp_fac),        # length A
  N = nrow(ld),
  a = as.integer(ld$a),
  r = as.integer(ld$r),
  y = as.integer(round(ld$count))
)

# ---- compile & sample ----
mod <- cmdstan_model("models/model_hs.stan")

fit <- mod$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 4000,
  iter_sampling = 4000,
  seed = 2017
  # , adapt_delta = 0.9
)

# ---- save & quick summary ----
dir.create("fits", showWarnings = FALSE, recursive = TRUE)
fit$save_object("fits/fit_hs.rds")

print(head(fit$summary(), 12))
