#!/usr/bin/env Rscript

# Diagnostics plotting script
# Usage:
#   Rscript --vanilla plot_diagnostics.r [--format EXT] [--scale S]
#
# Options:
#   --help           Show this help and exit.
#   --format EXT     Output image format/extension. Default: png
#                    Allowed: png, pdf, svg, jpeg, jpg, tiff, tif, bmp, eps, ps
#   --scale S        Positive real scale factor applied to all figure sizes.
#                    Default: 1.0 (e.g., --scale 1.5 makes plots 50% larger)
#
# Outputs go to: diagnostics/*.EXT

# -------- Help & CLI --------
print_help <- function() {
  cat("
Diagnostics plotting script

Usage:
  Rscript --vanilla plot_diagnostics.r [--format EXT] [--scale S]

Options:
  --help           Show this help and exit.
  --format EXT     Output image format/extension. Default: png
                   Allowed: png, pdf, svg, jpeg, jpg, tiff, tif, bmp, eps, ps
  --scale S        Positive real scale factor applied to all figure sizes.
                   Default: 1.0 (e.g., --scale 1.5 makes plots 50% larger)

Outputs:
  Files are written to the 'diagnostics/' directory with the chosen extension.
\n")
}

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("--help","-h"))) { print_help(); quit(status = 0) }

fmt <- "png"
scale_factor <- 1.0

if (length(args)) {
  j <- which(args == "--format")
  if (length(j)) {
    if (j == length(args)) stop("Flag --format needs a value, e.g., --format png")
    fmt <- tolower(args[j + 1])
  }
  k <- which(args == "--scale")
  if (length(k)) {
    if (k == length(args)) stop("Flag --scale needs a value, e.g., --scale 1.25")
    s_try <- suppressWarnings(as.numeric(args[k + 1]))
    if (!is.finite(s_try) || s_try <= 0) stop("Invalid --scale: must be a positive number")
    scale_factor <- s_try
  }
}

allowed_fmt <- c("png","pdf","svg","jpeg","jpg","tiff","tif","bmp","eps","ps")
if (!fmt %in% allowed_fmt) stop("Unsupported --format: ", fmt)

# -------- Deps (no attaches → no conflicts) --------
# Use explicit namespaces everywhere.
# Needed: bayesplot, posterior, ggplot2, dplyr, stringr
bayesplot::color_scheme_set("purple")

# -------- Helpers --------
.save_plot <- function(plot, base, width, height, units = "in", dpi = 600) {
  outdir <- "diagnostics"
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  outfile <- file.path(outdir, paste0(base, ".", fmt))
  w <- width  * scale_factor
  h <- height * scale_factor
  if (fmt %in% c("tiff","tif")) {
    ggplot2::ggsave(outfile, plot = plot, dpi = dpi, width = w, height = h,
                    units = units, compression = "lzw")
  } else {
    ggplot2::ggsave(outfile, plot = plot, dpi = dpi, width = w, height = h,
                    units = units)
  }
  message("Wrote: ", outfile)
}

get_param_ss <- function(param_regex, rhats, neff_ratio_vec){
  nm   <- names(rhats)
  idxs <- which(stringr::str_detect(nm, param_regex))
  if (!length(idxs)) return(data.frame(param_name=character(), unique_name=character(),
                                       rhat=numeric(), neff=numeric(), size=numeric()))
  data.frame(
    param_name  = rep(param_regex, length(idxs)),
    unique_name = nm[idxs],
    rhat        = as.numeric(rhats[idxs]),
    neff        = as.numeric(neff_ratio_vec[idxs]),  # ratio in [0,1]
    size        = 1/length(idxs)
  )
}

get_highest_rhat <- function(param_regex){
  cand <- params[stringr::str_detect(params$unique_name, param_regex), , drop = FALSE]
  if (!nrow(cand)) return(NA_character_)
  cand$unique_name[which.max(cand$rhat)]
}

# -------- Load fit & diagnostics --------
fit   <- readRDS("fits/fit_hs.rds")

# Use posterior draws (fixes mcmc_trace error on CmdStanR fits)
draws <- fit$draws(format = "draws_array", inc_warmup = FALSE)

# Diagnostics from posterior
rhats <- posterior::rhat(draws)
# posterior doesn't export neff_ratio(); compute it:
neff_ratio_vec <- posterior::ess_bulk(draws) / posterior::ndraws(draws)

param_names <- c("theta\\[", "tau\\[", "kappa\\[", "gamma_raw\\[")
params <- dplyr::bind_rows(lapply(
  param_names, function(rx) get_param_ss(rx, rhats, neff_ratio_vec)
))
params$param_name <- factor(params$param_name, levels = param_names)

b_mu     <- get_highest_rhat("theta\\[")
b_tau    <- get_highest_rhat("tau\\[")
b_kappa  <- get_highest_rhat("kappa\\[")
b_gamma  <- get_highest_rhat("gamma_raw\\[")
bad_pars <- unique(c(b_mu, b_tau, b_kappa, b_gamma))
bad_pars <- bad_pars[!is.na(bad_pars)]
bad_pars <- bad_pars[bad_pars %in% posterior::variables(draws)]  # ensure exists

# -------- Rhat vs Neff (ratio) --------
ggplot2::theme_set(ggplot2::theme_classic(base_size = 10))
ggplot2::theme_update(
  axis.title.y    = ggplot2::element_text(angle = 0, vjust = 0.5, face = "italic"),
  axis.title.x    = ggplot2::element_text(face = "italic"),
  legend.position = "none",
  legend.title    = ggplot2::element_blank()
)

p <- ggplot2::ggplot() +
  ggplot2::geom_point(
    data = params,
    ggplot2::aes(x = rhat, y = neff, fill = param_name, size = size, color = param_name),
    alpha = 0.50
  ) +
  ggplot2::geom_point(
    data = dplyr::filter(params, unique_name %in% bad_pars),
    ggplot2::aes(x = rhat, y = neff, fill = param_name, size = size),
    shape = 21
  ) +
  ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed") +
  ggplot2::geom_hline(yintercept = 0.1) +
  ggplot2::scale_size(guide = "none") +
  ggplot2::scale_color_brewer(palette = "Accent") +
  ggplot2::scale_fill_brewer(palette = "Accent") +
  ggplot2::xlab(expression(widehat(italic(R)))) +
  ggplot2::ylab(expression(italic(frac(N[eff], N))))
.save_plot(p, "neff_rhat", width = 0.6 * 5.2, height = 2.6)

# -------- Energy --------
np <- bayesplot::nuts_params(fit)
bayesplot::bayesplot_theme_set(ggplot2::theme_classic())
bayesplot::bayesplot_theme_update(text = ggplot2::element_text(size = 10))
p <- bayesplot::mcmc_nuts_energy(np, merge_chains = TRUE) +
     ggplot2::theme(legend.position.inside = c(0.85, 0.75))  # ggplot2 ≥ 3.5
.save_plot(p, "energy", width = 0.4 * 5.2, height = 2.6)

# -------- Trace (worst per family) --------
bayesplot::bayesplot_theme_set(ggplot2::theme_classic())
bayesplot::bayesplot_theme_update(
  text            = ggplot2::element_text(size = 10),
  axis.title.y    = ggplot2::element_text(angle = 0, vjust = 0.5),
  axis.text.y     = ggplot2::element_blank(),
  axis.ticks.y    = ggplot2::element_blank(),
  legend.position = "none"
)

.plot_one_trace <- function(par, base_col, lab_expr, base_name) {
  bayesplot::color_scheme_set(grDevices::colorRampPalette(c(base_col, "black"))(8)[1:6])
  p <- bayesplot::mcmc_trace(draws, pars = par, regex_pars = FALSE) +
       ggplot2::ylab(lab_expr) +
       ggplot2::theme(legend.position = "none")
  .save_plot(p, base_name, width = 5.2 - 0.162, height = 2.6/4)
}

if (length(bad_pars) >= 1) .plot_one_trace(bad_pars[1], "#7fc97f", expression(theta),        "bp_theta")
if (length(bad_pars) >= 2) .plot_one_trace(bad_pars[2], "#beaed4", expression(tau),          "bp_tau")
if (length(bad_pars) >= 3) .plot_one_trace(bad_pars[3], "#fdc086", expression(kappa),        "bp_kappa")
if (length(bad_pars) >= 4) .plot_one_trace(bad_pars[4], "#ffff99", expression(tilde(gamma)), "bp_gamma")
