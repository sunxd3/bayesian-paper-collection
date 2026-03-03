library(cmdstanr)
library(tidyverse)
library(bayesplot)
library(HDInterval)
library(xtable)
require(ggplot2)
library(egg)
library(optparse) # for input arguments
library(cowplot) # for plot_grid (combines plots with differing x-axes)
library(ggbreak) # for breaks in the axis
library(ggpubr) # für as_ggplot()

source("setup.local.R")

base_dir = 'analysis_pleskac/Reanalysis_Study_1'

fit_dir_b = file.path(base_dir, 'fits', 'fit_basicS1_basic')
fit_dir_bc = file.path(base_dir, 'fits', 'fit_basicS1_cens')
fit_dir_bcp = file.path(base_dir, 'fits', 'fit_basicS1_cens_prob')
fit_dir_t = file.path(base_dir, 'fits', 'fit_basicS1_trunc_over_cdf0+cdf1')
jags_dir = base_dir
#

{
#fit_b = as_cmdstan_fit(paste0(fit_dir_b, '/fit-', 1:4, '.csv'))
#fit_bc = as_cmdstan_fit(paste0(fit_dir_bc, '/fit-', 1:4, '.csv'))
#fit_bcp = as_cmdstan_fit(paste0(fit_dir_bcp, '/fit-', 1:4, '.csv'))
#fit_t = as_cmdstan_fit(paste0(fit_dir_t, '/fit-', 1:4, '.csv'))


summary_b = fit_b$summary()
summary_bc = fit_bc$summary()
summary_bcp = fit_bcp$summary()
summary_t = fit_t$summary()

min(summary_b$ess_bulk) #487
min(summary_bc$ess_bulk) #1036
min(summary_bcp$ess_bulk) #1878
min(summary_t$ess_bulk) #1160

max(summary_b$rhat) #1.011
max(summary_bc$rhat) #1.005
max(summary_bcp$rhat) #1.002
max(summary_t$rhat) #1.007

summary_t[summary_t$ess_bulk < 400,] 
summary_b[summary_b$rhat > 1.01,] 

summary_b_mu = summary_b[which(substr(summary_b$variable, 1, 2) == 'mu'),]
summary_bc_mu = summary_bc[which(substr(summary_bc$variable, 1, 2) == 'mu'),]
summary_bcp_mu = summary_bcp[which(substr(summary_bcp$variable, 1, 2) == 'mu'),]
summary_t_mu = summary_t[which(substr(summary_t$variable, 1, 2) == 'mu'),]

#write_csv(summary_b_mu, file.path(fit_dir_b, 'summary_mu.csv'))
#write_csv(summary_bc_mu, file.path(fit_dir_bc, 'summary_mu.csv'))
#write_csv(summary_bcp_mu, file.path(fit_dir_bcp, 'summary_mu.csv'))
#write_csv(summary_t_mu, file.path(fit_dir_t, 'summary_mu.csv'))

summary_b_mu = read_csv(file.path(fit_dir_b, 'summary_mu.csv'))
summary_bc_mu = read_csv(file.path(fit_dir_bc, 'summary_mu.csv'))
summary_bcp_mu = read_csv(file.path(fit_dir_bcp, 'summary_mu.csv'))
summary_t_mu = read_csv(file.path(fit_dir_t, 'summary_mu.csv'))


fit_JAGS = read_csv(file.path(jags_dir, 'S3_Study1_pleskac_JAGS_results.csv'))

min(summary_b_mu$ess_bulk) #487
min(summary_bc_mu$ess_bulk) #1036
min(summary_bcp_mu$ess_bulk) #2174
min(summary_t_mu$ess_bulk) #1160

max(summary_b_mu$rhat) #1.011
}


{
#make HDI or read in overview
vars = summary_t_mu$variable
fit_Stan = fit_t
summary_Stan_mu = summary_t_mu
fit_dir = fit_dir_t

#  HDI = as.data.frame(t(hdi(fit_Stan$draws(format = "data.frame", variables = vars), credMass = 0.95)))
#  HDI = HDI[vars,]
#  HDI$variable = rownames(HDI)
#  table_cols = c("variable", "mean", "lower", "median", "upper", "rhat", "ess_bulk", "ess_tail")
#  overview = merge(summary_Stan_mu, HDI, by="variable", sort = FALSE)[,table_cols]
#  write_csv(overview, file.path(fit_dir, 'overview.csv'))



#save traceplots
#color_scheme_set("mix-blue-pink")
#jpeg(file=paste0(fit_dir, "/traces_overview.jpeg"), width=1200, height=1200)
#print(bayesplot::mcmc_trace(fit_Stan$draws(inc_warmup = TRUE), n_warmup = fit_Stan$metadata()$iter_warmup, pars=vars))
#dev.off()
}


#Read overview
overviewgg_b = read_csv(file.path(fit_dir_b, 'overview.csv'))
overviewgg_bc = read_csv(file.path(fit_dir_bc, 'overview.csv'))
overviewgg_bcp = read_csv(file.path(fit_dir_bcp, 'overview.csv'))
overviewgg_trunc = read_csv(file.path(fit_dir_t, 'overview.csv'))

fit_JAGS = read_csv(file.path(jags_dir, 'S3_Study1_pleskac_JAGS_results.csv'))





#prepare the overviews for plotting
{
overviewgg_b = overviewgg_b %>%
  mutate(model = 'basic') %>%
  dplyr::select(variable, mean, lower, median, upper, model)
overviewgg_bc = overviewgg_bc %>%
  mutate(model = 'cens.') %>%
  dplyr::select(variable, mean, lower, median, upper, model)
overviewgg_bcp = overviewgg_bcp %>%
  mutate(model = 'cens. with CCDF') %>%
  dplyr::select(variable, mean, lower, median, upper, model)
overviewgg_trunc = overviewgg_trunc %>%
  mutate(model = 'trunc.') %>%
  dplyr::select(variable, mean, lower, median, upper, model)

overviewggJAGS = fit_JAGS
overviewggJAGS = overviewggJAGS[match(overviewgg_bc$variable, overviewggJAGS$...1),] %>%
  mutate(model = 'JAGS') %>%
  dplyr::select(...1, Lower95, Median, Upper95, Mean, model) %>%
  rename(variable = ...1, mean = Mean, lower = Lower95, median = Median, upper = Upper95) 

}


#will be ordered alphabetically in plot
overview_gg = rbind(overviewgg_b, overviewgg_bc, overviewgg_bcp, 
                    overviewggJAGS, overviewgg_trunc)
model_selection = c(1, 2, 3, 4, 5)
#overview_gg = rbind(overviewggJAGS)
#model_selection = c(4)




#create overview tables for each parameter (combination)
{
number_models = length(model_selection)
overviewgg_aw = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muA', 'muB'),]
overviewgg_aw$facet = factor(rep(rep(c("w", "a"), each=2), number_models),
                             levels=c("w", "a"), ordered = TRUE)
overviewgg_aw$cond_1 = factor(rep(rep(c("W", "B"), 2), number_models),
                              levels=c("W", "B"), ordered = TRUE)


#check correct assignment of labels
#matrix [nWCon, nBtwn] muBeta;
#matrix [nWCon, nBtwn] muAlpha;
#nWCon = 2: 1 white, 2 black
#nBtwn = 2: 1 safe, 2 danger
#-> passt!



overviewgg_a = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muA'),]
overviewgg_a$facet = factor(rep(rep(c("a"), each=2), number_models),
                             levels=c("a"), ordered = TRUE)
overviewgg_a$cond_1 = factor(rep(rep(c("W", "B"), 1), number_models),
                              levels=c("W", "B"), ordered = TRUE)




overviewgg_w = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muB'),]
overviewgg_w$facet = factor(rep(rep(c("w"), each=2), number_models),
                             levels=c("w"), ordered = TRUE)
overviewgg_w$cond_1 = factor(rep(rep(c("W", "B"), 1), number_models),
                              levels=c("W", "B"), ordered = TRUE)




overviewgg_vt = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muD', 'muN'),]
overviewgg_vt$facet = factor(rep(rep(c("v", "t0"), each=4), number_models),
                          levels=c("v", "t0"), ordered = TRUE)
overviewgg_vt$cond_1 = factor(rep(rep(c("W", "B"), 4), number_models),
                            levels=c("W", "B"), ordered = TRUE)
overviewgg_vt$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 2), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)

#check correct assignment of labels
#matrix [nWConTarget, nBtwn] muDelta;
#matrix [nWConTarget, nBtwn] muNDT;
#nWConTarget = 4: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G


overviewgg_v = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muD'),]
overviewgg_v$facet = factor(rep(rep(c("v"), each=4), number_models),
                             levels=c("v"), ordered = TRUE)
overviewgg_v$cond_1 = factor(rep(rep(c("W", "B"), 2), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_v$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 1), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)


overviewgg_t = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muN'),]
overviewgg_t$facet = factor(rep(rep(c("t0"), each=4), number_models),
                             levels=c("t0"), ordered = TRUE)
overviewgg_t$cond_1 = factor(rep(rep(c("W", "B"), 2), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_t$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 1), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)
}


#create plots for each parameter (combination)
{
colors = c("black", "red", "blue", "green", "violet", "yellow", "brown")[model_selection]
axis_title_size = 15
strip_text_size = 12.5
param_plot_aw = ggplot(overviewgg_aw, aes(x = interaction(cond_1, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("Mean") +
  xlab('') +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  scale_color_manual(values=colors) +
  theme(legend.position='none')



param_plot_a = ggplot(overviewgg_a, aes(x = interaction(cond_1, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("") +
  xlab('') +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  scale_color_manual(values=colors) +
  theme(legend.position='none', strip.text=element_text(size=strip_text_size))



param_plot_w = ggplot(overviewgg_w, aes(x = interaction(cond_1, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("Mean") +
  xlab('') +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  scale_color_manual(values=colors) +
  theme(legend.position='none', axis.title=element_text(size=axis_title_size),
        strip.text=element_text(size=strip_text_size))





param_plot_vt = ggplot(overviewgg_vt, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("Mean") +
  xlab('') +
  scale_y_break(c(-1.5,1.5)) +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  labs(color = "Model:") +
  scale_color_manual(values=colors) +
  theme(legend.position='right', legend.direction='horizontal')
  theme(legend.position='none')




param_plot_v = ggplot(overviewgg_v, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("Mean") +
  xlab('') +
  scale_y_break(c(-1.1,0.9)) +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  scale_color_manual(values=colors) +
  theme(legend.position='none', axis.title=element_text(size=axis_title_size),
        strip.text=element_text(size=strip_text_size))



param_plot_t = ggplot(overviewgg_t, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
  geom_pointrange(aes(ymax = upper, ymin = lower),
                  size = 0.5,
                  shape = 1,
                  position = position_dodge(width=0.5)) +
  ylab("") +
  xlab('') +
  facet_wrap(~facet, ncol=2, scales="free_y") +
  guides(x = ggh4x::guide_axis_nested(delim = "&")) +
  labs(color = "Model") +
  scale_color_manual(values=colors) +
  theme(legend.position='none', strip.text=element_text(size=strip_text_size))
}


#print and save plot
{
#extract legend from param_plot_vt
leg = get_legend(param_plot_vt)
plot_legend = as_ggplot(leg)
plot_legend = plot_legend + 
  theme(legend.margin=margin(c(0,0,0,0)))
#param_plot_aw
#param_plot_vt

#cowplot::plot_grid(param_plot_aw, param_plot_vt, nrow=2)

plot = aplot::plot_list(param_plot_w, param_plot_a, param_plot_v, param_plot_t)
plot_with_legend = aplot::plot_list(plot, plot_legend, nrow=2, heights=c(95,5))

#pdf(file.path(base_dir, "plots/Pleskac_Study1_all_models.pdf"), width=8, height=5)
plot_with_legend
#dev.off()
#
}
plot_with_legend
#







##### Check: insert the estimates into formula: p_r = CCDF0(U|a,v,w) + CCDF1(U|a,-v,1-w)
# where p_r is the likelihood of a right-censored data point
{
library(WienR)
#nWConTarget = 4: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
data = read_csv(paste0( 'analysis_pleskac', '/Pleskac_Daten/Study1/Study1TrialData.csv'))
n_cond = table(data$conditionRaceObj)
true_out = table(data$ybin, data$conditionRaceObj)
true_anteil_out = (true_out['0',] + true_out['2',])

est_anteil_out = data.frame()
for(s in 1:4) {
  index = 1
  summary = list(summary_b_mu, summary_bc_mu, summary_bcp_mu, summary_t_mu)[[s]]
  for(i_vt in 1:4) {
    i_ab = ifelse((i_vt == 1 | i_vt == 3), 1, 2)
    alpha = summary[summary$variable == paste0('muAlpha[', i_ab, ']'),]$mean
    beta = summary[summary$variable == paste0('muBeta[', i_ab, ']'),]$mean
    delta = summary[summary$variable == paste0('muDelta[', i_vt, ']'),]$mean
    tau = summary[summary$variable == paste0('muNDT[', i_vt, ']'),]$mean
    N = n_cond[index]
    p = 
      WienR::WienerCDF(1000, 1, alpha, delta, beta, tau)[[1]] - WienR::WienerCDF(0.851, 1, alpha, delta, beta, tau)[[1]] +
      WienR::WienerCDF(1000, 2, alpha, -delta, 1-beta, tau)[[1]]- WienR::WienerCDF(0.851, 2, alpha, -delta, 1-beta, tau)[[1]]
    est_anteil_out[s, index] = round(p*N,0)
    index = index + 1
  }
}
names(est_anteil_out) = names(true_anteil_out)
row.names(est_anteil_out) = c('basic', 'cens', 'cens prob', 'trunc')
est_anteil_out#/1400
true_anteil_out#/1400
rowSums(est_anteil_out)/(1400*4)
sum(true_anteil_out)/(1400*4)

# > est_anteil_out#/1400
#              1   2  3 4
#  basic     172 212  5 1
#  cens      221 287  6 1
#  cens prob 221 286  6 1
#  trunc     421 555 19 3
# > true_anteil_out#/1400
#  1  2  3  4 
#  52 72 29 15 
}






##### Check: Das trunkierte Modell beschreibt die Daten im Antwortfenster besser beschreibt. 
#Die Vorhersagen für die beobachteten Häufigkeiten und mittleren Antwortzeiten von Gun und Tool Antworten 
#pro Bedingung berechnen. Eine einfache Möglichkeit, die zu berechnen, wäre Zufallswerte mit den jeweiligen 
#Modellparametern zu generieren und relative Häufigkeiten bzw. Mittelwerte zu berechnen. 
#Für die zensierten Modelle würde man dabei solche Zufallswerte mit Zeiten außerhalb des Fensters 
#rauswerfen, für das trunkierte Modell müsste man mit oberer Schranke = Antwortdeadline ziehen.
library(WienR)
upper_bound_Study_1 = 0.851
data = read_csv(paste0( 'analysis_pleskac', '/Pleskac_Daten/Study1/Study1TrialData.csv'))
data_rts_freq = data %>%
  drop_na(rt) %>%
  group_by(conditionRaceObj) %>% 
  summarize(mean_rt = round(mean(rt/1000),2), obs_freq = round(mean(resp0DS1S),2))

obs_freqs = data.frame()
mean_rts = data.frame()
all_data = data.frame()
for(s in 1:4) {
  index = 1
  summary = list(summary_b_mu, summary_bc_mu, summary_bcp_mu, summary_t_mu)[[s]]
  for(i_vt in 1:4) {
    i_ab = ifelse((i_vt == 1 | i_vt == 3), 1, 2)
    alpha = summary[summary$variable == paste0('muAlpha[', i_ab, ']'),]$mean
    beta = summary[summary$variable == paste0('muBeta[', i_ab, ']'),]$mean
    delta = summary[summary$variable == paste0('muDelta[', i_vt, ']'),]$mean
    tau = summary[summary$variable == paste0('muNDT[', i_vt, ']'),]$mean
    upper_bound = ifelse(s==4, upper_bound_Study_1, Inf)
    sample = WienR::sampWiener(1400, alpha, delta, beta, tau, response='both', bound=upper_bound)
    sim_data = data.frame(
      resp = ifelse(sample$response == "upper", 1, 0),
      rt = sample$q)
    if(s %in% c(2,3)) { # for censored models
      sim_data = sim_data[sim_data$rt < upper_bound_Study_1,]
    }
    all_data = rbind(
      all_data,
      data.frame(
        all_conds = index,
        resp = ifelse(sample$response == "upper", 1, 0),
        rt = sample$q*1000,
        model = s))
    
    obs_freqs[s, index] = round(mean(sim_data$resp), 2)
    mean_rts[s, index] = round(mean(sim_data$rt), 2)
    index = index + 1
  }
}

obs_freqs = rbind(obs_freqs, data_rts_freq$obs_freq)
row.names(obs_freqs) = c('basic', 'cens', 'cens prob', 'trunc', 'true_data')
names(obs_freqs) = c('W/NG', 'B/NG', 'W/G', 'B/G')

mean_rts = rbind(mean_rts, data_rts_freq$mean_rt)
row.names(mean_rts) = c('basic', 'cens', 'cens prob', 'trunc', 'true_data')
names(mean_rts) = c('W/NG', 'B/NG', 'W/G', 'B/G')

# Study 1 beobachtete relative Häufigkeiten
# und mittlere Reaktionszeiten
obs_freqs
mean_rts
#


####Plot RT-Histogramme
all_data_plot = all_data[all_data$rt < upper_bound_Study_1*1000,]

data_no_na = data %>%
  drop_na() %>%
  select(conditionRaceObj, resp0DS1S, rt) %>%
  mutate(model = 0) %>%
  rename(resp = resp0DS1S, all_conds = conditionRaceObj)

all_data_plot = rbind(all_data_plot, data_no_na)
all_data_plot$model = factor(all_data_plot$model)
levels(all_data_plot$model) = c('0' ='data', '1' = 'basic','2' = 'cens.', 
                                '3' = 'cens. with CCDF', '4' = 'trunc.')



compare_model = 'basic'
compare_model = 'cens.'
compare_model = 'cens. with CCDF'
compare_model = 'trunc.'
#pdf(file.path(base_dir, paste0('plots/Study_1_Hist_RT_', compare_model, '.pdf')), width=8, height=5)
ggplot(all_data_plot[all_data_plot$model %in% c('data', compare_model),], 
       aes(x=rt, color=model, fill=model, group=model)) +
  geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2") +
  xlab("RT") +
  ylab("") +
  ylim(0, 80) +
  facet_wrap(~all_conds, labeller = as_labeller(c('1'='W/NG', '2'='B/NG', '3'='W/G', 
                                                  '4'='B/G'))) 
#dev.off()
#


plot_liste = list()
models = c('basic', 'cens.', 'cens. with CCDF', 'trunc.')
colors_plots = c("green", "black", "red", "blue", "violet")
for(i in 1:4) {
  compare_model = models[i]
  plot_liste[[i]] = ggplot(all_data_plot[all_data_plot$model %in% c('data', compare_model),], 
                           aes(x=rt, color=model, fill=model, group=model)) +
    geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
    scale_color_manual(values=c(colors_plots[1], colors_plots[i+1]))+
    scale_fill_manual(values=c(colors_plots[1], colors_plots[i+1])) +
    xlab("RT") +
    ylab("") +
    ylim(0, 270) +
    theme(legend.position='none')
}


plot_for_legend = ggplot(all_data_plot, 
                         aes(x=rt, color=model, fill=model, group=model)) +
  geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
  scale_color_manual(values=colors_plots, name='Model:')+
  scale_fill_manual(values=colors_plots, name='Model:') +
  theme(legend.position='bottom') 


plot = aplot::plot_list(plot_liste[[1]], plot_liste[[2]], plot_liste[[3]], plot_liste[[4]])
#plot
leg = get_legend(plot_for_legend)
plot_legend = as_ggplot(leg)
plot_with_legend = aplot::plot_list(plot, plot_legend, nrow=2, heights=c(95,5))
#pdf(file.path(base_dir, paste0('plots/Study_1_Hist_RT_all_models.pdf')), width=8, height=5)
plot_with_legend
#dev.off()
#


