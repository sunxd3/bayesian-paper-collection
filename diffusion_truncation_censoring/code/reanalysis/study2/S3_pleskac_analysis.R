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

base_dir = 'analysis_pleskac/Reanalysis_Study_2'

fit_dir_b = file.path(base_dir, 'fits', 'fit_basicS2')
fit_dir_bc = file.path(base_dir, 'fits', 'fit_basicS2_cens')
fit_dir_bcp = file.path(base_dir, 'fits', 'fit_basicS2_cens_prob')
fit_dir_t = file.path(base_dir, 'fits', 'fit_basicS2_trunc_over_cdf0+cdf1')
jags_dir = 'analysis_pleskac/Reanalysis_Study_2'
#

{
fit_b = as_cmdstan_fit(paste0(fit_dir_b, '/fit-', 1:4, '.csv'))
fit_bc = as_cmdstan_fit(paste0(fit_dir_bc, '/fit-', 1:4, '.csv'))
fit_bcp = as_cmdstan_fit(paste0(fit_dir_bcp, '/fit-', 1:4, '.csv'))
fit_t = as_cmdstan_fit(paste0(fit_dir_t, '/fit-', 1:4, '.csv'))


summary_b = fit_b$summary()
summary_bc = fit_bc$summary()
summary_bcp = fit_bcp$summary()
summary_t = fit_t$summary()

min(summary_b$ess_bulk) #262
min(summary_bc$ess_bulk) #538
min(summary_bcp$ess_bulk) #409
min(summary_t$ess_bulk) #579

max(summary_b$rhat) #1.017
max(summary_bc$rhat) #1.007
max(summary_bcp$rhat) #1.008
max(summary_t$rhat) #1.01

summary_b[summary_b$ess_bulk < 400,] # 4
summary_b[summary_b$rhat > 1.01,] # 4

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


fit_JAGS = read_csv(file.path(jags_dir, 'S3_Study2_pleskac_JAGS_results.csv'))

min(summary_b_mu$ess_bulk) # 556
min(summary_bc_mu$ess_bulk) #1174
min(summary_bcp_mu$ess_bulk) #857
min(summary_t_mu$ess_bulk) #1548

max(summary_b_mu$rhat) # 1.00999
}




#rename variables: 11->1, 21->2, 31->3, 41->4, 12->5, 22->6, 32->7, 42->8
# 1st place: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G, 2nd place: 1 secure, 2 danger
# ist context so richtig kodiert? steht mit 0/1 in den Daten und 1 safe, 2 danger ...
#                  11->1,3;  21->2,4; 12->5,7; 22->6,8
# 1st place: 1 W, 2 B, 2nd place: 1 secure, 2 danger

{
#make HDI or read in overview
vars = summary_bc_mu$variable
fit_Stan = fit_bc
summary_Stan_mu = summary_bc_mu
fit_dir = fit_dir_bc

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

fit_JAGS = read_csv(file.path(jags_dir, 'S3_Study2_pleskac_JAGS_results.csv'))




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
overviewggJAGS = overviewggJAGS[match(overviewgg_bc$variable, overviewggJAGS$...1),]
overviewggJAGS = overviewggJAGS %>%
  mutate(model = 'JAGS') %>%
  dplyr::select(...1, Lower95, Median, Upper95, Mean, model) %>%
  rename(variable = ...1, mean = Mean, lower = Lower95, median = Median, upper = Upper95) 
}


#will be ordered alphabetically in plot
overview_gg = rbind(overviewgg_b, overviewgg_bc, overviewgg_bcp, 
                    overviewggJAGS, overviewgg_trunc)
model_selection = c(1, 2, 3, 4, 5)
#overview_gg = rbind(overviewggJAGS)
#model_selection = c(3)




#create overview tables for each parameter (combination)
{
number_models = length(model_selection)
overviewgg_aw = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muA', 'muB'),]
overviewgg_aw$facet = factor(rep(rep(c("w", "a"), each=4), number_models),
                             levels=c("w", "a"), ordered = TRUE)
overviewgg_aw$cond_1 = factor(rep(rep(c("W", "B"), 4), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_aw$cond_2 = factor(rep(rep(rep(c("Neutral", "Danger"), each=2), 2), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)

#check correct assignment of labels
#matrix [nWCon, nBtwn] muBeta;
#matrix [nWCon, nBtwn] muAlpha;
#nWCon = 2: 1 white, 2 black
#nBtwn = 2: 1 safe, 2 danger
#-> passt!



overviewgg_a = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muA'),]
overviewgg_a$facet = factor(rep(rep(c("a"), each=4), number_models),
                             levels=c("a"), ordered = TRUE)
overviewgg_a$cond_1 = factor(rep(rep(c("W", "B"), 2), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_a$cond_2 = factor(rep(rep(rep(c("Neutral", "Danger"), each=2), 1), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)




overviewgg_w = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muB'),]
overviewgg_w$facet = factor(rep(rep(c("w"), each=4), number_models),
                             levels=c("w"), ordered = TRUE)
overviewgg_w$cond_1 = factor(rep(rep(c("W", "B"), 2), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_w$cond_2 = factor(rep(rep(rep(c("Neutral", "Danger"), each=2), 1), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)





overviewgg_vt = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muD', 'muN'),]
overviewgg_vt$facet = factor(rep(rep(c("v", "t0"), each=8), number_models),
                          levels=c("v", "t0"), ordered = TRUE)
overviewgg_vt$cond_1 = factor(rep(rep(c("W", "B"), 8), number_models),
                            levels=c("W", "B"), ordered = TRUE)
overviewgg_vt$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 4), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)
overviewgg_vt$cond_3 = factor(rep(rep(rep(c("Neutral", "Danger"), each=4), 2), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)

#check correct assignment of labels
#matrix [nWConTarget, nBtwn] muDelta;
#matrix [nWConTarget, nBtwn] muNDT;
#nWConTarget = 4: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
#nBtwn = 2: 1 safe, 2 danger

overviewgg_v = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muD'),]
overviewgg_v$facet = factor(rep(rep(c("v"), each=8), number_models),
                             levels=c("v"), ordered = TRUE)
overviewgg_v$cond_1 = factor(rep(rep(c("W", "B"), 4), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_v$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 2), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)
overviewgg_v$cond_3 = factor(rep(rep(rep(c("Neutral", "Danger"), each=4), 1), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)

overviewgg_t = overview_gg[substr(overview_gg$variable, 1, 3) %in% c('muN'),]
overviewgg_t$facet = factor(rep(rep(c("t0"), each=8), number_models),
                             levels=c("t0"), ordered = TRUE)
overviewgg_t$cond_1 = factor(rep(rep(c("W", "B"), 4), number_models),
                              levels=c("W", "B"), ordered = TRUE)
overviewgg_t$cond_2 = factor(rep(rep(rep(c("No-Gun", "Gun"), each=2), 2), number_models),
                              levels=c("No-Gun", "Gun"), ordered = TRUE)
overviewgg_t$cond_3 = factor(rep(rep(rep(c("Neutral", "Danger"), each=4), 1), number_models),
                              levels=c("Neutral", "Danger"), ordered = TRUE)
}


#create plots for each parameter (combination)
{
colors = c("black", "red", "blue", "green", "violet", "yellow", "brown")[model_selection]
axis_title_size = 15
strip_text_size = 12.5
  param_plot_aw = ggplot(overviewgg_aw, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
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



param_plot_a = ggplot(overviewgg_a, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
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



param_plot_w = ggplot(overviewgg_w, aes(x = interaction(cond_1, cond_2, sep='&'), y = mean,  color=model)) +
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





param_plot_vt = ggplot(overviewgg_vt, aes(x = interaction(cond_1, cond_2, cond_3, sep='&'), y = mean,  color=model)) +
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




param_plot_v = ggplot(overviewgg_v, aes(x = interaction(cond_1, cond_2, cond_3, sep='&'), y = mean,  color=model)) +
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



param_plot_t = ggplot(overviewgg_t, aes(x = interaction(cond_1, cond_2, cond_3, sep='&'), y = mean,  color=model)) +
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

plot = aplot::plot_list(param_plot_w, param_plot_a, param_plot_v, param_plot_t)
plot_with_legend = aplot::plot_list(plot, plot_legend, nrow=2, heights=c(95,5))

#pdf(file.path(base_dir, "plots/Pleskac_Study2_all_models.pdf"), width=8, height=5)
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
data = read_csv('analysis_pleskac/Pleskac_Daten/Study2/dataTable_fromPleskacViaMail.csv') %>%
  rename('Subject' = 'subject', 'Race012B' = 'race', 'RT' = 'rt',
         'upper' = 'upperCensorLimit', 'Resp0NS1Sh' = 'resp',
         'Object0NG1G' = 'targetOb', 'Context1Safe2Danger' = 'targetDanger',
         'RaceObject' = 'conditionRT')
data$all_conds = as.numeric(interaction(data$RaceObject, data$Context1Safe2Danger)) #Levels: 1.0 2.0 3.0 4.0 1.1 2.1 3.1 4.1
true_out = table(data$ybin, data$all_conds)
true_anteil_out = (true_out['0',] + true_out['2',])#/n_cond
data_context_0 = data[data$Context1Safe2Danger == 0, ]
data_context_1 = data[data$Context1Safe2Danger == 1, ]
n_cond_context_0 = table(data_context_0$RaceObject)
n_cond_context_1 = table(data_context_1$RaceObject)
n_cond_context = c(n_cond_context_0, n_cond_context_1)

est_anteil_out = data.frame()
for(s in 1:4) {
  index = 1
  summary = list(summary_b_mu, summary_bc_mu, summary_bcp_mu, summary_t_mu)[[s]]
  for(j_vt in 1:2) { # runs over place2: context
    for(i_vt in 1:4) { # runs over place1
      i_ab = ifelse((i_vt == 1 | i_vt == 3), 1, 2)
      j_ab = j_vt
      N = n_cond_context[index]
#Levels: 1.0 2.0 3.0 4.0 1.1 2.1 3.1 4.1
#place1: nWConTarget = 4: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
#place2: nBtwn = 2: 1 safe, 2 danger
#alpha, beta: vary over place1: W/B, place2: Neutral/Danger
#für alpha, beta: 11 21 11 21 12 22 12 22
      alpha = summary[summary$variable == paste0('muAlpha[', i_ab, ',' , j_ab, ']'),]$mean
      beta = summary[summary$variable == paste0('muBeta[', i_ab, ',' , j_ab, ']'),]$mean
      delta = summary[summary$variable == paste0('muDelta[', i_vt, ',', j_vt, ']'),]$mean
      tau = summary[summary$variable == paste0('muNDT[', i_vt, ',', j_vt, ']'),]$mean
      p = 
        WienR::WienerCDF(1000, 1, alpha, delta, beta, tau)[[1]] - WienR::WienerCDF(0.631, 1, alpha, delta, beta, tau)[[1]] +
        WienR::WienerCDF(1000, 2, alpha, -delta, 1-beta, tau)[[1]]- WienR::WienerCDF(0.631, 2, alpha, -delta, 1-beta, tau)[[1]]
      est_anteil_out[s, index] = round(p*N,0)
      index = index + 1
    }
  }
}
names(est_anteil_out) = names(true_anteil_out)
row.names(est_anteil_out) = c('basic', 'cens', 'cens prob', 'trunc')
est_anteil_out#/1160
true_anteil_out#1160
rowSums(est_anteil_out)/(1160*8)
sum(true_anteil_out)/(1160*8)

#
# > est_anteil_out
#             1    2   3   4    5    6   7   8
#  basic     297  310  21  21  288  352  35  18
#  cens      430  430  32  23  464  535  43  24
#  cens prob 431  424  31  24  460  533  45  24
#  trunc     961 1022 146 130 1019 1148 226 117
# > true_anteil_out
#   1   2   3   4   5   6   7   8 
#  179 170  66  63 163 176  92  62 
}




##### Check: Das trunkierte Modell beschreibt die Daten im Antwortfenster besser beschreibt. 
#Die Vorhersagen für die beobachteten Häufigkeiten und mittleren Antwortzeiten von Gun und Tool Antworten 
#pro Bedingung berechnen. Eine einfache Möglichkeit, die zu berechnen, wäre Zufallswerte mit den jeweiligen 
#Modellparametern zu generieren und relative Häufigkeiten bzw. Mittelwerte zu berechnen. 
#Für die zensierten Modelle würde man dabei solche Zufallswerte mit Zeiten außerhalb des Fensters 
#rauswerfen, für das trunkierte Modell müsste man mit oberer Schranke = Antwortdeadline ziehen.
library(WienR)
upper_bound_Study_2 = 0.631
data = read_csv('analysis_pleskac/Pleskac_Daten/Study2/dataTable_fromPleskacViaMail.csv') %>%
  rename('Subject' = 'subject', 'Race012B' = 'race', 'RT' = 'rt',
         'upper' = 'upperCensorLimit', 'Resp0NS1Sh' = 'resp',
         'Object0NG1G' = 'targetOb', 'Context1Safe2Danger' = 'targetDanger',
         'RaceObject' = 'conditionRT')
data$all_conds = as.numeric(interaction(data$RaceObject, data$Context1Safe2Danger)) #Levels: 1.0 2.0 3.0 4.0 1.1 2.1 3.1 4.1
data_rts_freq = data %>%
  drop_na(RT) %>%
  group_by(all_conds) %>% 
  summarize(mean_rt = round(mean(RT/1000),2), obs_freq = round(mean(Resp0NS1Sh),2))

obs_freqs = data.frame()
mean_rts = data.frame()
all_data = data.frame()
for(s in 1:4) {
  index = 1
  summary = list(summary_b_mu, summary_bc_mu, summary_bcp_mu, summary_t_mu)[[s]]
  for(j_vt in 1:2) { # runs over place2: context
    for(i_vt in 1:4) { # runs over place1
      i_ab = ifelse((i_vt == 1 | i_vt == 3), 1, 2)
      j_ab = j_vt
      #Levels: 1.0 2.0 3.0 4.0 1.1 2.1 3.1 4.1
      #place1: nWConTarget = 4: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G
      #place2: nBtwn = 2: 1 safe, 2 danger
      #alpha, beta: vary over place1: W/B, place2: Neutral/Danger
      #für alpha, beta: 11 21 11 21 12 22 12 22
      alpha = summary[summary$variable == paste0('muAlpha[', i_ab, ',' , j_ab, ']'),]$mean
      beta = summary[summary$variable == paste0('muBeta[', i_ab, ',' , j_ab, ']'),]$mean
      delta = summary[summary$variable == paste0('muDelta[', i_vt, ',', j_vt, ']'),]$mean
      tau = summary[summary$variable == paste0('muNDT[', i_vt, ',', j_vt, ']'),]$mean
      upper_bound = ifelse(s==4, upper_bound_Study_2, Inf)
      sample = WienR::sampWiener(1000, alpha, delta, beta, tau, response='both', bound=upper_bound)
      sim_data = data.frame(
          resp = ifelse(sample$response == "upper", 1, 0),
          rt = sample$q)
      if(s %in% c(2,3)) { # for censored models
        sim_data = sim_data[sim_data$rt < upper_bound_Study_2,]
      }
      all_data = rbind(
        all_data,
        data.frame(
          all_conds = index,
          resp = ifelse(sample$response == "upper", 1, 0),
          RT = sample$q*1000,
          model = s))
      
      obs_freqs[s, index] = round(mean(sim_data$resp), 2)
      mean_rts[s, index] = round(mean(sim_data$rt), 2)
      index = index + 1
      
    }
  }
}
obs_freqs = rbind(obs_freqs, data_rts_freq$obs_freq)
row.names(obs_freqs) = c('basic', 'cens', 'cens prob', 'trunc', 'true_data')
names(obs_freqs) = c('W/NG/N', 'B/NG/N', 'W/G/N', 'B/G/N', 'W/NG/D', 'B/NG/D', 'W/G/D', 'B/G/D')

mean_rts = rbind(mean_rts, data_rts_freq$mean_rt)
row.names(mean_rts) = c('basic', 'cens', 'cens prob', 'trunc', 'true_data')
names(mean_rts) = c('W/NG/N', 'B/NG/N', 'W/G/N', 'B/G/N', 'W/NG/D', 'B/NG/D', 'W/G/D', 'B/G/D')

# Study 2 beobachtete relative Häufigkeiten
# und mittlere Reaktionszeiten
obs_freqs
mean_rts
#


####Plot RT-Histogramme
all_data_plot = all_data[all_data$RT < upper_bound_Study_2*1000,]

data_no_na = data %>%
  drop_na() %>%
  select(all_conds, Resp0NS1Sh, RT) %>%
  mutate(model = 0) %>%
  rename(resp = Resp0NS1Sh)

all_data_plot = rbind(all_data_plot, data_no_na)
all_data_plot$model = factor(all_data_plot$model)
levels(all_data_plot$model) = c('0' ='data', '1' = 'basic','2' = 'cens.', 
                                '3' = 'cens. with CCDF', '4' = 'trunc.')



compare_model = 'basic'
compare_model = 'cens.'
compare_model = 'cens. with CCDF'
compare_model = 'trunc.'
#pdf(file.path(base_dir, paste0('plots/Study_2_Hist_RT_', compare_model, '.pdf')), width=8, height=5)
ggplot(all_data_plot[all_data_plot$model %in% c('data', compare_model),], 
       aes(x=RT, color=model, fill=model, group=model)) +
  geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Dark2") +
  xlab("RT") +
  ylab("") +
  ylim(0, 80) +
  facet_wrap(~all_conds, labeller = as_labeller(c('1'='W/NG/N', '2'='B/NG/N', '3'='W/G/N', 
                                                  '4'='B/G/N', '5'='W/NG/D', '6'='B/NG/D', 
                                                  '7'='W/G/D', '8'='B/G/D'))) 
#dev.off()
#

plot_liste = list()
models = c('basic', 'cens.', 'cens. with CCDF', 'trunc.')
colors_plots = c("green", "black", "red", "blue", "violet")
for(i in 1:4) {
 compare_model = models[i]
 plot_liste[[i]] = ggplot(all_data_plot[all_data_plot$model %in% c('data', compare_model),], 
       aes(x=RT, color=model, fill=model, group=model)) +
  geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
   scale_color_manual(values=c(colors_plots[1], colors_plots[i+1]))+
   scale_fill_manual(values=c(colors_plots[1], colors_plots[i+1])) +
  xlab("RT") +
  ylab("") +
  ylim(0, 500) +
   theme(legend.position='none')
}


plot_for_legend = ggplot(all_data_plot, 
       aes(x=RT, color=model, fill=model, group=model)) +
  geom_histogram(position='identity', alpha=0.5, binwidth = 10) +
  scale_color_manual(values=colors_plots, name='Model:')+
  scale_fill_manual(values=colors_plots, name='Model:') +
  theme(legend.position='bottom') 


plot = aplot::plot_list(plot_liste[[1]], plot_liste[[2]], plot_liste[[3]], plot_liste[[4]])
#plot
leg = get_legend(plot_for_legend)
plot_legend = as_ggplot(leg)
plot_with_legend = aplot::plot_list(plot, plot_legend, nrow=2, heights=c(95,5))
#pdf(file.path(base_dir, paste0('plots/Study_2_Hist_RT_all_models.pdf')), width=8, height=5)
plot_with_legend
#dev.off()
#



# Posterior Predictive Check ... von johnson
{
library(WienR)
library(ggplot2)
library(dplyr)
library(tidyverse)
data_obs = read_csv('analysis_pleskac/Pleskac_Daten/Study2/Study2TrialData.csv') %>%
  mutate(cnd = as.numeric(interaction(RaceObject, Context1Safe2Danger))) %>%
  rename(subj=Subject, resp=Resp0NS1Sh, rt=RT) %>%
  dplyr::select(subj, cnd, resp, rt) %>%
  mutate(rt = rt/1000)


#rename variables: 11->1, 21->2, 31->3, 41->4, 12->5, 22->6, 32->7, 42->8
# 1st place: 1 W/NG, 2 B/NG, 3 W/G, 4 B/G, 2nd place: 1 secure, 2 danger
#                  11->1,3;  21->2,4; 12->5,7; 22->6,8
# 1st place: 1 W, 2 B, 2nd place: 1 secure, 2 danger
#        
pars_vars = list('muAlpha[1]' = 'muAlpha[1,1]', 'muAlpha[2]' = 'muAlpha[2,1]', 'muAlpha[3]' = 'muAlpha[1,2]', 'muAlpha[4]' = 'muAlpha[2,2]',
                 'muBeta[1]' = 'muBeta[1,1]', 'muBeta[2]' = 'muBeta[2,1]', 'muBeta[3]' = 'muBeta[1,2]', 'muBeta[4]' = 'muBeta[2,2]',
                 'muDelta[1]' = 'muDelta[1,1]', 'muDelta[2]' = 'muDelta[2,1]', 'muDelta[3]' = 'muDelta[3,1]', 'muDelta[4]' = 'muDelta[4,1]', 
                 'muDelta[5]' = 'muDelta[1,2]', 'muDelta[6]' = 'muDelta[2,2]', 'muDelta[7]' = 'muDelta[3,2]', 'muDelta[8]' = 'muDelta[4,2]',
                 'muNDT[1]' = 'muNDT[1,1]', 'muNDT[2]' = 'muNDT[2,1]', 'muNDT[3]' = 'muNDT[3,1]', 'muNDT[4]' = 'muNDT[4,1]',
                 'muNDT[5]' = 'muNDT[1,2]', 'muNDT[6]' = 'muNDT[2,2]', 'muNDT[7]' = 'muNDT[3,2]', 'muNDT[8]' = 'muNDT[4,2]')

data = c()
for(cnd in 1:8) {
  if (cnd %in% c(1, 2)) {
    j = cnd
  } else if (cnd %in% c(3, 4, 5, 6)) {
    j = cnd - 2
  } else if (cnd %in% c( 7, 8)) {
    j = cnd - 4
  }
  params_b = summary_b_mu[which(summary_b_mu$variable %in% 
                                   c(pars_vars[[paste0('muAlpha[', j, ']')]], pars_vars[[paste0('muBeta[', j, ']')]], 
                                     pars_vars[[paste0('muDelta[', cnd, ']')]], pars_vars[[paste0('muNDT[', cnd, ']')]])),] %>%
    select('variable', 'median')
  
  params_bc = summary_bc_mu[which(summary_bc_mu$variable %in% 
                             c(pars_vars[[paste0('muAlpha[', j, ']')]], pars_vars[[paste0('muBeta[', j, ']')]], 
                               pars_vars[[paste0('muDelta[', cnd, ']')]], pars_vars[[paste0('muNDT[', cnd, ']')]])),] %>%
    select('variable', 'median')
  
  params_bcp = summary_bcp_mu[which(summary_bcp_mu$variable %in% 
                             c(pars_vars[[paste0('muAlpha[', j, ']')]], pars_vars[[paste0('muBeta[', j, ']')]], 
                               pars_vars[[paste0('muDelta[', cnd, ']')]], pars_vars[[paste0('muNDT[', cnd, ']')]])),] %>%
    select('variable', 'median')
  
  sample_b = WienR::sampWiener(length(data_obs[data_obs$cnd == cnd,]$rt), params_b$median[params_b$variable==pars_vars[[paste0('muAlpha[', j, ']')]]], 
                             params_b$median[params_b$variable==pars_vars[[paste0('muDelta[', cnd, ']')]]], 
                             params_b$median[params_b$variable==pars_vars[[paste0('muBeta[', j, ']')]]], 
                             params_b$median[params_b$variable==pars_vars[[paste0('muNDT[', cnd, ']')]]], 
                             sv=0, sw=0, st0=0) 
  data = rbind(
    data,
    data.frame(
      subj = 1,
      cnd = cnd,
      resp = ifelse(sample_b$response == "upper", 1, 0),
      rt = sample_b$q,
      origin = 'pred_b'
    )
  )
  
  sample_bc = WienR::sampWiener(length(data_obs[data_obs$cnd == cnd,]$rt), params_bc$median[params_bc$variable==pars_vars[[paste0('muAlpha[', j, ']')]]], 
                               params_bc$median[params_bc$variable==pars_vars[[paste0('muDelta[', cnd, ']')]]], 
                               params_bc$median[params_bc$variable==pars_vars[[paste0('muBeta[', j, ']')]]], 
                               params_bc$median[params_bc$variable==pars_vars[[paste0('muNDT[', cnd, ']')]]], 
                               sv=0, sw=0, st0=0)
  
  data = rbind(
    data,
    data.frame(
      subj = 1,
      cnd = cnd,
      resp = ifelse(sample_bc$response == "upper", 1, 0),
      rt = sample_bc$q,
      origin = 'pred_bc'
    )
  )
  
  sample_bcp = WienR::sampWiener(length(data_obs[data_obs$cnd == cnd,]$rt), params_bcp$median[params_bcp$variable==pars_vars[[paste0('muAlpha[', j, ']')]]], 
                               params_bcp$median[params_bcp$variable==pars_vars[[paste0('muDelta[', cnd, ']')]]], 
                               params_bcp$median[params_bcp$variable==pars_vars[[paste0('muBeta[', j, ']')]]], 
                               params_bcp$median[params_bcp$variable==pars_vars[[paste0('muNDT[', cnd, ']')]]], 
                               sv=0, sw=0, st0=0)
  
  data = rbind(
    data,
    data.frame(
      subj = 1,
      cnd = cnd,
      resp = ifelse(sample_bcp$response == "upper", 1, 0),
      rt = sample_bcp$q,
      origin = 'pred_bcp'
    )
  )
  
  
}
data_obs$origin = 'obs'
data = rbind(data, data_obs)


p1 = data %>%
  filter(data$resp==1) %>%
  ggplot(aes(x=rt, colour=origin)) +
  geom_density() +
  ggtitle('Densities of reaction times for resp=1') +
  facet_wrap(~cnd)

p0 = data %>%
  filter(data$resp==0) %>%
  ggplot(aes(x=rt, colour=origin)) +
  geom_density() +
  ggtitle('Densities of reaction times for resp=0') +
  facet_wrap(~cnd)

p1
p0

#pdf(file.path(base_dir, "post_pred_1.pdf"))
p1
#dev.off()
}
