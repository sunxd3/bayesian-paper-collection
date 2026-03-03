#Analyze results from study 3. We want to report:
# Coverage (HDI Tabelle)
# Correlations: posterior median with true value
# SBC: rank histograms, chi-square test
# Table: Param, Correlation, Coverage 50%, Coverage 95%

#basic: truncated with 80% of the data truncated (variable upper_bound) -> conceptionally not good, 
  # from here .91 derived: mean of all upper_bounds in this study
#basic_cens: censored with .91
#basic_fix_cdf1_cdf0_log_sum_exp: truncated with .91


suppressPackageStartupMessages(library(tidyverse))
library(grid) #for textGrob()
library(gridExtra)
library(bayesplot)
library(bayestestR)
library(mcmcse)
library(rlist)  #for list.rbind()
library(xtable) #for  xtable()

L = 399

folder = '_basic_cens' 
#folder = '_basic_fix_cdf1_cdf0_log_sum_exp'
data_dir =  paste0("S3_truncated", folder)

# Preparation of the data
names_params = c("a", "v1", "v2", "t0", "w", "sv", "sw", "st0")
pars_params = list("a"="a", "w"="w", "t0"="t0", "sv"="sv", "sw"="sw", "st"="st0")
pars_summary  = list("a"="a", "w"="zr_m", "v1"="v_m[1]", "v2"="v_m[2]", "t0"="t0_m", "sv"="v_s", "sw"="zr_s", "st"="t0_s")


if(grepl('basic', folder, fixed = T)) {
  names_params = names_params[1:5]
  pars_params = pars_params[1:3]
  pars_summary  = pars_summary[1:5]
}

results_dir = paste0("results", folder, '/plots')
results = read.csv(file.path(results_dir, "results.csv"))
with_lp = 1
SBC_lp = read.csv(file.path(results_dir, "lp_ranks.csv"))



# Chi-square and Fisher's combined probability test
#alpha level: 5% -> if p > .05 -> uniformly distributed
{
chi_p_values = list()
chi_square = list()
for (t in c(100, 500)) { 
  #for all parameters
  for (i in 1:(length(names_params)+with_lp)) {
    if (i <= length(names_params)) {
      ranks = results %>%
        filter(parameter == names_params[i] & trial_number == t) %>%
        dplyr::select(rank)
    } else {
      ranks = SBC_lp %>%
        filter(trial_number == t) %>%
        dplyr::select("lp_rank")
    }
    ranks = ranks[!(is.na(ranks))]
    
    table_ranks = merge(as.data.frame(0:399), as.data.frame(table(ranks)), by.x = c("0:399"), by.y=c("ranks"), all.x=TRUE)
    table_ranks[is.na(table_ranks)] = 0 
    table_ranks_binned = c()
    for (j in 1:((L+1)/4)) {
      table_ranks_binned[j] = table_ranks$Freq[(j-1)*4+1] + table_ranks$Freq[(j-1)*4+2] + 
        table_ranks$Freq[(j-1)*4+3] + table_ranks$Freq[(j-1)*4+4]
    }
    table_ranks_binned = as.data.frame(table_ranks_binned)
    colnames(table_ranks_binned) = c("obs")
    exp = as.data.frame(rep(1/100,100))
    colnames(exp) = c("exp")
    
    chi_p_values = append(chi_p_values, chisq.test(x = table_ranks_binned$obs, p = exp$exp, simulate.p.value = FALSE)[3]) #[1:3] for the chi-square values as well
    chi_square = append(chi_square, chisq.test(x = table_ranks_binned$obs, p = exp$exp, simulate.p.value = FALSE)[1]) #[1:3] for the chi-square values as well
  }  
  #for the log-PDF

} 
if (with_lp == 1) {
  names(chi_p_values) = c(paste0(names_params, "_100"), "log_PDF_100", paste0(names_params, "_500"), "log_PDF_500")
} else {
  names(chi_p_values) = c(paste0(names_params, "_100"), paste0(names_params, "_500"))
}
chi_p_values
#die folgenden beiden Werte kommen ins Paper!
pchisq(sum(-2*log(unlist(chi_p_values))),24) #2* Anzahl Parameter: bei 12 sind es 24
sum(-2*log(unlist(chi_p_values)))

if (with_lp == 1) {
  names(chi_square) = c(paste0(names_params, "_100"), "log_PDF_100", paste0(names_params, "_500"), "log_PDF_500")
} else {
  names(chi_square) = c(paste0(names_params, "_100"), paste0(names_params, "_500"))
}
chi_square
#this sum should be near one of the products below (depending on the number of parameters included)
sum(unlist(chi_square))
#test whether the sum of all chi-square values is near the sum of all degrees of freedom
99*18 # 2*9 (8 params + lpdf)
99*12 # 2*6 (5 params + lpdf)
99*10
}


# select trial number to save SBC-plots
trial_num = 100
save_plot = 0 #1 to save, 0 to not save and only plot here
{
names_in_plots = list("a"="a", "v1"=expression(v[1]), "v2"=expression(v[2]), 
                      "t0"=expression(t[0]), "w"="w", "sv"=expression(s[v]), 
                      "sw"=expression(s[w]), "st0"=expression(s[t0]), 
                      "log_PDF"="log-likelihood")


if (!dir.exists(results_dir)) {
  dir.create(results_dir, showWarnings = FALSE)
}

if(save_plot == 1) {
  if(with_lp == 1) {
    pdf(paste0(results_dir, "/SBC", folder, '_', trial_num, "_with_lPDF.pdf"), width=11.7, height=8.3)  #set path correspondingly
  } else {
    pdf(paste0(results_dir, "/SBC", folder, '_', trial_num, ".pdf"), width=11.7, height=8.3)  #set path correspondingly
  }
}
#par(mfrow=c(3,3), mar = c(3.5,3,2.5,3))
par(mfrow=c(3,2), mar = c(3.5,3,3.1,3))
for (i in 1:(length(names_params)+with_lp)) {
  if (i <= length(names_params)) {
    par = names_params[i]
    ranks = results %>%
      filter(parameter == names_params[i] & trial_number == trial_num) %>%
      dplyr::select(rank)
  } else {
    par = "log_PDF"
    ranks = SBC_lp %>%
      filter(trial_number == trial_num) %>%
      dplyr::select("lp_rank")
  }
  par_chi = paste0(par, "_", trial_num)
  ranks = ranks[!(is.na(ranks))]
  breaks = seq(0, (L+1), 4) - 0.5
  low = qbinom(0.005, length(ranks), prob=1/(length(breaks) - 1))
  mid = qbinom(0.5, length(ranks), prob=1/(length(breaks) - 1))
  up = qbinom(0.995, length(ranks), prob=1/(length(breaks) - 1))
  font_size_cex = 1.2
  hist(ranks, main=names_in_plots[[par]], font.main=1, breaks=breaks, 
       axes=F, col="white", border=NA, ylab="", xlab="", cex.main=2.2)
  polygon(c(-15, 0, -15, L+1+15, L+1, L+1+15), 
          c(low, mid, up, up, mid, low), 
          col=rgb(222/255,222/255,222/255), border=NA)
  segments(0, mid, L+1,mid, lwd=2, col=rgb(153/255,153/255,153/255))
  hist(ranks, breaks=breaks, add=T, axes=F, col=rgb(163/255,79/255,79/255), 
       border="darkred", ylab="", xlab="")
  mtext(expression(chi^2), side=3, line=1, at=300, cex=font_size_cex)
  mtext(paste0(" = ", chi_square[[par_chi]][[1]]), side=3, line=1.2, at=310, cex=font_size_cex, adj=0)
  mtext(paste0("p"), side=3, line=-0.1, at=297, cex=font_size_cex)
  mtext(paste0(" = ", str_remove(round(chi_p_values[[par_chi]][1],3),"^0+")), 
        side=3, line=-0.1, at=310, cex=font_size_cex, adj=0)
  axis(1)
}
title(outer = TRUE, cex.sub =2.3, line=-1,
      sub = "Rank Statistic")
title(outer = TRUE, cex.lab =2.3, line=-2,
      ylab = "Frequencies")

if (save_plot == 1) {
  dev.off()
}
par(mfrow=c(1,1))
}
#









#Coverage: Percentage true value in HDI  AND
#Mean-MCSE: mean(SD/sqrt(NEff))
{
percentage_HDI = results %>%
  mutate(in50 = ifelse(true <= `X50..HDI_high` & true >= `X50..HDI_low`, 1, 0)) %>% 
  mutate(in95 = ifelse(true <= `X95..HDI_high` & true >= `X95..HDI_low`, 1, 0)) %>% 
  drop_na() %>%
  group_by(parameter, trial_number) %>%
  summarize(in50 = mean(in50), in95 = mean(in95), .groups = "drop") %>%
  arrange(trial_number)

percentage_HDI
}



#Correlations: between posterior median and true vlaue
{

correlations = results %>%
  group_by(parameter, trial_number) %>%
  drop_na() %>%
  nest() %>%
  mutate(correlation = map(data, ~cor(.x$true, .x$`X50.`, method="pearson"))) %>%
  unnest(correlation, .drop = TRUE) %>%
  select(parameter, trial_number, correlation)

correlations$correlation = round(correlations$correlation, 2)
correlations
}






#look at the plot: median and true
param = unique(results$parameter)[1]
trial_num = 500
save_plot_2 = 1 #1 to save, 0 to not save and only plot here
{
if (save_plot_2 == 1) {
  pdf(paste0(results_dir, "/true_median", folder, '_', trial_num, ".pdf"), width=11.7, height=8.3)  #set path correspondingly
}
results %>%
  filter(trial_number == trial_num) %>%
  ggplot(aes(x=true, y=`X50.`)) +  #y=`X50.`, or mean
  geom_point(shape=1, colour = "darkred", fill = "white", size = 1, stroke = 0.5) +
  geom_abline(slope=1, intercept=0, alpha=.6) +
  ylab('Median') +
  xlab('True Value') +
  facet_wrap(~factor(parameter, levels=c('a', 'v1', 'v2', 't0', 'w')), scales='free') +
  theme(axis.title=element_text(size=20), axis.text=element_text(size=15),
        strip.text=element_text(size=20))

if (save_plot_2 == 1) {
  dev.off()
}
}






#make the table that will be reported: Correlations, Coverage, MCSE
report_table = merge(x=correlations, y=percentage_HDI, 
                     by.x=c("parameter", "trial_number"), 
                     by.y=c("parameter", "trial_number"), all=T, sort=F)
report_table = merge(report_table, mean_MCSE,
                     by.x=c("parameter"),
                     by.y=c("parameter"), all=T, sort=F)
report_table = select(report_table, c("parameter", "correlation", "in50", "in95", "mean_MCSE"))
names(report_table) = c("Par.", "Corr.", "50%", "95%", "MCSE")
row.names(report_table) = report_table$Par.
report_table = select(report_table, c("Corr.", "50%", "95%", "MCSE"))
report_table

print(xtable(report_table, type = "latex", digits=c(3,3,2,2,5)), file = paste0(data_dir, "/results/report_table_500.tex"))


#save workspace
#save.image(file = paste0(data_dir, "/results/workspace_", trial_number, ".RData"))
#


#Runtimes in minutes. 4 subsequent chains, each parallelized
times = results_a %>%
  select(timetosample, threads_per_chain) %>%
  mutate(threads_per_chain = as.factor(threads_per_chain))
ggplot(times, aes(x = timetosample/60, color=threads_per_chain, fill=threads_per_chain)) + 
  scale_color_manual(values=c("black", "white"), name = "Threads") +
  scale_fill_manual(values=c("#999999", "#E69F00"), name = "Threads") +
  geom_histogram(bins=30) +
  xlab('Computation time in minutes') +
  ylab('Count') 
 # vline_at(c(20, 44))
#  annotate(geom="text", x=300, y=580, label="mean(40)= 44",
#           color="#999999") +
#  annotate(geom="text", x=300, y=530, label="mean(80)= 21",
#         color="#999999")
  

table(times$threads_per_chain)

times_short = times %>%
  group_by(threads_per_chain) %>%
  summarize(mean_time = mean(timetosample/60), median_time=median(timetosample/60),
            min_time = min(timetosample/60), max_time = max(timetosample/60))



#min and max for NEff over all 4000 datasets
print(min(results$NEff))
print(max(results$NEff))

#check no N_equals for Modrak (2022)
print(sum(results$rank_equals))
