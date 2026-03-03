library(ggbreak) 
library(patchwork)
library(reshape2)
library(latex2exp)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(svglite)
library(MASS)
get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/elife_revision')
# setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/pain_control_git/paper_submission/experiment1')
base_dir = 'model_fit_analysis/'
out_dir_sh = paste(base_dir,'output/cs_results/',sep='')

#===============================================#
### ---Confidence scaling plots condition - Figure 6----
#===============================================#
conf_results_dir = paste(base_dir,'output/extra_analysis/confidence_scaling_cond_Fig6/',sep='')
if(!file.exists(conf_results_dir)){dir.create(conf_results_dir,showWarnings = TRUE,recursive = TRUE)}
cond_names = c('HVHS', 'HVLS', 'LVHS', 'LVLS') ####-------2-------###########

pl = vector(mode = "list", length = length(cond_names))
fname_apped = '_cond'
nCol=1
hr = 6/1.5
wr = 16/1.5
col_list = c('#117733','#882255','#44AA99','#AA4499')
cond_renamed = c('HVHS', 'HVLS', 'LVHS', 'LVLS') 
cond_renamed = c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low')
sub_title_list = c('A','B','C','D')
title_size=20

indPars = read.table(paste(out_dir_sh,'/parIndSum_cond.csv',sep=''),header=TRUE,sep=',')
indPars = indPars[indPars$modelName=='m2_KF_sliced2_halfStudentT_reparam_confEs_norm' & indPars$parName=='cs',]
xx = seq(0,1,0.01)
noise_val=1
for (c in 1:length(cond_names)){
  indPars_cond = indPars[indPars$cond==cond_names[c],]
  trans_noise_df = data.frame()

  for (s in 1:length(unique(indPars_cond$PID))){
    trans_noise_df = rbind(trans_noise_df,cbind(rep(unique(indPars_cond$PID)[s],
                                                    length(xx)),xx,noise_val*exp((1-xx)/indPars_cond$value[s])))
  }
  colnames(trans_noise_df) = c('PID','xx','noise_val')
  trans_noise_df$PID=as.factor(trans_noise_df$PID)
  max_y = mean(trans_noise_df[trans_noise_df$xx==0,]$noise_val)
  slope_c = round((max_y-noise_val)/(0-1),3)
  
  pl[[c]]=ggplot(data=trans_noise_df,aes(x=xx,y=noise_val,col=PID),show.legend=FALSE,linetype='solid')+
    geom_line()+
    geom_smooth(aes(x=xx,y=noise_val),method=lm,inherit.aes = FALSE,color='black',size=2,formula='y~x')+
    annotate(geom="text",x=0.125, y=max_y+0.01, label=paste('slope: ',toString(slope_c),sep=''),color='black',size=7)+
    guides(col="none")+
    coord_cartesian(ylim = c(noise_val, noise_val*1.5),xlim = c(-0.05,1))+
    xlab('Confidence rating')+
    ylab('Noise scaling')+
    theme(axis.text.y=element_text(size=22))+
    theme(axis.text.x=element_text(size=22,angle=0))+
    theme(plot.subtitle = element_text(size = 25,face='bold',hjust=-0.07,vjust=))+
    theme(text=element_text(size=28),
          legend.text = element_text(size = 28),
          legend.title = element_text(size = 28,hjust=-1.75))+
    theme(legend.position = 'top')+
    theme(text = element_text(family = "Arial"))+
    theme(plot.title=element_text(hjust = 0.5,vjust=-0.75))+
    labs(subtitle = sub_title_list[c])
  
  if (length(cond_names)>0){
    pl[[c]] = pl[[c]]+ labs(title=cond_renamed[c])+
      theme(plot.title = element_text(size = 20,vjust=-4.5))
  }
  pl[[c]]=print(pl[[c]])
  
  
}
par_plot=grid.arrange(grobs=pl,ncol=4)
ggplot2::ggsave(paste(conf_results_dir,"conf_scaling",fname_apped,".svg",sep=''), plot=par_plot,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(conf_results_dir,"Figure6.svg",sep=''), plot=par_plot,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(conf_results_dir,"conf_scaling",fname_apped,".eps",sep=''), plot=par_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(conf_results_dir,"Figure6.eps",sep=''), plot=par_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
dev.off()


#===============================================#
### -------Model comparison - Figure 5------- 
#===============================================#
mcomp_results_dir = paste(base_dir,'output/extra_analysis/model_comp_cond_Fig5',sep='')
if(!file.exists(mcomp_results_dir)){dir.create(mcomp_results_dir,showWarnings = TRUE,recursive = TRUE)}
cond_names = c('HVHS', 'HVLS', 'LVHS', 'LVLS') ####-------2-------###########

pl = vector(mode = "list", length = length(cond_names))

if (length(cond_names)==4){
  fname_apped = '_cond'
  nCol=1
  hr = 7.5*1.45
  wr = 8
  col_list = c('#117733','#882255','#44AA99','#AA4499')
  cond_renamed = c('HVHS', 'HVLS', 'LVHS', 'LVLS') 
  cond_renamed = c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low')
  sub_title_list = c('A','B','C','D')
  title_size=20
}else{
  fname_apped = '_concat'
  nCol=1
  hr = 7.5/2.5
  wr = 8
  # hr = 7.5/1.5
  # wr = 10
  col_list = c('#5D3A9B')
  cond_renamed = c('Concatenated')
  sub_title_list = c('A')
  title_size=20
}

for (c in 1:length(cond_names)){
  mcomp_table = read.table(paste(out_dir_sh,'/model_comp',fname_apped,'_',cond_names[c],'.csv',sep=''),header=TRUE,sep=',')
  modelNames_list = mcomp_table$modelName
  renamedModels_list = vector(mode="character",length=length(modelNames_list))
  for (m in 1:length(modelNames_list)){
    tmp_name = ''
    m_name = modelNames_list[m]
    if (!grepl('.*np.*',m_name)){
      tmp_name ='e'
    }
    if (grepl('_KF.*',m_name)){
      tmp_name = paste(tmp_name,'KF',sep='')
    }
    if (grepl('_RL.*',m_name)){
      tmp_name = paste(tmp_name,'RL',sep='')
    }
    if (grepl('_VKF.*',m_name)){
      tmp_name = paste(tmp_name,'VKF',sep='')
    }
    if (grepl('m4.*',m_name)){
      tmp_name = 'Random'
    }
    renamedModels_list[m] = tmp_name
  }
  
  mcomp_table$sigma = paste(round(abs(mcomp_table$elpd_diff/mcomp_table$se_diff),3),"\U03C3",sep='')
  mcomp_table$sigma[1] = ''
  mcomp_table$modelName = renamedModels_list
  mcomp_table[mcomp_table$modelName=='VKF',]
  mcomp_table = mcomp_table[!mcomp_table$modelName=='VKF',]
  renamedModels_list=renamedModels_list[!renamedModels_list=='VKF']
  
  
  left_int=round(mcomp_table$elpd_diff-mcomp_table$se_diff)
  right_int=round(mcomp_table$elpd_diff+mcomp_table$se_diff)
  bpts = vector(mode = "numeric", length = length(left_int)-1)
  
  dr = diff(right_int)
  for (i in 1:(length(left_int)-1)){
    if (i==1){
      bpts[i] = left_int[i+1]
    }else if(i==(length(left_int)-1)){
      bpts[i] = right_int[i+1]*0.95
    }else{
      if (abs(dr[i])>40){
        bpts[i] = right_int[i+1]
      }else{
        bpts[i] = left_int[i+1]
      }
    }
  }
  
  mcomp_table$modelName = as.factor(renamedModels_list)
  renamedModels_list[1] = paste(renamedModels_list[1],'(winner)') 
  irislabs1 <- renamedModels_list
  
  
  pl[[c]]=mcomp_table %>%
    arrange(elpd_diff) %>% 
    mutate(modelName=factor(modelName, levels=modelName)) %>%
    ggplot(aes(x=elpd_diff,y=as.numeric(modelName),label=sigma),show.legend = TRUE)+
    geom_text(hjust=-0.1, vjust=-0.6,size=6)+
    geom_point(size=1)+
    geom_pointrange(aes(xmin = elpd_diff-se_diff, xmax = elpd_diff+se_diff),color=col_list[c],shape=19, fatten = 1.5, size = 3,linewidth=3)+
    scale_x_break(bpts,scale=1,space=0.95,expand=TRUE)+
    ylab('Model')+
    xlab('ELPD Difference')+
    labs(subtitle = sub_title_list[c])+
    theme(axis.title=element_text(size=title_size))+
    theme(axis.text.y=element_text(size=20))+
    theme(axis.text.x=element_text(size=17,angle=0))+
    theme(plot.subtitle = element_text(size = 20,face='bold',hjust=-0.01))+
    theme(text = element_text(family = "Arial"))+
    scale_y_continuous(breaks = length(renamedModels_list):1,
                       labels = renamedModels_list,
                       sec.axis = dup_axis())+
    xlim(left_int[length(left_int)]*1.04,3)+
    theme(axis.text.x.top = element_blank(),
          axis.ticks.x.top = element_blank(),
          axis.line.x.top = element_blank())+
    theme(panel.grid.minor=element_blank(),
          panel.background = element_rect(fill = '#E4E3E4'))
  
  if (length(cond_names)>0){
    pl[[c]] = pl[[c]]+ labs(title=cond_renamed[c])+
      theme(plot.title = element_text(size = 20,vjust=-4.5))
  }
  pl[[c]]=print(pl[[c]])
  
}
par_plot=grid.arrange(grobs=pl,ncol=nCol)
# ggplot2::ggsave(paste(mcomp_results_dir,"/model_comp2",fname_apped,".tiff",sep=''), plot=par_plot,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(mcomp_results_dir,"/model_comp",fname_apped,".eps",sep=''), plot=par_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(mcomp_results_dir,"/Figure5.eps",sep=''), plot=par_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(mcomp_results_dir,"/model_comp",fname_apped,".png",sep=''), plot=par_plot,dpi=300,width=614*wr,height=376*hr,units="px")

dev.off()

#===============================================#
### ------Example sequences - Figure 1 ------
#===============================================#
seq_results_dir = paste(base_dir,'output/extra_analysis/example_seqs_Fig1',sep='')
if(!file.exists(seq_results_dir)){dir.create(seq_results_dir,showWarnings = TRUE,recursive = TRUE)}

### Load and select data ---------
model_data=readRDS(paste(base_dir,'data/data_for_stan_lin_dropout','.rds',sep=''))
sub=15

### Intensities DF ---------
seq_df <- data.frame(matrix(ncol = 3, nrow = 0))
seq_df = rbind(seq_df,cbind(1:320,model_data$TransfIntesSeq[sub,],'Input intensity'),
               cbind(model_data$PercIndexArray[sub,],model_data$PainValsAll[sub,model_data$PercIndexArray[sub,]],'Perception intensity'),
               cbind(model_data$PredIndexArray[sub,],model_data$PainValsAll[sub,model_data$PredIndexArray[sub,]],'Prediction intensity'))

colnames(seq_df) = c('Trial','Value','Type')
seq_df$Trial = as.numeric(seq_df$Trial)
seq_df$Value = as.numeric(seq_df$Value)
seq_df$Type = as.factor(seq_df$Type)

### Confidence ratings DF ---------
cf_df <- data.frame(matrix(ncol = 3, nrow = 0))
cf_df = rbind(cf_df, cbind(model_data$PercIndexArray[sub,],model_data$PercConf[sub,],'Perception confidence'), 
              cbind(model_data$PredIndexArray[sub,],model_data$PredConf[sub,],'Prediction confidence'))

colnames(cf_df) = c('Trial','Value','Type')
cf_df$Trial = as.numeric(cf_df$Trial)
cf_df$Value = as.numeric(cf_df$Value)
cf_df$Type = as.factor(cf_df$Type)

### Plot intensities ---------
hr = 7.5/2
wr = 14
title_size=25
ticks_plot = seq(0,320,20); ticks_plot[1]=1; ticks_plot = sort(c(ticks_plot,c(81,161,241)))
label_yloc = round(max(seq_df$Value)*1.1)
fname_apped = '_cond'

pl = ggplot(data=seq_df,aes(x=Trial,y=Value),show.legend=TRUE)+
  geom_line(aes(x=Trial,y=Value,col=Type),linewidth = 1.5,show.legend = TRUE)+
  geom_line(data=seq_df[seq_df$Type=='Input',],aes(x=Trial,y=Value,col=Type),size = 2,show.legend = TRUE)+
  theme(plot.title = element_text(hjust = 0.5,size=30))+
  labs(col = '')+
  ylab('Intensity (a.u.)')+
  scale_x_break(c(80,81,160,161,240,241),scale=1,space=1.8,expand=FALSE)+
  scale_x_continuous(breaks = ticks_plot)+
  theme(axis.title=element_text(size=title_size))+
  theme(axis.text=element_text(size=25))+
  theme(text = element_text(family = "Arial"))+
  theme(axis.text.x.top = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.line.x.top = element_blank())+
  theme(text=element_text(size=28),
        legend.text = element_text(size = 26),
        legend.title = element_text(size = 26))+
  theme(legend.position = 'top')+
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill = '#E4E3E4'))+
  scale_colour_manual(values = c('#1E88E5','#D81B60','#FFC107'))+
  # scale_colour_manual(values = c('#B3CC0F','#FF0606','#65495D'))+
  ylim(0,100)

ggplot2::ggsave(paste(seq_results_dir,"/seq_resp_input",fname_apped,".eps",sep=''), plot=pl,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(seq_results_dir,"/Figure1C",".eps",sep=''), plot=pl,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")

### Plot confidence ---------
hr = 7.5/2
wr = 14
title_size=25
ticks_plot = seq(0,320,20); ticks_plot[1]=1; ticks_plot = sort(c(ticks_plot,c(81,161,241)))
label_yloc = round(max(seq_df$Value)*1.1)

pl = ggplot(data=cf_df,aes(x=Trial,y=Value),show.legend=TRUE,linetype='solid')+
  geom_line(aes(x=Trial,y=Value,col=Type),size = 1.5,show.legend = TRUE)+
  theme(plot.title = element_text(hjust = 0.5,size=30))+
  labs(col = '')+
  ylab('Confidence')+
  scale_x_break(c(80,81,160,161,240,241),scale=1,space=1.8,expand=FALSE)+
  scale_x_continuous(breaks = ticks_plot)+
  theme(axis.title=element_text(size=title_size))+
  theme(axis.text=element_text(size=25))+
  theme(text = element_text(family = "Arial"))+
  theme(axis.text.x.top = element_blank(),
        axis.ticks.x.top = element_blank(),
        axis.line.x.top = element_blank())+
  theme(text=element_text(size=28),
        legend.text = element_text(size = 26),
        legend.title = element_text(size = 26))+
  theme(legend.position = 'top')+
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill = '#E4E3E4'))+
  scale_colour_manual(values = c('#D81B60','#FFC107'))+
  # scale_colour_manual(values = c('#FF0606','#65495D'))+
  ylim(0,1)
  # xlim(0,321)

ggplot2::ggsave(paste(seq_results_dir,"/conf_ratings",fname_apped,".eps",sep=''), plot=pl,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(seq_results_dir,"/Figure1D",".eps",sep=''), plot=pl,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")



#===============================================#
### ------Confidence scaling demo - Figure 4 ------
#===============================================#
confdemo_results_dir = paste(base_dir,'output/extra_analysis/conf_scaling_demo_Fig4',sep='')
if(!file.exists(confdemo_results_dir)){dir.create(confdemo_results_dir,showWarnings = TRUE,recursive = TRUE)}

# scal_exp = round(c(0.1,0.3,0.6seq(1,20,length.out=10),100),2)
scal_exp = round(c(0.25,0.5,1,5,12.5,1000),2)

no_scales = length(scal_exp)
nRow = floor(sqrt(no_scales))
nCol = ceiling(sqrt(no_scales))
nCol=no_scales

exp_scale_fn <- function(xx,c_ind){
  yy = exp((1-xx)/scal_exp[c_ind]);
  return(yy)
}
xx=seq(0,1,0.01)


nc_points = 15
xi = 10
mu_x = 50
Nd = 200
results_dir = paste(base_dir,'paper_stuff',sep='')
pl = vector(mode = "list", length = no_scales)
letters_list = toupper(letters)
for (idx in 1:no_scales){
  conf_vect = round(seq(0,1,length.out=nc_points),2)
  yy2=exp_scale_fn(conf_vect,idx)
  
  
  draws = data.frame(mvrnorm(Nd,rep(mu_x,nc_points),xi*diag(yy2)))
  colnames(draws) = paste('',conf_vect,sep='')
  draws$draw=1:Nd
  
  draws_long <- melt(setDT(draws), id.vars = c("draw"), variable.name = "C")
  colnames(draws_long)[2]='C'
  draws_long$C = as.numeric(as.character(draws_long$C))
  draws_long = draws_long[draws_long$value>0,]
  if (idx==1 | idx==1+no_scales) {ylabel = 'Response'}else{ylabel=''}
  pl[[idx]] = ggplot(draws_long,aes(x=C,y=value,col=C))+geom_point(size=2,shape=23)+
    xlab('')+
    ylab(ylabel)+
    # labs(title = paste('C = ',scal_exp[idx],sep=''))+
    coord_cartesian(ylim = c(0, 125))+
    labs(subtitle = letters_list[idx],col='Confidence rating      ')+
    # theme(axis.title=element_text(size=title_size))+
    theme(axis.text.y=element_text(size=28))+
    theme(axis.text.x=element_text(size=28,angle=0))+
    theme(plot.subtitle = element_text(size = 28,face='bold',hjust=-0.07))+
    theme(text=element_text(size=28),
          legend.text = element_text(size = 28),
          legend.title = element_text(size = 28,hjust=-1.75))+
    theme(legend.position = 'top')+
    guides(colour = guide_colourbar(direction = "horizontal",barwidth = unit(10,'cm')))+
    theme(text = element_text(family = "Arial"))+
    theme(plot.title=element_text(hjust = 0.5,vjust=-0.75))+
    scale_x_continuous(breaks=c(0,0.5,1))
  
  xx=seq(0,1,0.001)
  yy=exp_scale_fn(xx,idx)
  df_temp = data.frame(cbind(xx,yy))
  if (idx==1 | idx==1+no_scales) {ylabel = 'Noise scaling'}else{ylabel=''}
  
  if (round(max(df_temp$yy))==1){ymax_lim=1.5}else{ymax_lim=round(max(df_temp$yy)*1.25)}
  ymax_lim = round(ymax_lim,1)
  pl[[idx+no_scales]] = ggplot(df_temp,aes(x=xx,y=yy))+geom_line(size=5)+
    xlab('Confidence rating')+
    ylab(ylabel)+
    labs(title = paste('C = ',scal_exp[idx],sep=''))+
    coord_cartesian(ylim = c(0, ymax_lim))+
    labs(subtitle = letters_list[idx+no_scales])+
    theme(axis.title=element_text(size=title_size))+
    theme(axis.text.y=element_text(size=28))+
    theme(axis.text.x=element_text(size=28,angle=0))+
    theme(plot.subtitle = element_text(size = 28,face='bold',hjust=-0.07))+
    theme(text=element_text(size=28),
          legend.text = element_text(size = 28),
          legend.title = element_text(size = 28,hjust=-0.25))+
    theme(text = element_text(family = "Arial"))+
    theme(plot.title=element_text(size=28,hjust = 0.5,vjust=0.75))+
    scale_x_continuous(breaks=c(0,0.5,1))
}

wr=15
hr=7.5
legend <- get_legend(pl[[1]])    
p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=nCol,top=legend,guide_legend(direction = 'horizontal',nrow=2, byrow=TRUE))
ggplot2::ggsave(paste(confdemo_results_dir,"/conf_vis",fname_apped,".png",sep=''), plot=eps_ar_plot,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(confdemo_results_dir,"/Figure4.eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(confdemo_results_dir,"/conf_vis",fname_apped,".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")


#===============================================#
### ------Plot RMSE accuracies - Figure 2 ------
#===============================================#
rmse_results_dir = paste(base_dir,'output/extra_analysis/rmse_performance_Fig2',sep='')
if(!file.exists(rmse_results_dir)){dir.create(rmse_results_dir,showWarnings = TRUE,recursive = TRUE)}
rmse_scores_df = read.table(paste(base_dir,"/output/cs_results/indpptRmseScore_cond.csv",sep=''),sep=',',header = TRUE)
fname_apped = ''

col_list = c('#117733','#882255','#44AA99','#AA4499')
cond_renamed = c('HVHS', 'HVLS', 'LVHS', 'LVLS') 
cond_label_names = c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low')
rmse_scores_df$cond = as.factor(rmse_scores_df$cond)
levels(rmse_scores_df$cond) = cond_label_names
dodge <- position_dodge(width = 0.9)
font_size = 25
sub_title_list = c('A','B','C','D')


hr = 7.5/1.5
wr = 12

pl = vector(mode = "list", length = 2)

pl[[1]] = ggplot(data=rmse_scores_df)+
  geom_violin(aes(x = cond, y = indppt_rmse_rate, fill= cond),position = dodge)+
  geom_boxplot(aes(x = cond, y = indppt_rmse_rate, fill= cond),width=0.2, col='white',fill='white',alpha=0.5,position = dodge) +
  geom_jitter(aes(x = cond, y = indppt_rmse_rate, fill=cond),col='black',shape=19,size=2,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
  scale_fill_manual(values = col_list,labels=cond_label_names)+
  theme(legend.position = 'bottom')+
  xlab('Condition')+
  ylab('Perception error')+
  labs(fill='',subtitle = sub_title_list[1])+
  theme(plot.subtitle = element_text(size = font_size,face='bold',hjust=-0.025))+
  theme(axis.title=element_text(size=font_size))+
  theme(axis.text=element_text(size=font_size))+
  theme(text = element_text(family = "Arial"))+
  theme(text=element_text(size=font_size),
        legend.text = element_text(size = font_size),
        legend.title = element_text(size = font_size))+
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill = '#E4E3E4'))+
  theme(axis.ticks.x = element_blank(),axis.text.x = element_blank())

pl[[2]] = ggplot(data=rmse_scores_df)+
  geom_violin(aes(x = cond, y = indppt_rmse_pred, fill= cond),position = dodge)+
  geom_boxplot(aes(x = cond, y = indppt_rmse_pred, fill= cond),width=0.2, col='white',fill='white',alpha=0.5,position = dodge) +
  geom_jitter(aes(x = cond, y = indppt_rmse_pred, fill=cond),col='black',shape=19,size=2,position=position_jitterdodge(jitter.width=0.3,dodge.width = 0.9))+
  scale_fill_manual(values = col_list,labels=cond_label_names)+
  theme(legend.position = 'none')+
  xlab('Condition')+
  ylab('Prediction error')+
  labs(fill='',subtitle = sub_title_list[2])+
  theme(plot.subtitle = element_text(size = font_size,face='bold',hjust=-0.025))+
  theme(axis.title=element_text(size=font_size))+
  theme(axis.text=element_text(size=font_size))+
  theme(text = element_text(family = "Arial"))+
  theme(text=element_text(size=font_size),
        legend.text = element_text(size = font_size),
        legend.title = element_text(size = font_size))+
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill = '#E4E3E4'))+
  theme(axis.ticks.x = element_blank(),axis.text.x = element_blank())

legend <- get_legend(pl[[1]])    
p_no_legend <- lapply(pl, function(x) x + theme(legend.position = "none"))
eps_ar_plot=grid.arrange(grobs=p_no_legend,ncol=2,bottom=legend,guide_legend(nrow=2, byrow=TRUE))

# par_plot=grid.arrange(grobs=pl,ncol=2)  
ggplot2::ggsave(paste(rmse_results_dir,"/rmse_performance",fname_apped,".eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")
ggplot2::ggsave(paste(rmse_results_dir,"/Figure2.eps",sep=''), plot=eps_ar_plot,device=cairo_ps,dpi=300,width=614*wr,height=376*hr,units="px")


#===============================================#
### ------Cast the individual parameter long tables for RM ANOVA analysis into wide format ------
#===============================================#

output_dir = paste(base_dir,'output/cs_results/',sep='')
rm_anova_pars_dir =paste(base_dir,'output/extra_analysis/supplement',sep='')
if(!file.exists(rm_anova_pars_dir)){dir.create(rm_anova_pars_dir,showWarnings = TRUE,recursive = TRUE)}

par_df = read.table(paste(output_dir,'parIndSum_cond.csv',sep=''),sep=',',header=TRUE)
par_df= par_df[par_df$modelName=='m2_KF_sliced2_halfStudentT_reparam_confEs_norm',] #winning model
par_df = par_df[,c(2,3,4,8)]

par_df_cast = dcast(par_df, PID + parName  ~ cond, value.var = "value",fun.aggregate = mean)
write.csv(par_df_cast,paste(rm_anova_pars_dir,'/parInd_cond_cast.csv',sep=''),row.names=FALSE)

#===============================================#
### ------ Calcuate more model diagnostistcs measures - revision------
#===============================================#
library(rstan)
library(posterior)
library(plyr)
if (Sys.info()["sysname"]=="Darwin"){
  # setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/pain_control_git/paper_submission/experiment1')
  setwd('/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/elife_revision/')
  rds_files_loc = '/Users/onyskj/Library/CloudStorage/OneDrive-UniversityofCambridge/CamNoxLab - OneDrive/tsl_paper/_MANUSCRIPT_SUBMISSION/RDS_fits/'
}else{
  rds_files_loc = ('/rds/project/rds-3IOyKgCQu4I/tsl_elife_rev_jao57/RDS_fits/')
  setwd('/rds/project/rds-3IOyKgCQu4I/tsl_elife_rev_jao57/')
}

cond_names = c('HVHS', 'HVLS', 'LVHS', 'LVLS') 
cond_names_labels = c('Vol. high - Stoch. high','Vol. high - Stoch. low','Vol. low - Stoch. high','Vol. low - Stoch. low')
model_names = list.files(paste(rds_files_loc,'HVHS/',sep=''))
model_names = gsub(paste0('_',paste(cond_names, collapse="|")),"", model_names)
model_names_long= gsub(paste0('_',paste(cond_names, collapse="|")),"", model_names)
model_names_short = gsub('_norm','',gsub('sliced2_halfStudentT_reparam_confEs_','',model_names_long))
model_names_short = gsub('RL','eRL',model_names_short); model_names_short = gsub('KF','eKF',model_names_short)
model_names_short = gsub('eRL_np','RL',model_names_short); model_names_short = gsub('eKF_np','KF',model_names_short)
model_names_short = gsub('_C','_Random',model_names_short)
model_names_short = gsub('m.*_','',model_names_short)
model_names_paper= model_names_short
model_names_short = mapvalues(model_names_short, from=model_names_short, to=c('m1','m2','m3','m4','m5'))
parameter_names <- list(m1=c("alpha", "cs", "E0", "eta", "gam"),
                        m2=c("alpha", "cs", "E0", "eta"),
                        m3=c("cs", "E0", "eps", "eta", "psi", "w0", "xi"),
                        m4=c("cs", "E0", "eta", "psi", "w0", "xi"),
                        m5=c("C", "cs", "eta"))
par_ind_latex_names = list(m1=c("$\\alpha","$C","$E^0","$\\xi","$\\gamma"),
                           m2=c("$\\alpha","$C","$E^0","$\\xi"),
                           m3=c("C","$E^0","$\\epsilon","$v","$s","$w^0","$\\xi"),
                           m4=c("C","$E^0","$v","$s","$w^0","$\\xi"),
                           m5=c("$R","$C","$\\xi"))


# model_names = model_names[1:2]
# cond_names = cond_names[1:2]

# iterate through models and conditions
param_ess_df = data.frame()
extra_diag_df = data.frame()
for (m in 1:length(model_names)){
  model_name = model_names[m]
  model_name_short = model_names_short[m]
  model_name_paper = model_names_paper[m]
  param_names=parameter_names[[model_name_short]]
  param_names_latex = par_ind_latex_names[[model_name_short]]
  print(paste0('model: ',model_name))
  for (c in 1:length(cond_names)){
    cond_name=cond_names[c]
    cond_name_label = cond_names_labels[c]
    print(paste0('cond: ',cond_name))
    tmp_rds_dir = paste(rds_files_loc,cond_name,'/',model_name,'_',cond_name,sep='')
    rds_file = dir(tmp_rds_dir,full.names = T, pattern='.*rds')
    stan_fit = readRDS(rds_file)
    # extra diagnostics - bfmi values, and divergent transitions counts
    low_bfmi = if (length(get_low_bfmi_chains(stan_fit))==1) get_low_bfmi_chains(stan_fit) else 0
    bfmi_vec = paste(round(get_bfmi(stan_fit),3), collapse=' ')
    div_num = get_num_divergent(stan_fit)
    tmp_row = cbind(model_name_paper,cond_name,low_bfmi,div_num,bfmi_vec)
    extra_diag_df = rbind(extra_diag_df,tmp_row)
    
    ## ess bulk tail for each parameter (individual level)
    for (p_ind in 1:length(param_names)){
      tmp_par = param_names[p_ind]
      tmp_par_latex = param_names_latex[p_ind]
      ess_bulk_val = round(ess_bulk(extract(stan_fit,permuted=TRUE)[[tmp_par]]),3)
      ess_tail_val = round(ess_tail(extract(stan_fit,permuted=TRUE)[[tmp_par]]),3)
      tmp_row_par = cbind(model_name_paper, cond_name, tmp_par, tmp_par_latex, ess_bulk_val, ess_tail_val)
      param_ess_df = rbind(param_ess_df,tmp_row_par)
    }
    
  }
}
write.csv(extra_diag_df,'model_fit_analysis/output/extra_analysis/extra_diag.csv',row.names = FALSE)
write.csv(param_ess_df,'model_fit_analysis/output/extra_analysis/ess_table.csv',row.names = FALSE)

