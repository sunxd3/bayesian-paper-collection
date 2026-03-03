suppressPackageStartupMessages(library(tidyverse))
library(optparse)

argument_list = list (
  make_option(c('-f', '--folder'), type='character', default='none',
              help='_basic, _basic_fix, _basic_cens', metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

output_dir = paste0("results", args$folder)
files = list.files(output_dir)


to_combine = substr(files, 1, 9) == "lp_ranks_"
SBC_lp = read_csv(file.path(output_dir, files[to_combine][[1]]))
length_data = length(SBC_lp$...1)/2

if (sum(to_combine) * length_data < 2000) {
  print('Noch nicht alle eingelesen!')
} else {
  # read in Stan files if complete and remove single group results
  print(args$folder)
  
  results = file.path(output_dir, files[to_combine]) %>% 
    lapply(read_csv, col_types=('dcc')) %>% 
    bind_rows
  print('Saving results') 
  write.csv(results, file.path(output_dir, "lp_ranks.csv"))
  #file.remove(file.path(output_dir, files[to_combine]))

}

