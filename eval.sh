#!/bin/sh
config_fn="$1"

# source the config file to get the following variables:
#   exp_id
#   script_dir, data_dir, exp_dir, visualize_dir
#   train_file, test_file
#   word_vecs_file, preload_dir
source $config_fn

# create output dir.
mkdir -p $visualize_dir/span $visualize_dir/surface $visualize_dir/coref

# generate visualization data.
echo "Generating data for visualization"
generate_visulize_data_script="$script_dir/generate_visulize_data.py"
exp_id="${exp_id}_test"
python $generate_visulize_data_script --exp_id $exp_id --data_file $test_file

# start server.
cd ./visualization
open http://localhost:8000/
./start.sh
