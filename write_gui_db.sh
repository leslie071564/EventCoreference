#!/bin/sh
doc_id="$1"
data_file="$2"

# create output dir.
visualize_dir=./visualization/corpus/$doc_id
mkdir -p $visualize_dir/span $visualize_dir/detail $visualize_dir/coref

# write visualization data to files.
generate_visualize_data_script=./generate_visualize_data.py
python $generate_visualize_data_script --gold --doc_id $doc_id --data_file $data_file --corpus_dir $visualize_dir

# start server.
cd ./visualization
open http://localhost:8000/
./start.sh
