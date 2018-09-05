#!/bin/sh
config_fn="$1"
gpu_node="${@:2}"

# source the config file to get the following variables:
#   exp_id, repeat_exp
#   script_dir, data_dir, exp_dir, visualize_dir
#   train_file, test_file, word_vecs_file
source $config_fn
mkdir -p $exp_dir

train_script="$script_dir/train_joint.py"
eval_script="$script_dir/eval_joint.py"

overall_eval_log="$exp_dir/eval.txt"
rm -f $overall_eval_log
for i in $(seq $repeat_exp $END); 
do 
    # train.
    model_file="$exp_dir/model_$i.hdf5"
    train_options="--train_file $train_file --word_vectors_file $word_vecs_file --model_file $model_file"

    if [ -z "$gpu_node" ]
    then
        python $train_script $train_options
        #echo "pass"
    else
        python $train_script $train_options --gpus $gpu_node
    fi

    # evaluate
    eval_options="--test_file $test_file --model_file $model_file --word_vectors_file $word_vecs_file"
    eval_log="$exp_dir/eval_$i.txt"

    mkdir -p $visualize_dir/coref
    python $eval_script $eval_options > $eval_log
    tail -n 8 $eval_log
    tail -n 8 $eval_log >> $overall_eval_log

done

# aggregate the scores, and write the GUI data.
python ./aggregate_result.py $overall_eval_log
tail -n 5 $overall_eval_log 

