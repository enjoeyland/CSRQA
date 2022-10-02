#!/bin/env bash
#SBATCH -J csqa_jointlk_train         
#SBATCH -o log.csqa_jointlk_train.txt         # The output file on the screen is redirected to test.out
#SBATCH -e slurm-%j.err                       # The error output file on the screen is redirected to "slurm-%j.err" , %j will be replaced with "jobid"
#SBATCH -p compute                            # The partition for job submission is CPU
#SBATCH -N 1                                  # Job requests 1 node
#SBATCH --ntasks-per-node=1                   # The number of processes started by a single node is 1
#SBATCH --cpus-per-task=4                     # The number of CPU cores used by a single task is 4
#SBATCH --mem=2GB                  
#SBATCH -t 1-00:00:00                         # The maximum time a task can run is 1 hour
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1     # Single node uses 1 GPU card
# SBATCH -w gpu24

source ~/.bashrc

dt=`date '+%Y%m%d_%H%M%S'`


dataset="csqa"

model='roberta-large'
shift
shift
args=$@


elr="1e-5"
dlr="1e-3"
bs=64
mbs=2
n_epochs=30
num_relation=38 

k=5 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

save_dir_pref='saved_models'
mkdir -p $save_dir_pref
mkdir -p logs


for seed in 0; do
  python3 -u jointlk.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs -mbs $mbs --seed $seed \
      --num_relation $num_relation \
      --n_epochs $n_epochs --max_epochs_before_stop 10  \
      --train_adj data/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   data/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.pk \
      --train_statements  data/${dataset}/statement/train.statement.jsonl \
      --dev_statements  data/${dataset}/statement/dev.statement.jsonl \
      --test_statements  data/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir ${save_dir_pref}/${dataset}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args \
  > logs/train_${dataset}__enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt}.log.txt
done
