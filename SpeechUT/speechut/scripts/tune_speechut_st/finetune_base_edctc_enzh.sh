# ####################################
# SpeechUT Base model #
# ####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <cpt_tag> [mount=${PWD}] [world_size=8] [update_freq=2]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

w2v_path=$1
cpt=$2
mount=$3
world_size=$4
update_freq=$5
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=8
[ -z $update_freq ] && update_freq=6

DATA_DIR=/modelblob/users/v-ziqzhang/dataset/S2ST/st/en_zh/manifest/ipa_phones
CODE_ROOT=${PWD}

exp_name=${w2v_path%/*}
exp_name=${exp_name##*/}
MODEL_DIR="${mount}/exp/finetune_asrst_enzh/$exp_name/edctc200k_from_${cpt}_lr3e-5_bz1.2m_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechut/config/finetune_asr \
  --config-name speechut_base_asr_st \
  common.user_dir=$CODE_ROOT/speechut \
  \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.labels=["en.ipa","zh.ipa"] \
  model.w2v_path=${w2v_path} \
  \
  optimization.lr=[0.00003] \
  optimization.max_update=200000 \
  optimization.update_freq=[${update_freq}] \
  distributed_training.distributed_world_size=${world_size} \
  \
  dataset.max_tokens=1200000 \
  dataset.train_subset="gigast_train_all" \
  dataset.valid_subset="gigast_test" \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=edctc40k_from_${cpt}
