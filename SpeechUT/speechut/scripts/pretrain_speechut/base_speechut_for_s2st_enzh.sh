# ####################################
# SpeechUT Base model #
# ####################################
[ $# -lt 1 ] && echo "Usage: $0 [mount=${PWD}] [world_size=32] [update_freq=1]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1
mount=$1
world_size=$2
update_freq=$3
[ -z $mount ] && mount=${PWD}
[ -z $world_size ] && world_size=32
[ -z $update_freq ] && update_freq=1

CODE_ROOT=${PWD}
MODEL_DIR="${mount}/exp/pretrain/base_speechut4enzh_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

DATA_DIR="/modelblob/users/v-ziqzhang/dataset/librilight/chunkdata/ipa-phones"
TEXT_DATA_DIR="/modelblob/users/v-ziqzhang/dataset/S2ST/mt_new/zh-en/data-bin"

PYTHONPATH=$PWD/fairseq python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechut/config/pretrain \
  --config-name speechut_base_librispeech \
  common.user_dir=$CODE_ROOT/speechut \
  \
  task.labels='["ipa"]' \
  model.label_rate=50 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  task.text_cfg.tokens_per_sample=512 \
  \
  model.add_text_ctc=false \
  model.text_transformer.share_decoder_input_output_embed=true \
  criterion.u2t_ed_weight=1.0 \
  criterion.u2t_ctc_weight=0 \
  \
  dataset.train_subset=\"train+aic.en-zh,mt8corpus_filt01.en-zh\" \
  dataset.valid_subset=\"dev_clean+newstest2020.en-zh\" \
  dataset.num_workers=0 \
  dataset.max_tokens=1400000 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[${update_freq}] \
  checkpoint.save_interval=1 \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=base_speechut4enzh_${lang}_${world_size}gpu_${update_freq}accum

