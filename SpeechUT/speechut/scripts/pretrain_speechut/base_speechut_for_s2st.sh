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
MODEL_DIR="${mount}/exp/pretrain/base_speechut4zhen_${world_size}gpu_${update_freq}accum"
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR

DATA_DIR="/modelblob/users/v-ziqzhang/dataset/WenetSpeech/ipa_phones"
TEXT_DATA_DIR="/modelblob/users/v-ziqzhang/dataset/S2ST/mt_new/zh-en/data-bin"

PYTHONPATH=$PWD/fairseq python $CODE_ROOT/fairseq/fairseq_cli/hydra_train.py \
  --config-dir $CODE_ROOT/speechut/config/pretrain \
  --config-name speechut_base_librispeech \
  common.user_dir=$CODE_ROOT/speechut \
  \
  task.labels='["ipa10"]' \
  model.label_rate=100 \
  task.data=$DATA_DIR \
  task.label_dir=$DATA_DIR \
  task.text_cfg.text_data=$TEXT_DATA_DIR \
  +task.merge_pieces=True \
  task.text_cfg.tokens_per_sample=512 \
  \
  model.add_text_ctc=false \
  model.text_transformer.share_decoder_input_output_embed=true \
  criterion.u2t_ed_weight=1.0 \
  criterion.u2t_ctc_weight=0 \
  \
  dataset.train_subset=\"wenetspeech_train_l_chunk+aic.zh-en,mt8corpus_filt01.zh-en\" \
  dataset.valid_subset=\"wenet_dev+newstest2020.zh-en\" \
  dataset.num_workers=0 \
  dataset.max_tokens=1050000 \
  distributed_training.distributed_world_size=${world_size} \
  optimization.update_freq=[${update_freq}] \
  distributed_training.ddp_backend="legacy_ddp" \
  checkpoint.save_interval=1 \
  \
  common.tensorboard_logdir=$MODEL_DIR \
  checkpoint.save_dir=$MODEL_DIR \
  hydra.run.dir=$MODEL_DIR \
  hydra.job.name=base_speechut4en${lang}_${world_size}gpu_${update_freq}accum

