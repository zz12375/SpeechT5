#####################################
# SpeechUT Base model #
#####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

# DATA_DIR=/modelblob/users/v-ziqzhang/dataset/S2ST/asr/zh
model_path=$1
DATA_DIR=$2
gen_set=$3
[ -z $gen_set ] && gen_set="test_all14"
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}

results_path=$src_dir/decode_${cpt}_ctc/${gen_set}
[ ! -d $results_path ] && mkdir -p $results_path

export PYTHONPATH=$CODE_ROOT/fairseq/
python $CODE_ROOT/speechut/infer.py \
--config-dir $CODE_ROOT/speechut/config/decode \
--config-name infer_viterbi \
common.user_dir=$CODE_ROOT/speechut \
\
dataset.gen_subset=${gen_set} \
task.data=$DATA_DIR \
task.label_dir=$DATA_DIR \
task.labels="['zh.ltr']" \
task.normalize=false \
decoding.results_path=${output_path} \
common_eval.results_path=${results_path} \
common_eval.path=${model_path}

sleep 5s
cat ${results_path}/viterbi/hypo.word | sort -t'-' -nk 2 | cut -d'|' -f1 | sed 's| ||g' > ${results_path}/viterbi/asr_results.zh

spm_model="/modelblob/users/v-ziqzhang/dataset/S2ST/mt_new/zh-en/sentencepiece/sentencepiece_zh/spm_unigram32000.model"
spm_encode --model=$spm_model < ${results_path}/viterbi/asr_results.zh > ${results_path}/viterbi/asr_results.zh-en.zh
cp /modelblob/users/v-ziqzhang/dataset/S2ST/mt_new/zh-en/sentencepiece/data-bin/dict.{en,zh}.txt ${results_path}/viterbi/
cp /modelblob/users/v-ziqzhang/dataset/S2ST/mt_new/zh-en/sentencepiece/emime_test.zh-en.en ${results_path}/viterbi/asr_results.zh-en.en
echo "Processed data for mt in: ${results_path}/viterbi/asr_results.zh-en.{en,zh}"
