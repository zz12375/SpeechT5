#####################################
# SpeechUT Base model #
#####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <data_dir> [gen-set=dev_clean,dev_other,test_clean,test_other]" && exit 1
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

model_path=$1
DATA_DIR=$2
gen_set=$3
[ -z $gen_set ] && gen_set="test"
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}

results_path=$src_dir/decode_${cpt}_ctc/${gen_set}
[ ! -d $results_path ] && mkdir -p $results_path

export PYTHONPATH=$CODE_ROOT/fairseq/
python $CODE_ROOT/speechut/infer_fa.py \
--config-dir $CODE_ROOT/speechut/config/decode \
--config-name infer_viterbi \
common.user_dir=$CODE_ROOT/speechut \
\
dataset.gen_subset=${gen_set} \
task.data=$DATA_DIR \
task.label_dir=$DATA_DIR \
task.labels="['zh.ipa']" \
task.normalize=false \
decoding.results_path=${output_path} \
common_eval.results_path=${results_path} \
common_eval.path=${model_path}

cat ${results_path}/viterbi/hypo.units | sort -t'-' -nk2 | cut -d'(' -f1 | sed 's| ii_3_E $||' > ${results_path}/${gen_set}.zh
cp $DATA_DIR/dict.zh.ipa.txt ${results_path}/dict.txt
echo "${results_path}/${gen_set}.zh"
