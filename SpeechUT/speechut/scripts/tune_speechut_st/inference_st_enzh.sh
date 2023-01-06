# ####################################
# SpeechUT Base model #
# ####################################
[ $# -lt 2 ] && echo "Usage: $0 <model_path> <data_dir> [gen-set=dev] [beam_size=10] [lenpen=1.0]" && exit 0
[ ${PWD##*/} != SpeechUT ] && echo "Error: dir not match! Switch to SpeechUT/ and run it again!" && exit 1

model_path=$1
DATA_DIR=$2
gen_set=$3
beam_size=$4
lenpen=$5
[ -z $gen_set ] && gen_set="emime_test_1"
[ -z $beam_size ] && beam_size=10
[ -z $lenpen ] && lenpen=1
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

CODE_ROOT=${PWD}
results_path=$src_dir/decode_${cpt}_beam${beam_size}/${gen_set}
[ ! -d $results_path ] && mkdir -p $results_path

python $CODE_ROOT/fairseq/fairseq_cli/generate.py $DATA_DIR \
    --user-dir $CODE_ROOT/speechut \
    --label-dir ${DATA_DIR} \
    --labels '["zh.ipa"]' \
    --single-target \
    --gen-subset ${gen_set} \
    --batch-size 1 \
    --num-workers 0 \
    \
    --task joint_sc2t_pretraining \
    --add-decoder-target \
    --fine-tuning \
    --pad-audio \
    --random-crop \
    \
    --path ${model_path} \
    --results-path $results_path \
    \
    --beam ${beam_size} \
    --lenpen $lenpen \
    --scoring sacrebleu --max-len-a 0.00078125 --max-len-b 200 \



echo $results_path
tail -n 1 $results_path/generate-*.txt
cat $results_path/generate-*.txt | grep "^D-" | cut -d'-' -f2- | sort -nk1 | cut -f3- > $results_path/generate_semantic.txt
echo $results_path/generate_semantic.txt
sleep 1s
