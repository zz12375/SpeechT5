set -e -o pipefail -u

output_path=$1
mkdir -p ${output_path}/asr_results

echo "Manifest data ..."
if [ ! -f ${output_path}/asr_results/aishell3_test_sclt.tsv ]; then
    python ${HOME}/code/speech2speech/examples/audiolm/scripts/infer/wav2vec_manifest.py $output_path --ext wav
    mv ${output_path}/train.tsv ${output_path}/asr_results/aishell3_test_sclt.tsv 
    sleep 5s
fi
if [ ! -f ${output_path}/asr_results/aishell3_test_sclt.zh.ltr ]; then
    cp /modelblob/users/lozhou/speechdata/AISHELL-3/test/processed/aishell3_test_sclt.zh.ltr ${output_path}/asr_results/aishell3_test_sclt.zh.ltr
fi
wc -l ${output_path}/asr_results/aishell3_test_sclt.*


CODE_ROOT=${PWD}
DATA_DIR=${output_path}/asr_results
results_path=${output_path}/asr_results
gen_set="aishell3_test_sclt"
model_path=/modelblob/users/v-ziqzhang/data/speech2speech/speechut_zhen_model/exp/finetune_asr_zh/base_speechut4zhen_32gpu_1accum/ctc200k_from_360k_lr3e-5_bz1m_16gpu_2accum/checkpoint10.pt

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
decoding.results_path=${results_path} \
common_eval.results_path=${results_path} \
common_eval.path=${model_path}

sleep 5s
cat ${results_path}/hypo.word | sort -t'-' -nk 2 | cut -d'|' -f1 | sed 's| ||g' > ${results_path}/asr_results.zh
cat ${results_path}/aishell3_test_sclt.zh.ltr | sed 's| ||g' | sed 's/|/ /g' > ${results_path}/asr_reference.zh

sleep 3s
bleu=`sacrebleu ${results_path}/asr_reference.zh -i ${results_path}/asr_results.zh -m bleu -b -w 4 --tokenize zh`
echo "BLEU (4): $bleu"
