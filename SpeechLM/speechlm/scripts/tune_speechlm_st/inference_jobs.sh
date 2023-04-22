unset WORLD_SIZE
set -e -o pipefail -u

## codebase setup
if [ ! -d ~/code/SpeechT5/SpeechLM ]; then
    mkdir -p ~/code && cd ~/code && git clone https://zz12375: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@github.com/zz12375/SpeechT5
    cd SpeechT5/SpeechLM && git submodule update --init fairseq && pip install -e ./fairseq
    pip install editdistance
    pip install sacrebleu==1.5.1
fi
cd ~/code/SpeechT5/SpeechLM 

# gen_set=test
# for lang in ar de ca tr; do
#     data=/valle/v-ziqzhang/dataset/CommonVoice/v4/en/en-${lang}
#     for seed in seed2 seed3; do
#         modeldir=/valle/v-ziqzhang/data/speechulm/exp/finetune_covost/large_speechlmp_32gpu_4accum/legacy_en${lang}_${seed}_bz3.6m_lr1e-4
#         model=$modeldir/checkpoint_avgnbest.pt
#         if [ ! -f $model ]; then
#             python fairseq/scripts/average_checkpoints.py --inputs $modeldir/checkpoint.best_accuracy*.pt --output $model
#         fi
#         for beam_size in 5 10; do 
#             bash speechlm/scripts/tune_speechlm_st/inference_large.sh $model $data $lang $gen_set $beam_size
#         done
#     done
# done

gen_set=test
for lang in ar de ca tr; do
    data=/valle/v-ziqzhang/dataset/CommonVoice/v4/en/en-${lang}
    for seed in seed2 seed3; do
        modeldir=/valle/v-ziqzhang/data/speechulm/exp/finetune_covost/base_speechlmh_32gpu_1accum/legacy_en${lang}_${seed}_bz3.2m_lr1e-4
        model=$modeldir/checkpoint_best.pt
        for beam_size in 5 10; do 
            bash speechlm/scripts/tune_speechlm_st/inference_large.sh $model $data $lang $gen_set $beam_size
        done
    done
done
