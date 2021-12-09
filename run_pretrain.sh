savePrefix='./models/pretrain'
epsStart=0.1
epsEnd = 0.1
epsDecay = 1000000
rlmodel='DQN'
restore='./model_weights_a'

if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

python main.py \
    --savePrefix ${savePrefix} \
    --epsStart ${epsStart} \
    --epsEnd ${epsEnd} \
    --epsDecay ${epsDecay} \
    --rlmodel ${rlmodel} \
    --restore ${restore} \