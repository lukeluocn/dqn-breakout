savePrefix='./models/DuelingDQN'
epsStart=1
rlmodel='DuelingDQN'
restore='None'

if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    export CUDA_VISIBLE_DEVICES=1
fi

python main.py \
    --savePrefix ${savePrefix} \
    --epsStart ${epsStart} \
    --rlmodel ${rlmodel} \
    --restore ${restore} \