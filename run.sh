#!/bin/bash
if [ -d models ]; then
    read -p "Would remove the existing models/ dir. Proceed on? [y/n] " ins
    if [[ ${ins} == "y" ]]; then
        rm -r models
    else
        exit
    fi
fi
rm -r eval_*

savePrefix='./models/natural'
epsStart=1
rlmodel='DQN'
restore='None'

if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

python main.py \
    --savePrefix ${savePrefix} \
    --epsStart ${epsStart} \
    --rlmodel ${rlmodel} \
    --restore ${restore} \