#!/bin/bash
rm -r eval_*

if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    echo ${CUDA_VISIBLE_DEVICES}
    export CUDA_VISIBLE_DEVICES="0"
fi
echo ${CUDA_VISIBLE_DEVICES}

python main.py