#!/usr/bin/env bash
# ------------------------------------------------------------------------
# SeqFormer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

set -x # -x 执行指令后，会先显示该指令及所下的参数 https://juejin.cn/post/6960848214954082340

GPUS=$1 # 取输入命令的第一个参数
RUN_COMMAND=${@:2} # 取输入命令的第2个数到最后一个参数
if [ $GPUS -lt 8 ]; then # “-lt”表示小于，即如果GPUS < 8，运行then语句；否则运行else
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS} # 如果$GPUS_PER_NODE存在且不为空，GPUS_PER_NODE就是$GPUS_PER_NODE，如果$GPUS_PER_NODE不存在或为空，那么GPUS_PER_NODE就是$GPUS
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} # 同理若MASTER_ADDR不存在或者为空，则取127.0.0.1
MASTER_PORT=${MASTER_PORT:-"29501"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python3 ./launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}

