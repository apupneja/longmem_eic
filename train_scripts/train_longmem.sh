task=pile
DATA_DIR=/data/zyu401_data/anirudh/longmem_data_1:/data/zyu401_data/anirudh/longmem_data_2:/data/zyu401_data/anirudh/longmem_data_3:/data/zyu401_data/anirudh/longmem_data_4
CKPT_DIR=/data/zyu401_data/anirudh/longmem_data/train_ckpt/re
PTM_PATH=/data/zyu401_data/anirudh/longmem_data/LongMem_public_checkpoints/gpt2_medium/checkpoint_last.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TORCH_DISTRIBUTED_DEBUG=DETAIL fairseq-train ${DATA_DIR}  \
    --save-dir ${CKPT_DIR} \
    --task language_modeling --arch transformer_lm_sidenet_gpt2_small \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --lr 2e-4 --lr-scheduler polynomial_decay \
    --weight-decay 0.01 \
    --save-interval-updates 10000 --sample-break-mode none \
    --tokens-per-sample 1024 \
    --batch-size 1 --total-num-update 100000 --seed 42 --update-freq 32 \
    --pretrained-model-path ${PTM_PATH} \
    --layer-reduction-factor 2 \
    --use-external-memory --memory-size 65536 \
    --k 64 --chunk-size 4 \
    --fp16 \
    --use-gpu-to-search \
    --no-token-positional-embeddings \
    --data-no-shuffle \
    --retrieval-layer-index 17 \
    --reload-ptm-layer \
    --disable-validation \
    --precompute-mem-layer 4 \
    --distributed-world-size 8