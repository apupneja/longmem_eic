task=pile
DATA_DIR=/data/zyu401_data/anirudh/longmem_data/data-bin/longmem
CKPT_DIR=/data/zyu401_data/anirudh/longmem_data/train_ckpt/25x/
PTM_PATH=/data/zyu401_data/anirudh/longmem_data/LongMem_public_checkpoints/gpt2_medium/checkpoint_last.pt

TORCH_DISTRIBUTED_DEBUG=DETAIL fairseq-train ${DATA_DIR}  \
    --save-dir ${CKPT_DIR} \
    --task language_modeling --arch transformer_lm_sidenet_gpt2_small \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --lr 2e-4 --lr-scheduler polynomial_decay \
    --weight-decay 0.01 \
    --save-interval-updates 1000 --sample-break-mode none \
    --tokens-per-sample 1024 \
    --batch-size 1 --total-num-update 100000 --seed 42 --update-freq 8 \
    --pretrained-model-path ${PTM_PATH} \
    --layer-reduction-factor 2 \
    --use-external-memory --memory-size 1638400 \
    --k 64 --chunk-size 4 \
    --fp16 \
    --use-gpu-to-search \
    --no-token-positional-embeddings \
    --data-no-shuffle \
    --retrieval-layer-index 17 \
    --reload-ptm-layer \
    --disable-validation \
    --precompute-mem-layer 0 \
    --distributed-world-size 8 \

# The --pre-trained-model path refers to the path to reproduced GPT-2-Medium checkpoints. You can find the downloading Google Drive url in README.
