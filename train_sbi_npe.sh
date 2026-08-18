python3 train_sbi_npe.py \
    --data-dir bfield_dataset_100k \
    --out-dir sbi_npe_run \
    --device cuda \
    --learning-rate 2e-4 \
    --batch-size 4096 \
    --max-epochs 400 \
    --stop-after-epochs 200
