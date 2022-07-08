python3 ./train_ag.py --arch RNIA-Nets --batch_size 16 --gpu '0,1,2,3' --nepoch 1000 \
      --train_ps 256 \
      --train_gt_dir /home/amax/train/gt/  \
      --train_input_dir /home/amax/train/input/ \
      --val_gt_dir /home/amax/val_gt/ \
      --val_input_dir /home/amax/val_input/ \
      --embed_dim 64 --warmup --checkpoint 500 \
      --env RNIA --lr_initial 0.0001