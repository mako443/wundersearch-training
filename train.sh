# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 512 --margin 0.15
# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 512 --margin 0.2
# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 512 --margin 0.25

# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 512 --margin 0.3

# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 1024 --margin 0.25

# python3 -m training.vse --epochs 8 --batch_size 32 --embed_dim 1024 --margin 0.25 --resize_image 250
# python3 -m training.vse --epochs 8 --batch_size 64 --embed_dim 1024 --margin 0.25 --resize_image 250
# python3 -m training.vse --epochs 8 --batch_size 128 --embed_dim 1024 --margin 0.25 --resize_image 250

# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 1024 --margin 0.25 --resize_image 250 
# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 1024 --margin 0.25 --resize_image 250 --lr_gamma 0.75

# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 2048 --margin 0.25 --resize_image 250
# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 2048 --margin 0.25 --resize_image 250 --lr_gamma 0.85

# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 4096 --margin 0.25 --resize_image 250 --lr_gamma 0.85

# python3 -m training.vse --epochs 32 --batch_size 64 --embed_dim 4096 --margin 0.25 --resize_image 250 --lr_gamma 0.85 --bi_dir

python3 -m training.vse --epochs 4 --batch_size 64  --embed_dim 4096 --margin 0.25 --resize_image 250 --bi_dir --loss HRL --continue ./checkpoints/em4096_biDir_bs64_ep32_m0.25_g0.85_rs250.pth
python3 -m training.vse --epochs 4 --batch_size 128 --embed_dim 4096 --margin 0.25 --resize_image 250 --bi_dir --loss HRL --continue ./checkpoints/em4096_biDir_bs64_ep32_m0.25_g0.85_rs250.pth

python3 -m training.vse --epochs 4 --batch_size 64  --embed_dim 4096 --margin 0.35 --resize_image 250 --bi_dir --loss HRL --continue ./checkpoints/em4096_biDir_bs64_ep32_m0.25_g0.85_rs250.pth
python3 -m training.vse --epochs 4 --batch_size 64  --embed_dim 4096 --margin 0.15 --resize_image 250 --bi_dir --loss HRL --continue ./checkpoints/em4096_biDir_bs64_ep32_m0.25_g0.85_rs250.pth

# WEITER: more LSTM layers, MS COCO + mix, stronger image model (check runtime)