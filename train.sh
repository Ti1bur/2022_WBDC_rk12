cd ./src/single_stream
python pretrain.py
python main.py
python swa.py
cd ../double_stream
python -m torch.distributed.launch --nproc_per_node=2 pretrain.py
python main.py
python swa.py
cd ..
cd ..