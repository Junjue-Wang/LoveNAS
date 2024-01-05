export CUDA_VISIBLE_DEVICES=6
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='lovenas.loveda.train_resnet_lovedecoder'
model_dir='./logs/Train/resnet50_lovedecoder'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port $RANDOM train_loveda.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    learning_rate.params.max_iters 15000 \
    train.num_iters 15000 \
    data.train.params.batch_size 16 \
    model.params.parse_decoder.connect_map './searched_archs/loveda/resnet50_lovedecoder/c3.npy' \
    model.params.parse_decoder.ops_map './searched_archs/loveda/resnet50_lovedecoder/p2.npy'
