export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='lovenas.loveda.nas_resnet_lovedecoder'
model_dir='./logs/NAS/resnet50_lovedecoder'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port $RANDOM train_loveda.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    learning_rate.params.max_iters 30000 \
    train.num_iters 30000 \
    data.train.params.batch_size 16