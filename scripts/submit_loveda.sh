export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='lovenas.loveda.nas_resnet_lovedecoder'
ckpt_path='./logs/loveda/resnet50_lovedecoder/lovenas_loveda_15k.pth'
submit_dir='./submit_loveda'
python submit.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --submit_dir=${submit_dir}
