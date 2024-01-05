import torch
import ever as er
import glob, os
import shutil
er.registry.register_all()



'''
MobileNet
'''

# # 1. 转换searched encoder weights
# checkpoint_path=r"H:/home/wjj/work_space/NASDecoder/swin_log/nas_decoder/FINAL_nas_decoder_mobilenet_stack2/model-30000.pth"
# save_ckpt_path ='logs/loveda/mobilenet_lovedecoder/mobilenet_loveda_30k.pth'
# arch_dir = './searched_archs/mobilenet_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)
#
#
# # 2. 转换模型整个参数
#
# config_path='lovenas.loveda.train_mobilenet_lovedecoder'
# checkpoint_path='H:/home/wjj/work_space/NASDecoder/swin_log/parse_decoder/mobilenetv2/N3C2/FINAL_parallel1-1/model-15000.pth'
# save_ckpt_path ='./logs/loveda/mobilenet_lovedecoder/lovenas_loveda_15k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# # 验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)


'''
EFNET-B3
'''
# 1. 转换searched encoder weights
# checkpoint_path=r"H:/home/wjj/work_space/NASDecoder/swin_log/nas_decoder/FINAL_efnetb3_stack2/model-30000.pth"
# save_ckpt_path ='logs/loveda/efnetb3_lovedecoder/efnetb3_loveda_30k.pth'
# arch_dir = './searched_archs/efnetb3_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)
#
#
# # 2. 转换模型整个参数
#
# config_path='lovenas.loveda.train_efnet_lovedecoder'
# checkpoint_path='H:/home/wjj/work_space/NASDecoder/swin_log/parse_decoder/efnetb3/N3C2/FINAL_parallel1-1/model-15000.pth'
# save_ckpt_path ='./logs/loveda/efnetb3_lovedecoder/lovenas_loveda_15k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# # 验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)



'''
Swin-base
'''
# checkpoint_path=r"H:/home/wjj/work_space/NASDecoder/swin_log/nas_decoder/FINAL_nas_swin_base_stack2/model-30000.pth"
# save_ckpt_path ='logs/loveda/swinbase_lovedecoder/swinbase_loveda_30k.pth'
# arch_dir = './searched_archs/swinbase_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)


# 2. 转换模型整个参数

# config_path='lovenas.loveda.train_swin_lovedecoder'
# checkpoint_path='H:/home/wjj/work_space/NASDecoder/swin_log/parse_decoder/swin_base/N3C2/FINAL_parallel1-3/model-15000.pth'
# save_ckpt_path ='./logs/loveda/swinbase_lovedecoder/lovenas_loveda_15k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# #验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)




'''
FloodNet mobilenet
'''
# checkpoint_path=r"H:\home\wjj\work_space\NASDecoder\log\FLOODNet\NAS\mobilenet_stack2\model-60000.pth"
# save_ckpt_path ='logs/floodnet/mobilenet_lovedecoder/mobilenet_floodnet_60k.pth'
# arch_dir ='searched_archs/floodnet/mobilenet_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)


# 2. 转换模型整个参数

# config_path='lovenas.floodnet.train_mobilenet_lovedecoder'
# checkpoint_path=r'H:\home\wjj\work_space\NASDecoder\log\FLOODNet\New\mobilenetv2_nodel4_cell3\model-60000.pth'
# save_ckpt_path ='./logs/floodnet/mobilenet_lovedecoder/lovenas_floodnet_60k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# #验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)




'''
FloodNet resnet50
'''
# checkpoint_path=r"H:\home\wjj\work_space\NASDecoder\log\FLOODNet\NAS\resnet50_stack2\model-60000.pth"
# save_ckpt_path ='logs/floodnet/resnet50_lovedecoder/resnet50_floodnet_60k.pth'
# arch_dir ='searched_archs/floodnet/resnet50_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)


# 2. 转换模型整个参数

# config_path='lovenas.floodnet.train_resnet_lovedecoder'
# checkpoint_path=r'H:\home\wjj\work_space\NASDecoder\log\FLOODNet\New\FINAL_resnet50_nodel4_cell3_384_parallel1\model-60000.pth'
# save_ckpt_path ='./logs/floodnet/resnet50_lovedecoder/lovenas_floodnet_60k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# #验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)


'''
FloodNet efnetb3
'''
# checkpoint_path=r"H:\home\wjj\work_space\NASDecoder\log\FLOODNet\NAS\efnetb3_stack2\model-60000.pth"
# save_ckpt_path ='logs/floodnet/efnetb3_lovedecoder/efnetb3_floodnet_60k.pth'
# arch_dir ='searched_archs/floodnet/efnetb3_lovedecoder'
#
# npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
# os.makedirs(arch_dir, exist_ok=True)
# os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
# for npy_p in npy_plist:
#     if 'new' in npy_p:
#         idx = int(os.path.basename(npy_p).split('_')[0][-1])
#         if 'cell' in npy_p:
#             save_path = os.path.join(arch_dir, f'p{idx}.npy')
#         else:
#             save_path = os.path.join(arch_dir, f'c{idx}.npy')
#         shutil.copy(npy_p, save_path)
#
# ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
# deleted_params = dict(model=dict())
# for k, v in ckpt_params['model'].items():
#     if 'en' in k:
#         deleted_params['model'][k] = v
#
# torch.save(deleted_params, save_ckpt_path)
#
#
# # 2. 转换模型整个参数
# config_path='lovenas.floodnet.train_efnet_lovedecoder'
# checkpoint_path=r'H:\home\wjj\work_space\NASDecoder\log\FLOODNet\New\FINAL_efnetb3_nodel4_cell3\model-60000.pth'
# save_ckpt_path ='./logs/floodnet/efnetb3_lovedecoder/lovenas_floodnet_60k.pth'
# deled_state = dict(model=dict())
#
# model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
# deled_state['model'] = model.state_dict()
# deled_state['global_step'] = gs
# torch.save(deled_state, save_ckpt_path)
#
#
# #3. 验证
# model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
# print('验证成功！', gs)




'''
FloodNet swin-base
'''
checkpoint_path=r"H:\home\wjj\work_space\NASDecoder\log\FLOODNet\NAS\nas_swin_base_stack2\model-60000.pth"
save_ckpt_path ='logs/floodnet/swinbase_lovedecoder/swinbase_floodnet_60k.pth'
arch_dir ='searched_archs/floodnet/swinbase_lovedecoder'

npy_plist = glob.glob(os.path.join(os.path.dirname(checkpoint_path), '*new.npy'))
os.makedirs(arch_dir, exist_ok=True)
os.makedirs(os.path.dirname(save_ckpt_path), exist_ok=True)
for npy_p in npy_plist:
    if 'new' in npy_p:
        idx = int(os.path.basename(npy_p).split('_')[0][-1])
        if 'cell' in npy_p:
            save_path = os.path.join(arch_dir, f'p{idx}.npy')
        else:
            save_path = os.path.join(arch_dir, f'c{idx}.npy')
        shutil.copy(npy_p, save_path)

ckpt_params = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
deleted_params = dict(model=dict())
for k, v in ckpt_params['model'].items():
    if 'en' in k:
        deleted_params['model'][k] = v

torch.save(deleted_params, save_ckpt_path)


# 2. 转换模型整个参数
config_path='lovenas.floodnet.train_swin_lovedecoder'
checkpoint_path=r'H:\home\wjj\work_space\NASDecoder\log\FLOODNet\New\swin_nodel4_cell3\model-60000.pth'
save_ckpt_path ='./logs/floodnet/swinbase_lovedecoder/lovenas_floodnet_60k.pth'
deled_state = dict(model=dict())

model, gs = er.infer_tool.build_and_load_from_file(config_path, checkpoint_path)
deled_state['model'] = model.state_dict()
deled_state['global_step'] = gs
torch.save(deled_state, save_ckpt_path)


#3. 验证
model, gs = er.infer_tool.build_and_load_from_file(config_path, save_ckpt_path)
print('验证成功！', gs)