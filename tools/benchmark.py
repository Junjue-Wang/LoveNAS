from ever.core.builder import make_model
import argparse
import torch
import time
from ever.util import param_util
import prettytable as pt
from ever.core.config import import_config
import ever as er
er.registry.register_all()


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')

run_num = 500

def run(args):
    cfg = import_config(args.config_path)

    model = make_model(cfg['model']).cuda()
    cnt = param_util.count_model_parameters(model)
    model.eval()

    inputs = torch.ones(1, 3, 512, 512).cuda()
    start = time.time()
    with torch.no_grad():
        for i in range(run_num):
            model(inputs)
            torch.cuda.synchronize()

    total_time = (time.time() - start) / run_num
    FPS = run_num / (time.time() - start)
    tb = pt.PrettyTable()
    tb.field_names = ['#params', 'speed', 'FPS']
    tb.add_row(['{} M'.format(round(cnt / float(1e6), 3)), '{} s'.format(total_time), '{} PFS'.format(FPS)])
    print(tb)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
