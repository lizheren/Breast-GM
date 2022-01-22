import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from utils.hungarian import hungarian
from data.data_loader import GMDataset, get_dataloader
from utils.evaluation_metric import matching_accuracy
from utils.parallel import DataParallel
from utils.model_sl import load_model

from utils.config import cfg
from Motif_Position.utils_pgnn import device

def eval_model(model, dataloader, eval_epoch=None, verbose=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.cls

    lap_solver = hungarian

    accs = torch.zeros(len(classes), device=device)

    mass_accs_avg = 0

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.cls = cls
        acc_match_num = torch.zeros(1, device=device)
        acc_total_num = torch.zeros(1, device=device)

        acc_mass_match_num = 0
        iterator_num = 0

        for inputs in dataloader:

            iterator_num += 1

            # if 'images' in inputs:
            #     data1, data2 = [_.to(device) for _ in inputs['images']]
            #     inp_type = 'img'
            # elif 'features' in inputs:
            #     data1, data2 = [_.to(device) for _ in inputs['features']]
            #     inp_type = 'feat'
            # else:
            #     raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')

            data1, data2 = [_.to(device) for _ in inputs['features']]
            inp_type = 'feat'


            P1_gt, P2_gt = [_.to(device) for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.to(device) for _ in inputs['ns']]
            e1_gt, e2_gt = [_.to(device) for _ in inputs['es']]
            G1_gt, G2_gt = [_.to(device) for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.to(device) for _ in inputs['Hs']]
            KG, KH = [_.to(device) for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].to(device)

            mass_pair_index = inputs['mass_pair_index']


            batch_num = data1.size(0)

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                s_pred, pred = \
                    model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)

            print("s_pred", s_pred)
            print("s_pred shape", s_pred.size())

            s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)

            print("s_pred_perm", s_pred_perm)
            print("s_pred_perm shape", s_pred_perm.size())


            print("ground truth", perm_mat)
            print("mass_pair_index", mass_pair_index)
            print("mass_pair", mass_pair_index.size())

            mass_acc_list = []

            for mass_acc_num_index in range(batch_num):
                mass_acc_row_index = int(mass_pair_index[mass_acc_num_index][0])
                mass_acc_col_index = int(mass_pair_index[mass_acc_num_index][1])

                # print("mass_acc_row_index",mass_acc_row_index)
                mass_acc = float(s_pred[mass_acc_num_index][mass_acc_row_index][mass_acc_col_index])
                mass_acc_list.append(mass_acc)

            print("mass_acc_batch_list", mass_acc_list)
            mass_acc_avg = np.mean(np.array(mass_acc_list))
            print("mass_acc_batch_avg", mass_acc_avg)

            acc_mass_match_num += mass_acc_avg


            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt)
            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num


            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
                running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

        mass_accs_avg = acc_mass_match_num / iterator_num


    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)
    ds.cls = cls_cache

    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
        print('{} = {:.4f}'.format(cls, single_acc))

    print('mass_avg  = {:.4f}'.format(mass_accs_avg) )

    print('average = {:.4f}'.format(torch.mean(accs)))



    return accs


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PAIR.RESCALE)
    dataloader = get_dataloader(image_dataset)


    model = Net()
    model = model.to(device)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True)
