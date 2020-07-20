from collections import OrderedDict
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed=1027

def get_gpus(gpus):
    # return gpus
    return range(gpus)

def set_determined_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def test(model_cfg, dataset_cfg, checkpoint, batch_size=64, gpus=1, workers=4):
    #cnt = 0
    #confusion
    conf_matrix = torch.zeros(model_cfg.num_class, model_cfg.num_class)
    #confusion
    set_determined_seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=get_gpus(gpus)).cuda()
    model.eval()

    results = []
    labels = []
    prog_bar = ProgressBar(len(dataset))
    for data, label in data_loader:
        with torch.no_grad():
            # cnt += 1
            # print("\n"+str(cnt))
            # torch.cuda.empty_cache()
            output = model(data).data.cpu().numpy()
        results.append(output)
        labels.append(label)
        for i in range(len(data)):
            prog_bar.update()
    results = np.concatenate(results)
    labels = np.concatenate(labels)
    
    #confusion
    conf_matrix = confusion_matrix(torch.max(torch.from_numpy(results), 1)[1], labels, conf_matrix)
    np.save('/home/computer/WBH/GCN/INTERGCN/conf.npy', conf_matrix)
    #confusion

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


def train(
        work_dir,
        model_cfg,
        loss_cfg,
        dataset_cfg,
        optimizer_cfg,
        batch_size,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
):

    set_determined_seed(seed)
    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]
    data_loaders = [
        torch.utils.data.DataLoader(dataset=call_obj(**d),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=workers,
                                    drop_last=True) for d in dataset_cfg
    ]

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    model.apply(weights_init)
    model = MMDataParallel(model, device_ids=get_gpus(gpus)).cuda()
    loss = call_obj(**loss_cfg)

    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level)
    runner.register_training_hooks(**training_hooks)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)

    # run
    workflow = [tuple(w) for w in workflow]
    runner.run(data_loaders, workflow, total_epochs, loss=loss)


# process a batch of data
def batch_processor(model, datas, train_mode, loss):

    data, label = datas
    data = data.cuda()
    label = label.cuda()

    # forward
    output = model(data)
    losses = loss(output, label)

    # output
    log_vars = dict(loss=losses.item())
    if not train_mode:
        log_vars['top1'] = topk_accuracy(output, label)
        log_vars['top5'] = topk_accuracy(output, label, 5)

    outputs = dict(loss=losses, log_vars=log_vars, num_samples=len(data.data))
    return outputs


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
