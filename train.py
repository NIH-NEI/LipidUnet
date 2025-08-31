import os, sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split

from ldataset import SegmentDataset
from evaluate import dice_loss, evaluate
from unet import UNet

class Avg(object):
    def __init__(self):
        self.reset()
    #
    def reset(self):
        self.cnt = 0
        self.acc = 0.
    #
    def __add__(self, v):
        self.acc += v
        self.cnt += 1
        return self
    #
    @property
    def avg(self):
        return 0 if self.cnt <= 0 else self.acc / self.cnt
    #

class DummyCallback(object):
    def progress(self, pct):
        pass
    def status(self, txt):
        pass
    def stop_requested(self):
        return False

def train_net(net, device, dataset,
        start_epoch:int=1,
        end_epoch:int=5,
        learning_rate:float=1e-5,
        val_split:float=0.1,
        save_checkpoint:bool=True,
        val_after:float=0.2,
        callback=None
        ):
    amp = False
    batch_size = 1
    
    if callback is None:
        callback = DummyCallback()
        
    callback.status(f'About to start training <<{dataset.set_name}>>...')
    callback.progress(0.)
    
    n_epochs = end_epoch + 1 - start_epoch
    
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    global_n_train = n_train * n_epochs
    
    print(f'Training steps = {n_train}, validation steps = {n_val}')

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    #grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_scaler = torch.amp.GradScaler(str(device), enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    
    validate_after = int(n_train*val_after) + 1
    next_validate_step = validate_after
    
    aloss = Avg()
    
    last_val_score = ''

    # 5. Begin training
    for epoch in range(start_epoch, end_epoch+1):
        start_ts = datetime.datetime.now()
        net.train()
        epoch_loss = 0
        print(f'--- Epoch {epoch} / {end_epoch} ---')

        for step, batch in enumerate(train_loader):
            if callback.stop_requested():
                return 0
            images = batch['image']
            true_masks = batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.amp.autocast(str(device), enabled=amp):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks) \
                       + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            _step = step+1

            if callback.stop_requested():
                return 0
            _loss = loss.item()
            epoch_loss += _loss
            aloss += _loss
            if global_step % 10 == 0:
                pct = global_step * 100. / global_n_train
                callback.progress(pct)
                pct = '%1.2f' % (pct,)
                print(f'Epoch {epoch}: Step {_step} / {n_train}: {pct}% total : Avg.loss(last {aloss.cnt} steps) = {aloss.avg}')
                callback.status(f'Training <<{dataset.set_name}>> -- Epoch {epoch} / {end_epoch} -- Step {_step} / {n_train} -- loss = {aloss.avg:.4f}. {last_val_score}')
                aloss.reset()
            
            if global_step >= next_validate_step:
                next_validate_step += validate_after
                #
                print(f'Validation time, {n_val} steps...')
                v_start = datetime.datetime.now()
                val_score = evaluate(net, val_loader, device, callback)
                if callback.stop_requested():
                    return 0
                scheduler.step(val_score)
                v_elapsed = datetime.datetime.now() - v_start
                print(f'Validation results: epoch {epoch}, dice score = {val_score}; elapsed: {v_elapsed}.')
                last_val_score = f'Last Validation Score = {val_score:.4f}'

        if save_checkpoint:
            os.makedirs(dataset.checkpoints_dir, exist_ok=True)
            fpath = dataset.checkpoint_path(epoch)
            torch.save(net.state_dict(), fpath)
            print('Checkpoint saved to:', fpath)
        elapsed = datetime.datetime.now() - start_ts
        if n_train > 0:
            avg_loss = epoch_loss / n_train
            print(f'Epoch {epoch} is done, average loss: {avg_loss}, elapsed time: {elapsed}')
    #
    return epoch

def train_proc(train_dir, weights_dir, n_epochs, use_cuda, callback):
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f'Using device: {device}')

    # 1. Create dataset
    train_dir = os.path.abspath(train_dir)
    if weights_dir:
        weights_dir = os.path.abspath(weights_dir)
    else:
        weights_dir = 'model_weights'
    dataset = SegmentDataset(train_dir, checkpoints_subdir=weights_dir)
    # Find last checkpoint (if any)
    epoch, mfpath = dataset.last_checkpoint()
    print(f'Using training data from {train_dir}')
    dsize = len(dataset)
    assert dsize > 10, f'Training data set too small: {dsize}; must be at least 10.'
    #
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)
    if not mfpath is None:
        print('Loading model weights from', mfpath)
        net.load_state_dict(torch.load(mfpath, map_location=device, weights_only=True))
    #
    train_net(net, device, dataset,
        start_epoch=epoch+1,
        end_epoch=epoch+n_epochs,
        learning_rate=0.00001,
        val_split=0.1,
        save_checkpoint=True,
        val_after=0.2,
        callback=callback)
    #
    if callback and callback.stop_requested():
        print('Terminated by user.')
    #
    
if __name__ == '__main__':
    
    train_dir = 'training_data/actin'
    weights_dir = 'model_weights'
    n_epochs = 5
    use_cuda = True
    callback = None
    
    rc = 0
    start_ts = datetime.datetime.now()
    try:
        train_proc(train_dir, weights_dir, n_epochs, use_cuda, callback)
    except Exception as ex:
        print('Exception:', str(ex))
        rc = 3
    elapsed = datetime.datetime.now() - start_ts
    print(f'Elapsed time: {elapsed}.')
    
    print(f'Exiting({rc}).')
    sys.exit(rc)

