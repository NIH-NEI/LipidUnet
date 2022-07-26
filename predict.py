import os, sys
import datetime

import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torchvision import transforms

from ldataset import SegmentDataset
from unet import UNet

class DummyCallback(object):
    def progress(self, pct):
        pass
    def status(self, txt):
        pass
    def stop_requested(self):
        return False

def predict_proc(predict_dir, weights_dir, prob, use_cuda, callback):
    start_ts = datetime.datetime.now()
    if callback is None:
        callback = DummyCallback()
    #
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f'Using device: {device}')
    #
    predict_dir = os.path.abspath(predict_dir)
    weights_dir = os.path.abspath(weights_dir)
    masks_dir = os.path.join(predict_dir, 'predicted_masks')
    dataset = SegmentDataset(predict_dir, checkpoints_subdir=weights_dir, predict=True)
    n_imgs = len(dataset)
    if n_imgs == 0:
        print('No source images found. Nothing to do.')
        return
    #
    epoch, mfpath = dataset.last_checkpoint()
    assert os.path.isfile(mfpath), 'No model weights found.'
    #
    callback.status(f'About to start ML predictions on {n_imgs} images from <<{dataset.set_name}>>...')
    callback.progress(0.)
    os.makedirs(masks_dir, exist_ok=True)
    #
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    net.to(device=device)
    net.load_state_dict(torch.load(mfpath, map_location=device))
    net.eval()
    print('Device:', device)
    print('Model weights:', mfpath)
    #
    for i in range(n_imgs):
        if callback.stop_requested():
            break
        fn, maskfn = dataset.dataitems[i]
        bn, ext = os.path.splitext(fn)
        imgpath = os.path.join(dataset.imgs_dir, fn)
        #
        ii = i+1
        print(f'Evaluating {ii} / {n_imgs} -- {imgpath}')
        callback.status(f'Evaluating <<{dataset.set_name}>> -- {ii} / {n_imgs} -- {fn}')
        #
        img = SegmentDataset.load_image(imgpath)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            mask = net(img)
            mask = torch.sigmoid(mask)[0]
            mask = mask.cpu().squeeze()
            bkg = mask[0].numpy()
            fgd = mask[1].numpy()
        #
        maskpath = os.path.join(masks_dir, bn+'_mask.tif')
        
        omask = np.zeros(shape=fgd.shape, dtype=np.uint8)
        omask[fgd > prob] = 0xFF
        print('Write:', maskpath)
        imageio.imwrite(maskpath, omask)

        pct = i * 100. / n_imgs
        callback.progress(pct)
    #
    elapsed = datetime.datetime.now() - start_ts
    if callback.stop_requested():
        print('Terminated by user.')
    else:
        print('Done.')
    print(f'Elapsed: {elapsed}')
    #


if __name__ == '__main__':
    pass