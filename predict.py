import os, sys
import datetime

import numpy as np
import imageio
import torch
import torch.nn.functional as F
from torchvision import transforms

from ldataset import SegmentDataset
from unet import UNet

from tilesplitter import TileSplitter2D

class DummyCallback(object):
    def progress(self, pct):
        pass
    def status(self, txt):
        pass
    def threshold(self, v):
        pass
    def stop_requested(self):
        return False

class SensitivityEvaluator():
    def __init__(self, wtneg=2.):
        self.wtneg = wtneg
        #
        self.gt_count = 0
        self.fpos_counts = [0]*100
        self.fneg_counts = [0]*100
    #
    def accumulate(self, fgd, vmask):
        self.gt_count += np.count_nonzero(vmask)
        imask = np.zeros(shape=fgd.shape, dtype=np.uint8)
        for ii in range(100):
            i = 99 - ii
            prob = 0.01*i
            self.fpos_counts[i] += np.count_nonzero((fgd > prob) & (vmask == 0))
            self.fneg_counts[i] += np.count_nonzero((fgd <= prob) & (vmask != 0))
    #
    def best(self):
        pct = -1
        bsf = None
        norm = 1. / (self.gt_count + 1)
        for i in range(1,100):
            loss = (self.fpos_counts[i] + self.wtneg * self.fneg_counts[i]) * norm
            if bsf is None or loss < bsf:
                bsf = loss
                pct = i
        return pct
    #
    
def infer(img, net, device):
    img = SegmentDataset.load_image(img)
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
    return fgd, bkg

def predict_proc(predict_dir, weights_dir, prob, autosense, use_cuda, callback):
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
    if autosense:
        eval = SensitivityEvaluator(2.)
        n_val = dataset.lenval()
        for i in range(n_val):
            fn, maskfn = dataset.valitems[i]
            imgpath = os.path.join(dataset.imgs_dir, fn)
            maskpath = os.path.join(dataset.masks_dir, maskfn)
            cdata = imageio.imread(imgpath)
            cmask = SegmentDataset.load_mask(maskpath)
            spl = TileSplitter2D(cdata, cmask, mpsize=1)
            #
            ii = i+1
            print(f'Adjusting Probability Threshold {ii} / {n_val} -- {imgpath} : {maskpath}')
            callback.status(f'Adjusting PT <<{dataset.set_name}>> -- {ii} / {n_val} -- {fn}')
            #
            for idx in range(spl.numtiles()):
                img = spl.getDataTile(idx)
                vmask = spl.getMaskTile(idx)
                fgd, bkg = infer(img, net, device)
                eval.accumulate(fgd, vmask)
            #
            pct = i * 100. / n_val
            callback.progress(pct)
        #
        callback.progress(0.)
        #print('false positives:', eval.fpos_counts)
        #print('false negatives:', eval.fneg_counts)
        iprob = eval.best()
        callback.threshold(iprob)
        prob = 0.01 * iprob
        print('Probability Threshold adjusted to: %0.3f' % (prob,))
    #
    for i in range(n_imgs):
        if callback.stop_requested():
            break
        fn, maskfn = dataset.dataitems[i]
        bn, ext = os.path.splitext(fn)
        imgpath = os.path.join(dataset.imgs_dir, fn)
        cdata = imageio.imread(imgpath)
        cmask = np.empty(shape=(cdata.shape[0], cdata.shape[1]), dtype=np.uint8)
        spl = TileSplitter2D(cdata, cmask, mpsize=1)
        #
        ii = i+1
        print(f'Evaluating {ii} / {n_imgs} -- {imgpath}')
        callback.status(f'Evaluating <<{dataset.set_name}>> -- {ii} / {n_imgs} -- {fn}')
        #
        for idx in range(spl.numtiles()):
            print('Infer:', idx)
            img = spl.getDataTile(idx)
            fgd, bkg = infer(img, net, device)
            tmask = np.zeros(shape=fgd.shape, dtype=np.uint8)
            tmask[fgd > prob] = 0xFF
            spl.setMaskTile(idx, tmask)
        #
        maskpath = os.path.join(masks_dir, bn+'_mask.tif')
        print('Write:', maskpath)
        imageio.imwrite(maskpath, cmask)
        #
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