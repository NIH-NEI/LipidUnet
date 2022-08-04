import os, sys

import numpy as np
import imageio
import tifffile
import skimage.filters

import torch
from torch.utils.data import Dataset

class SegmentDataset(Dataset):
    ALLOWED_EXTENSIONS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    def __init__(self, base_dir:str,
            imgs_subdir:str='imgs',
            masks_subdir:str='masks',
            mask_suffix:str='_mask',
            checkpoints_subdir:str='.checkpoints',
            set_name=None,
            predict:bool=False):
        self.base_dir = os.path.abspath(base_dir)
        self.parent_dir = os.path.dirname(self.base_dir)
        self.imgs_dir = os.path.join(self.base_dir, imgs_subdir)
        self.masks_dir = os.path.join(self.base_dir, masks_subdir)
        self.mask_suffix = mask_suffix
        self.checkpoints_dir = os.path.join(self.parent_dir, checkpoints_subdir)
        self.predict = predict
        #
        self.set_name = str(set_name) if set_name else os.path.basename(self.base_dir)
        self.dataitems = []
        self.valitems = []
        #
        if not os.path.isdir(self.imgs_dir):
            return
        #
        img_map = {}
        for fn in os.listdir(self.imgs_dir):
            fpath = os.path.join(self.imgs_dir, fn)
            if os.path.isfile(fpath):
                bn, ext = os.path.splitext(fn)
                if ext.lower() in self.ALLOWED_EXTENSIONS:
                    img_map[bn] = fn
        #
        dataitems = self.dataitems
        if self.predict:
            for bn in sorted(img_map.keys()):
                self.dataitems.append((img_map[bn], None))
            dataitems = self.valitems
        #
        if not os.path.isdir(self.masks_dir):
            return
        #
        try:
            sflen = len(mask_suffix)
        except Exception:
            mask_suffix = ''
            sflen = 0
        for fn in os.listdir(self.masks_dir):
            fpath = os.path.join(self.masks_dir, fn)
            if os.path.isfile(fpath):
                bn, ex = os.path.splitext(fn)
                if sflen > 0 and not bn.endswith(mask_suffix):
                    continue
                bn = bn[:-sflen]
                if not bn in img_map: continue
                imgfn = img_map[bn]
                dataitems.append((imgfn, fn))
    #
    def __len__(self):
        return len(self.dataitems)
    #
    def lenval(self):
        return len(self.valitems)
    #
    @staticmethod
    def is_miniswhite(imgpath):
        try:
            with tifffile.TiffFile(imgpath) as tif:
                cmap = tif.pages[0].tags['ColorMap'].value
            return cmap[0][0] != 0
        except Exception:
            return False
    @staticmethod
    def load_image(imgpath):
        img = imageio.imread(imgpath)
        if img.dtype == np.uint16:
            img = img.astype(np.float32)
            #otsu = skimage.filters.threshold_otsu(img, 4096)
            #sc = 0.33 / otsu if otsu > 0.001 else 1.
            sc = 1./65535.
            img = img * sc
            #img[img > 1.] = 1.
        elif img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = img / 255.
        if img.ndim == 2:
            h = img.shape[0]
            w = img.shape[1]
            rgb = np.empty(shape=(3,h,w), dtype=img.dtype)
            rgb[0] = img
            rgb[1] = img
            rgb[2] = img
            img = rgb
        else:
            img = img.transpose((2, 0, 1))
        return img
    #
    @staticmethod
    def load_mask(maskpath):
        mask = imageio.imread(maskpath)
        if SegmentDataset.is_miniswhite(maskpath):
            # handle indexed TIFF with inverted color map (0=white)
            new_mask = np.zeros(shape=mask.shape, dtype=mask.dtype)
            new_mask[mask == 0] = 1
            return new_mask
        mask[mask != 0] = 1
        mask = mask.astype(np.uint8)
        return mask
    #
    def __getitem__(self, idx):
        imgfn, maskfn = self.dataitems[idx]
        imgpath = os.path.join(self.imgs_dir, imgfn)
        img = self.load_image(imgpath)
        torch_img = torch.as_tensor(img).float().contiguous()
        #
        if maskfn is None:
            torch_mask = None
        else:
            maskpath = os.path.join(self.masks_dir, maskfn)
            mask = self.load_mask(maskpath)
            torch_mask = torch.as_tensor(mask).long().contiguous()
        #
        return {
            'image': torch_img,
            'mask': torch_mask,
        }
    #
    def checkpoint_fn(self, epoch):
        return '%s_epoch%04d.pth' % (self.set_name, epoch)
    def checkpoint_path(self, epoch):
        return os.path.join(self.checkpoints_dir, self.checkpoint_fn(epoch))
    #
    def last_checkpoint(self):
        last_epoch = 0
        last_wpath = None
        try:
            prefix = '%s_epoch' % (self.set_name,)
            plen = len(prefix)
            for fn in os.listdir(self.checkpoints_dir):
                bn, ext = os.path.splitext(fn)
                if not bn.startswith(prefix): continue
                epoch = int(bn[plen:])
                if last_wpath is None or epoch > last_epoch:
                    last_epoch = epoch
                    last_wpath = os.path.join(self.checkpoints_dir, fn)
        except Exception:
            pass
        return last_epoch, last_wpath
    #
#
if __name__ == '__main__':
    tr_dir = os.path.abspath('training_data/actin')
    print('Reading training data directory:', tr_dir)
    ds = SegmentDataset(tr_dir)
    #print(ds.dataitems)
    print(len(ds.dataitems))
    a = ds[2]
    #print(a)

