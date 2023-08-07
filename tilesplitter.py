import os, sys

import numpy as np

class TileSplitter2D(object):
    def __init__(self, cdata, mask, mpsize=1, tilepad=8):
        self.cdata = cdata
        self.mask = mask
        self.mpsize = mpsize
        self.tilepad = tilepad
        #
        self.tilemap = []
        #
        self.tsize = self.mpsize*1048576
        self.ntx = 1
        self.nty = 1
        self.h, self.w = self.cdata.shape[0:2]
        self.th = self.h
        self.tw = self.w
        while self.th*self.tw > self.tsize:
            if self.th >= self.tw:
                self.nty += 1
                self.th = self.h // self.nty + self.tilepad*2
            else:
                self.ntx += 1
                self.tw = self.w // self.ntx + self.tilepad*2
        #
        for j in range(self.nty):
            y0 = j * self.h // self.nty
            y1 = (j+1) * self.h // self.nty
            ty = y1 - y0
            yy0 = y0 - self.tilepad
            if yy0 < 0:
                yy0 = 0
            yy1 = y1 + self.tilepad
            if yy1 > self.h:
                yy1 = self.h
            for i in range(self.ntx):
                x0 = i * self.w // self.ntx
                x1 = (i+1) * self.w // self.ntx
                tx = x1 - x0
                xx0 = x0 - self.tilepad
                if xx0 < 0:
                    xx0 = 0
                xx1 = x1 + self.tilepad
                if xx1 > self.w:
                    xx1 = self.w
                self.tilemap.append((x0,x1,tx,xx0,xx1, y0,y1,ty,yy0,yy1))
    #
    def numtiles(self):
        return len(self.tilemap)
    #
    def getDataTile(self, idx):
        x0,x1,tx,xx0,xx1, y0,y1,ty,yy0,yy1 = self.tilemap[idx]
        return self.cdata[yy0:yy1,xx0:xx1]
    #
    def getMaskTile(self, idx):
        x0,x1,tx,xx0,xx1, y0,y1,ty,yy0,yy1 = self.tilemap[idx]
        return self.mask[yy0:yy1,xx0:xx1]
    #
    def setMaskTile(self, idx, mtile):
        x0,x1,tx,xx0,xx1, y0,y1,ty,yy0,yy1 = self.tilemap[idx]
        mx0 = x0 - xx0
        my0 = y0 - yy0
        self.mask[y0:y1,x0:x1] = mtile[my0:my0+ty,mx0:mx0+tx]
    #
