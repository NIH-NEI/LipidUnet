import os, sys
import threading
import json
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk

from ldataset import SegmentDataset
from train import train_proc
from predict import predict_proc

APP_NAME = 'Lipid U-Net'
APP_VERSION = '0.0.2 (2022-08-04)'

class LipidUnetMainWinnow(object):
    def __init__(self, homedir):
        self.homedir = homedir
        #
        self.statefile = os.path.join(self.homedir, 'state.json')
        #
        self.window = window  = tk.Tk()
        window.title(APP_NAME+' ver. '+APP_VERSION)
        #
        self.lock = threading.Lock()
        self._busy = False
        self._stop_requested = False
        self._train_dataset = None
        self._predict_dataset = None
        #
        self._busy = False
        self._pt_map = {}
        #
        # screen_width = window.winfo_screenwidth()
        # window_width = screen_width * 50 // 100
        # screen_height = window.winfo_screenheight()
        # window_height = screen_height * 40 // 100
        # center_x = int(screen_width/2 - window_width / 2)
        # center_y = int(screen_height/2 - window_height / 2)
        # window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        #
        self.trainDirVar = tk.StringVar()
        self.modelWeightsDirVar = tk.StringVar()
        self.numEpochsVar = tk.IntVar()
        self.numEpochsVar.set(10)
        self.sensitivityVar = tk.IntVar()
        self.sensitivityVar.set(65)
        self.autoVar = tk.IntVar()
        self.autoVar.set(0)
        self.predictDirVar = tk.StringVar()
        #
        self.frame = frame = tk.Frame(window, relief=tk.RAISED, borderwidth=1)
        frame.pack(fill=tk.BOTH, expand=True)
        #
        self.lftrain = lftrain = tk.LabelFrame(frame, text="Train")
        lftrain.grid(column=0, row=0, pady=5)
        trainDirLbl = tk.Label(lftrain, text="Training Data Directory:", padx=10)
        trainDirLbl.grid(column=0, row=0, sticky=tk.E)
        self.trainDirTxt = trainDirTxt = tk.Entry(lftrain, width=80, textvariable=self.trainDirVar)
        trainDirTxt.grid(column=1, row=0)
        def trainDirBtnClicked():
            trainDir = filedialog.askdirectory(initialdir = self.trainDir,
                    title="Select directory containing training data",
                    mustexist=True)
            if not trainDir: return
            self.trainDir = trainDir
        trainDirBtn = tk.Button(lftrain, text="...", padx=5, command=trainDirBtnClicked)
        trainDirBtn.grid(column=2, row=0, padx=5, pady=5)
        #
        self.lbTrainSet = lbTrainSet = tk.Label(lftrain, text='No training data', padx=10, fg='#005')
        lbTrainSet.grid(column=0, row=1, columnspan=3, sticky=tk.W)
        frpar = tk.Frame(lftrain)
        frpar.grid(column=0, row=2, columnspan=2, pady=5, sticky=tk.E)
        lbNumEpochs = tk.Label(frpar, text='Number of epochs:', padx=10)
        lbNumEpochs.grid(column=0, row=0)
        self.numEpochsTxt = numEpochsTxt = tk.Entry(frpar, width=16, textvariable=self.numEpochsVar)
        numEpochsTxt.grid(column=1, row=0)
        trainBtn = tk.Button(frpar, text="Train", padx=15, command=self.train)
        trainBtn.grid(column=2, row=0)
        #
        self.lfweights = lfweights = tk.LabelFrame(frame, text="Model Weights")
        lfweights.grid(column=0, row=1, pady=5)
        weightsDirLbl = tk.Label(lfweights, text="Model Weights Directory:", padx=10)
        weightsDirLbl.grid(column=0, row=0, sticky=tk.E)
        self.weightsDirTxt = weightsDirTxt = tk.Entry(lfweights, width=80, textvariable=self.modelWeightsDirVar)
        weightsDirTxt.grid(column=1, row=0)
        def weightsDirBtnClicked():
            weightsDir = filedialog.askdirectory(initialdir = self.weightsDir,
                    title="Select directory containing model weights",
                    mustexist=True)
            if not weightsDir: return
            self.weightsDir = weightsDir
        weightsDirBtn = tk.Button(lfweights, text="...", padx=5, command=weightsDirBtnClicked)
        weightsDirBtn.grid(column=2, row=0, padx=5, pady=5)
        #
        self.lfpredict = lfpredict = tk.LabelFrame(frame, text="Predict")
        lfpredict.grid(column=0, row=2, pady=5)
        predictDirLbl = tk.Label(lfpredict, text="Prediction Data Directory:", padx=10)
        predictDirLbl.grid(column=0, row=0, sticky=tk.E)
        self.predictDirTxt = predictDirTxt = tk.Entry(lfpredict, width=80, textvariable=self.predictDirVar)
        predictDirTxt.grid(column=1, row=0)
        def predictDirBtnClicked():
            predictDir = filedialog.askdirectory(initialdir = self.predictDir,
                    title="Select directory containing source data",
                    mustexist=True)
            if not predictDir: return
            self.predictDir = predictDir
        predictDirBtn = tk.Button(lfpredict, text="...", padx=5, command=predictDirBtnClicked)
        predictDirBtn.grid(column=2, row=0, padx=5, pady=5)
        self.lbPredictSet = lbPredictSet = tk.Label(lfpredict, text='No source images', padx=10, fg='#005')
        lbPredictSet.grid(column=0, row=1, columnspan=3, sticky=tk.W)
        #
        mwLbl = tk.Label(lfpredict, text="Model Weights:", padx=10)
        mwLbl.grid(column=0, row=2, sticky=tk.E)
        self.mwFileLbl = mwFileLbl = tk.Label(lfpredict, text="N/A", padx=10, width=62, fg='#005', anchor=tk.W)
        mwFileLbl.grid(column=1, row=2, columnspan=2, sticky=tk.W)
        #
        frpred = tk.Frame(lfpredict)
        frpred.grid(column=0, row=3, columnspan=2, pady=5, sticky=tk.E)
        lbSens = tk.Label(frpred, text="Probability Threshold (PT):", padx=10)
        lbSens.grid(column=0, row=0, sticky=tk.E)
        self.sensitivityTxt = tk.Entry(frpred, width=16, textvariable=self.sensitivityVar)
        self.sensitivityTxt.grid(column=2, row=0)
        self.autoCheck = tk.Checkbutton(frpred, text='Auto Adjust PT',variable=self.autoVar, onvalue=1, offvalue=0)
        self.autoCheck.grid(column=3, row=0, padx=10)
        self.autoCheck['state'] = 'disabled'
        predictBtn = tk.Button(frpred, text="Predict", padx=15, command=self.predict)
        predictBtn.grid(column=4, row=0)
        #
        self.lfprogr = lfprogr = tk.LabelFrame(frame, text="Task Progress")
        lfprogr.grid(column=0, row=4, sticky=tk.W+tk.E, pady=5)
        self.lbCurTask = lbCurTask = tk.Label(lfprogr, text='Idle', padx=10, fg='#005')
        lbCurTask.grid(column=0, row=0, columnspan=2, sticky=tk.W)
        self.pbTrain = pbTrain = ttk.Progressbar(lfprogr, orient=tk.HORIZONTAL, length=500, mode='determinate')
        pbTrain.grid(column=0, row=1, padx=5, sticky=tk.W)
        self.lbPct = lbPct = tk.Label(lfprogr, text='', padx=10, width=8, fg='#005', anchor=tk.W)
        lbPct.grid(column=1, row=1, sticky=tk.W)
        def onStopBtnClicked():
            with self.lock:
                if not self._busy: return
                self._stop_requested = True
            self.status('Stopping current task...')
        stopBtn = tk.Button(lfprogr, text="Stop", padx=15, command=onStopBtnClicked)
        stopBtn.grid(column=2, row=1, pady=5, sticky=tk.E)
        #
        self.trainDirVar.trace('w', self.onTrainDirVar)
        self.modelWeightsDirVar.trace('w', self.onModelWeightsDirVar)
        self.numEpochsVar.trace('w', self.onNumEpochsVar)
        self.predictDirVar.trace('w', self.onPredictDirVar)
        self.sensitivityVar.trace('w', self.onSensitivityVar)
        self.autoVar.trace('w', self.onAutoVar)
        #
        self.loadState()
    #
    def run(self):
        self.window.mainloop()
    #
    @property
    def trainDir(self):
        return self.trainDirVar.get()
    @trainDir.setter
    def trainDir(self, v):
        self.trainDirVar.set(v)
    #
    def onTrainDirVar(self, a, b, c):
        with self.lock:
            self._train_dataset = None
        train_dir = self.trainDir
        self.updateTrainDatasetInfo('No training data')
        if train_dir:
            imgs_dir = os.path.join(train_dir, 'imgs')
            masks_dir = os.path.join(train_dir, 'masks')
            if os.path.isdir(imgs_dir) and os.path.isdir(masks_dir):
                self.saveState()
                with self.lock:
                    self._train_ds_par = train_dir
                t = threading.Thread(target=self._get_training_dataset)
                t.setDaemon(True)
                t.start()
                return
            else:
                trainDir = None
    #
    @property
    def weightsDir(self):
        return self.modelWeightsDirVar.get()
    @weightsDir.setter
    def weightsDir(self, v):
        self.modelWeightsDirVar.set(v)
    #
    def onModelWeightsDirVar(self, a, b, c):
        with self.lock:
            weights_dir = self.weightsDir
        if weights_dir and os.path.isdir(weights_dir):
            self.saveState()
        self.get_predict_dataset()
    #
    @property
    def numEpochs(self):
        try:
            return self.numEpochsVar.get()
        except Exception:
            return 10
    @numEpochs.setter
    def numEpochs(self, v):
        try:
            v = int(v)
            self.numEpochsVar.set(v)
        except Exception:
            self.numEpochsVar.set(10)
    #
    def onNumEpochsVar(self, a, b, c):
        try:
            v = self.numEpochsVar.get()
            if v > 0:
                self.saveState()
        except Exception:
            pass
    #
    @property
    def pt_map(self):
        return self._pt_map
    @pt_map.setter
    def pt_map(self, v):
        self._pt_map.clear()
        try:
            self._pt_map.update(v)
        except Exception:
            pass
    #
    def _update_pt_map(self):
        if self._busy: return
        dcls = self.predictClass
        if not dcls is None:
            self._pt_map[dcls] = [self.sensitivity, self.autoSens]
        self.saveState()
    #
    def _recall_pt_map(self):
        dcls = self.predictClass
        if dcls is None or not dcls in self._pt_map:
            sens, autoSens = 65, False
        else:
            sens, autoSens = self._pt_map[dcls]
        self.sensitivity = sens
        self.autoSens = autoSens
    #
    @property
    def sensitivity(self):
        try:
            return self.sensitivityVar.get()
        except Exception:
            return 65
    #
    @sensitivity.setter
    def sensitivity(self, v):
        try:
            v = int(v)
            assert v > 0 and v <= 100
            self.sensitivityVar.set(v)
        except Exception:
            self.sensitivityVar.set(65)
    #
    def onSensitivityVar(self, a, b, c):
        try:
            v = self.sensitivityVar.get()
            if v > 0 and v <= 100:
                self._update_pt_map()
        except Exception:
            pass
    #
    @property
    def autoSens(self):
        try:
            return self.autoVar.get()
        except Exception:
            return 0
    @autoSens.setter
    def autoSens(self, v):
        try:
            v = int(v)
            assert v in (0, 1)
            self.autoVar.set(v)
        except Exception:
            self.autoVar.set(0)
    #
    def onAutoVar(self, a, b, c):
        try:
            v = self.autoVar.get()
            if v in (0, 1):
                self._update_pt_map()
        except Exception:
            pass
    #
    @property
    def predictDir(self):
        return self.predictDirVar.get()
    @predictDir.setter
    def predictDir(self, v):
        self.predictDirVar.set(v)
    #
    @property
    def predictClass(self):
        try:
            with self.lock:
                dcls = self._predict_dataset.set_name
            return dcls
        except Exception:
            return None
    #
    def onPredictDirVar(self, a, b, c):
        with self.lock:
            predict_dir = self.predictDir
        if predict_dir and os.path.isdir(predict_dir):
            self._update_pt_map()
        self.get_predict_dataset()
    #
    def _get_training_dataset(self):
        with self.lock:
            train_dir = self._train_ds_par
        ds = SegmentDataset(train_dir)
        if len(ds) >= 10:
            self.window.after(5, self.updateTrainDatasetInfo,
                    'Class <<%s>> -- Item count: %d (training+validation)' % (ds.set_name, len(ds)))
            with self.lock:
                self._train_dataset = ds
    #
    def _get_predict_dataset(self):
        with self.lock:
            predict_dir, weights_dir = self._predict_ds_par
        ds = SegmentDataset(predict_dir, checkpoints_subdir=weights_dir, predict=True)
        n_imgs = len(ds)
        n_val = ds.lenval()
        if n_imgs > 0:
            if n_val == 0:
                dsinfo = f'Class <<{ds.set_name}>> -- Source images: {n_imgs}; No validation images.'
            else:
                dsinfo = f'Class <<{ds.set_name}>> -- Source images: {n_imgs}; Validation images: {n_val}'
            self.window.after(1, self.updatePredictDatasetInfo, dsinfo)
            with self.lock:
                self._predict_dataset = ds
        epoch, wpath = ds.last_checkpoint()
        if not wpath is None:
            fn = os.path.basename(wpath)
            self.window.after(5, self.updateModelWeightsFileInfo, fn)
        self.window.after(6, self._recall_pt_map)
    #
    def get_predict_dataset(self):
        with self.lock:
            self._predict_dataset = None
        self._recall_pt_map()
        predict_dir = self.predictDir
        weights_dir = self.weightsDir
        self.updatePredictDatasetInfo('No source images')
        self.updateModelWeightsFileInfo('N/A')
        if predict_dir and os.path.isdir(predict_dir) and weights_dir and os.path.isdir(weights_dir):
            with self.lock:
                self._predict_ds_par = (predict_dir, weights_dir)
            t = threading.Thread(target=self._get_predict_dataset)
            t.setDaemon(True)
            t.start()
    #
    def updatePredictDatasetInfo(self, txt):
        self.lbPredictSet['text'] = txt
        st = 'disabled'
        with self.lock:
            if self._predict_dataset and self._predict_dataset.lenval() > 0:
                st = 'normal'
        self.autoCheck['state'] = st
    #
    def updateModelWeightsFileInfo(self, txt):
        self.mwFileLbl['text'] = txt
    #
    def updateTrainDatasetInfo(self, txt):
        self.lbTrainSet['text'] = txt
    #
    def updateStatus(self, txt):
        self.lbCurTask['text'] = txt
    #
    def updateProgress(self, pct):
        if pct is None:
            self.pbTrain['value'] = 0
            self.lbPct['text'] = ''
        else:
            self.pbTrain['value'] = pct
            self.lbPct['text'] = '%1.1f%%' % (pct,)
    #
    def updateProbThreshold(self, v):
        self.sensitivityVar.set(v)
    #
    def status(self, txt):
        self.window.after(1, self.updateStatus, txt)
    def progress(self, pct):
        self.window.after(1, self.updateProgress, pct)
    def threshold(self, v):
        self.window.after(1, self.updateProbThreshold, v)
    def stop_requested(self):
        return self._stop_requested
    #
    def error_in_progress(self):
        self.window.after(1, messagebox.showerror, 'Error',
            'An operation is already in progress.\n'+
            'Please wait until it is complete or press "Stop" to cancel it first.')
    #
    def _train(self):
        try:
            train_proc(*self._train_param)
            self.status('Terminated by user.' if self._stop_requested else 'Idle')
        except Exception as ex:
            self.status(str(ex))
        self.progress(None)
        self.window.after(1, self.get_predict_dataset)
        with self.lock:
            self._busy = False
    def train(self):
        with self.lock:
            if self._busy:
                self.error_in_progress()
                return
            if self._train_dataset is None:
                self.window.after(1, messagebox.showerror, 'Error',
                    'Please select a directory containing training data '+
                    'in subdirectories "imgs/" (sources) and "masks/" (GT).\n'+
                    'Names of the images must match, except mask images\n'+
                    'must have suffix "_mask", for instance:\n'+
                    'imgs/image001.tif -> masks/image001_mask.tif;\n'+
                    'there must be at least 10 such pairs.')
                return
            if not self.weightsDir or not os.path.isdir(self.weightsDir):
                self.window.after(1, messagebox.showerror, 'Error',
                    'Please select an existing directory where\n'+
                    'trained model weights will be stored.')
                return
            self._busy = True
            self._stop_requested = False
            self._train_param = (self.trainDir, self.weightsDir, self.numEpochs, True, self)
        t = threading.Thread(target=self._train)
        t.setDaemon(True)
        t.start()
    #
    def _predict(self):
        try:
            predict_proc(*self._predict_param)
            self.status('Terminated by user.' if self._stop_requested else 'Idle')
        except Exception as ex:
            self.status(str(ex))
        self.progress(None)
        with self.lock:
            self._busy = False
    def predict(self):
        with self.lock:
            if self._busy:
                self.error_in_progress()
                return
            if self._predict_dataset is None:
                self.window.after(1, messagebox.showerror, 'Error',
                    'Please select a directory containing images to be\n'+
                    'evaluated in subdirectory "imgs/".\n')
                return
            epoch, wpath = self._predict_dataset.last_checkpoint()
            if wpath is None:
                _name = self._predict_dataset.set_name
                self.window.after(1, messagebox.showerror, 'Error',
                    'Please select a directory containing trained\n'+
                    f'model weights "{_name}_epochNNN.pth".\n'+
                    "If you don't have such a file, do the training\n"+
                    f'for <<{_name}>> first.')
                return
            self._busy = True
            self._stop_requested = False
            autosense = self._predict_dataset.lenval() > 0 and self.autoSens
            sense = self.sensitivity * 0.01
            self._predict_param = (self.predictDir, self.weightsDir, sense, autosense, True, self)
        t = threading.Thread(target=self._predict)
        t.setDaemon(True)
        t.start()
    #
    PERSISTENT_PROPERTIES = ('trainDir', 'numEpochs', 'weightsDir', 'predictDir', 'pt_map',)
    #
    def loadState(self):
        self._busy = True
        try:
            with open(self.statefile, 'r') as fi:
                params = json.load(fi)
            for prop in self.PERSISTENT_PROPERTIES:
                if not prop in params: continue
                setattr(self, prop, params[prop])
        except Exception:
            pass
        self._busy = False
    #
    def saveState(self):
        try:
            params = dict((prop, getattr(self, prop)) for prop in self.PERSISTENT_PROPERTIES)
            with open(self.statefile, 'w') as fo:
                json.dump(params, fo, indent=2)
        except Exception:
            pass


if __name__ == '__main__':
    
    myhome = os.path.join(os.path.expanduser('~'), '.lipidunet')
    os.makedirs(myhome, exist_ok=True)
    
    proc = LipidUnetMainWinnow(myhome)
    
    # proc.window.after(1000, proc.updateStatus, 'Working on it...')
    # proc.window.after(2000, proc.updateProgress, 33.33)
    # proc.window.after(3000, proc.updateProgress, None)
    
    proc.run()
    
    sys.exit(0)

