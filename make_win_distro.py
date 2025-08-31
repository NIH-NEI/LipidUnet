import os, sys
import zipfile
from version import APP_VERSION

EXCLUDE_DIRS = ('__pycache__', 'build', 'dist', )
EXCLUDE_EXTS = ('.bak', )
ZIP_PREFIX = 'LipidUnet/'
PREFIX = os.path.dirname(os.path.abspath(__file__))

def proc_dir(cdir):
    print('Scanning:', cdir)
    for fn in os.listdir(cdir):
        fpath = os.path.join(cdir, fn)
        if os.path.isdir(fpath):
            if fn in EXCLUDE_DIRS: continue
            for res in proc_dir(fpath):
                yield res
        else:
            yield fpath

if __name__ == '__main__':
    
    try:
        ver = '-'+APP_VERSION.split('(')[0].strip()
    except:
        ver = ''
    zipfn = f'LipidUnet{ver}-Win64.zip'
    zipdir = os.path.join(PREFIX, 'dist')
    os.makedirs(zipdir, exist_ok=True)
    zippath = os.path.join(zipdir, zipfn)
    print('Write:', zippath)
    
    with zipfile.ZipFile(zippath, mode='w', compression=zipfile.ZIP_LZMA) as zip:
        for fpath in proc_dir(PREFIX):
            fdir, fn = os.path.split(fpath)
            _, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext in EXCLUDE_EXTS: continue
            rpath = ZIP_PREFIX+os.path.relpath(fpath, PREFIX).replace('\\', '/')
            print(rpath)
            zip.write(fpath, rpath)

        pyprefix = os.path.dirname(sys.executable)
        pyzipprefix = ZIP_PREFIX + 'python312/'
        for fpath in proc_dir(pyprefix):
            rpath = pyzipprefix+os.path.relpath(fpath, pyprefix).replace('\\', '/')
            print(rpath)
            zip.write(fpath, rpath)
    
    
    sys.exit(0)
