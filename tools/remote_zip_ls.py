# List files inside a remote zip using HTTP Range requests (reads only central directory).
import sys, io, zipfile, urllib.request
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url=url; self.pos=0
        req=urllib.request.Request(url, headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(req) as r:
            cr=r.headers.get('Content-Range')
            self.size=int(cr.split('/')[-1]) if cr else int(r.headers['Content-Length']); self.final=r.geturl()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else (self.pos+off if whence==1 else self.size+off); return self.pos
    def read(self, n=-1):
        if n<0: n=self.size-self.pos
        if n==0 or self.pos>=self.size: return b''
        end=min(self.pos+n, self.size)-1
        req=urllib.request.Request(self.final, headers={'Range':f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req) as r: data=r.read()
        self.pos+=len(data); return data
    def readinto(self, b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)
url=sys.argv[1]; pat=sys.argv[2] if len(sys.argv)>2 else ''
f=HTTPRangeFile(url)
z=zipfile.ZipFile(io.BufferedReader(f, buffer_size=1<<16))
names=z.namelist()
print('zip size MB', round(f.size/1e6,1), 'entries', len(names))
sel=[n for n in names if pat in n]
print('matching', pat, len(sel))
for n in sel[:int(sys.argv[3]) if len(sys.argv)>3 else 15]: print('  ', n, z.getinfo(n).file_size)
if len(sys.argv)>4:
    out=sys.argv[4]
    for n in sel[:int(sys.argv[3])]:
        p=out+'/'+n.replace('/','__'); open(p,'wb').write(z.read(n)); print('extracted', p)
