import numpy as np
import pickle
import os
import random

SPLIT_SIZE=16

MAX_CACHE=30

def dist(a,b):
    return np.inner(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

class Base:
    def __init__(self):
        pass

class Node(Base):
    def __init__(self,level):
        self.data={}
        self.level=level
        self.lo=None
        self.hi=None

    def __len__(self):
        ret=len(self.data)
        if self.lo is not None:
            ret+=len(self.lo)
        if self.hi is not None:
            ret+=len(self.hi)
        return ret

    def put(self,k,m,v,overwrite=False):
        if self.lo is None:
            self.data=self.data
            h="".join(map(str,(m>0).astype("uint8")))
            if (not h in self.data) or overwrite:
                self.data[h]=(k,m,v)
                self.data=self.data
                if len(self.data)>SPLIT_SIZE:
                    self.split()
            else:
                v=self.data[h][-1]
            return v
        else:
            #FIXME make sure that there is always a hiid or add try except block
            if m[self.level]>0:
                return self.hi.put(k,m,v,overwrite)
            else:
                return self.lo.put(k,m,v,overwrite)

    def delete(self,k,m):
        if self.lo is None:
            self.data=self.data
            h="".join(map(str,(m>0).astype("uint8")))
            if h in self.data:
                ret=self.data[h]
                del self.data[h]
                return ret
        else:
            #FIXME make sure that there is always a hiid or add try except block
            if m[self.level]>0:
                return self.hi.delete(k,m)
            else:
                return self.lo.delete(k,m)

    def get(self,k,m):
        if self.lo is None:
            self.data=self.data
            h="".join(map(str,(m>0).astype("uint8")))
            if h in self.data:
                ret=self.data[h]
                return ret
        else:
            #FIXME make sure that there is always a hiid or add try except block
            if m[self.level]>0:
                return self.hi.get(k,m)
            else:
                return self.lo.get(k,m)

    def get_first_item(self):
        # get first item of self.data or self.lo.data if available
        if self.lo is None:
            if len(self.data.keys()) > 0:
                return (next(iter(self.data.items())))
            else:
                return {}
        else:
            return self.get_first_item(self.lo)

    def split(self):
        if self.lo is None:
            self.lo=Node(self.level+1)
        if self.hi is None:
            self.hi=Node(self.level+1)
        for h,(k,m,v) in self.data.items():
            if m[self.level]>0:
                self.hi.put(k,m,v)
            else:
                self.lo.put(k,m,v)
        self.data={}

    def _search(self,q,ret):
        for h,(k,m,v) in self.data.items():
            ret.append((dist(k,q),h,k,m,v))

    def _msearch(self,qs,rets):
        for h,(k,m,v) in self.data.items():
            for i,q in enumerate(qs):
                rets[i].append((dist(k,q),h,k,m,v))

    def recSearch(self,q,knn,v,v2,numbkt,ret):
        for h,(k,m,v) in self.data.items():
            ret.append((dist(k,q),h,k,m,v))
        if self.lo is not None:
            if v[self.level]>0 and numbkt>0:
                numbkt=self.hi.recSearch(q,knn,v,v2,numbkt,ret)
            if v[self.level]<=0 and numbkt>0:
                numbkt=self.lo.recSearch(q,knn,v,v2,numbkt,ret)
            return numbkt
        else:
            return numbkt-1

    def fullSearch(self,key,knn,ret):
        for h,(k,m,v) in self.data.items():
            ret.append((dist(k,key),h,k,m,v))
        if self.lo is not None:
            self.lo.fullSearch(key,knn,ret)
        if self.hi is not None:
            self.hi.fullSearch(key,knn,ret)
        if len(ret)>3*knn:
            ret.sort(key=lambda x: x[0],reverse=True)
            del ret[knn:]

        
            
    # AK should be fine for storing

class Tree(Base):
    def __init__(self):
        self.hashes=None
        self.root=Node(0)

    def _genHashes(self,dims):
        h=np.random.randn(256,dims)
        self.hashes=h*(1/np.linalg.norm(h,axis=1)).reshape(-1,1)

    def _calcVec(self,k):
        h=k/np.linalg.norm(k)
        return np.dot(self.hashes,h)


    def getRoot(self):
        return self.root

    def put(self,k,v,overwrite=False):
        if self.hashes is None:
            self._genHashes(k.shape[0])
        d=self._calcVec(k)
        return self.getRoot().put(k,d,v,overwrite)

    def delete(self,k):
        if self.hashes is None:
            self._genHashes(k.shape[0])
        d=self._calcVec(k)
        return self.getRoot().delete(k,d)

    def get(self,k):
        if self.hashes is None:
            self._genHashes(k.shape[0])
        d=self._calcVec(k)
        return self.getRoot().get(k,d)

    def get_first_item(self):
        return self.getRoot().get_first_item()

    def search(self,key,knn,numbkt=10):
        ret=[]
        if self.hashes is None:
            return ret
        v=self._calcVec(key)
        v2=v*v
        Q=[(0,self.getRoot())]
        while Q:
            prob,n=Q.pop()
            if len(n.data)>0:
                n._search(key,ret)
                numbkt-=1
                if numbkt<=0:
                    break
            if n.lo is not None:
                Q.append((prob+v2[n.level]*(v[n.level]>0),n.lo))
            if n.hi is not None:
                Q.append((prob+v2[n.level]*(v[n.level]<=0),n.hi))
            Q.sort(key=lambda x: x[0],reverse=True)
        ret.sort(key=lambda x: x[0],reverse=True)
        return ret[:knn]

    def msearch(self,keys,knn,numbkt=10):
        rets=[[] for k in keys]
        if self.hashes is None:
            return ret
        vs=[self._calcVec(k) for k in keys]
        v2s=[x*x for x in vs]
        Q=[(0,self.getRoot())]
        while Q:
            prob,n=Q.pop()
            if len(n.data)>0:
                n._msearch(keys,rets)
                numbkt-=1
                if numbkt<=0:
                    break
            if n.lo is not None:
                va=10000
                for i in range(len(vs)):
                    #print(i,vs[i],v2s[i])
                    va=min(va,v2s[i][n.level]*(vs[i][n.level]>0))
                Q.append((prob+va,n.lo))
            if n.hi is not None:
                va=10000
                for i in range(len(vs)):
                    va=min(va,v2s[i][n.level]*(vs[i][n.level]<=0))
                Q.append((prob+va,n.hi))
            Q.sort(key=lambda x: x[0],reverse=True)
        for ret in rets:
            ret.sort(key=lambda x: x[0],reverse=True)
            del ret[knn:]
        return rets   

    def recSearch(self,key,knn,numbkt=10):
        ret=[]
        if self.hashes is None:
            return ret
        v=self._calcVec(key)
        v2=v*v
        self.getRoot().recSearch(key,knn,v,v2,numbkt,ret)
        ret.sort(key=lambda x: x[0],reverse=True)
        return ret[:knn]

    def fullSearch(self,key,knn):
        ret=[]
        if self.hashes is None:
            return ret
        self.getRoot().fullSearch(key,knn,ret)
        ret.sort(key=lambda x: x[0],reverse=True)
        return ret[:knn]

    def __len__(self):
        return len(self.root)

    def numNodes(self):
        ret=0
        Q=[self.root]
        while len(Q)>0:
            n=Q.pop()
            ret+=1
            if n.lo is not None:
                Q.append(n.lo)
            if n.hi is not None:
                Q.append(n.lo)
        return ret

def fillTree(tree):
    for i in range(1000):
        k=np.random.randn(32)
        tree.put(k,i)
    return k


if __name__=="__main__":
    t=Tree()
    fillTree(t)
    keys=[np.random.randn(32) for i in range(4)]
    ret=t.msearch(keys,4)
