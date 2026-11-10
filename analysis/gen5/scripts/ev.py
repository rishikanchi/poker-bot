import numpy as np
from numba import njit
RANKS="23456789TJQKA"; SUITS="cdhs"
def card(t): return RANKS.index(t[0])*4+SUITS.index(t[1])
@njit(cache=True)
def rank7(cs,n):
    counts=np.zeros(13,np.int64); suits=np.zeros(4,np.int64); mask=0
    for i in range(n):
        c=cs[i]; counts[c>>2]+=1; suits[c&3]+=1; mask|=1<<(c>>2)
    fm=0
    for s in range(4):
        if suits[s]>=5:
            for i in range(n):
                if cs[i]&3==s: fm|=1<<(cs[i]>>2)
    def sh(m):
        for top in range(12,3,-1):
            if (m>>(top-4))&31==31: return top
        if m&0b1000000001111==0b1000000001111: return 3
        return -1
    if fm:
        h=sh(fm)
        if h>=0: return (8<<20)|(h<<16)
    quads=-1; trips=-1; p1=-1; p2=-1
    for r in range(12,-1,-1):
        k=counts[r]
        if k==4: quads=r
        elif k==3:
            if trips<0: trips=r
            elif p1<0: p1=r
            elif p2<0: p2=r
        elif k==2:
            if p1<0: p1=r
            elif p2<0: p2=r
    if quads>=0:
        k=-1
        for r in range(12,-1,-1):
            if counts[r] and r!=quads: k=r; break
        return (7<<20)|(quads<<16)|(k<<12)
    if trips>=0 and p1>=0: return (6<<20)|(trips<<16)|(p1<<12)
    if fm:
        v=5<<20; sft=16; k=0
        for r in range(12,-1,-1):
            if (fm>>r)&1:
                v|=r<<sft; sft-=4; k+=1
                if k==5: break
        return v
    h=sh(mask)
    if h>=0: return (4<<20)|(h<<16)
    if trips>=0:
        v=(3<<20)|(trips<<16); sft=12; k=0
        for r in range(12,-1,-1):
            if counts[r]==1:
                v|=r<<sft; sft-=4; k+=1
                if k==2: break
        return v
    if p1>=0 and p2>=0:
        k=-1
        for r in range(12,-1,-1):
            if counts[r]>=1 and r!=p1 and r!=p2: k=r; break
        return (2<<20)|(p1<<16)|(p2<<12)|(k<<8)
    if p1>=0:
        v=(1<<20)|(p1<<16); sft=12; k=0
        for r in range(12,-1,-1):
            if counts[r]==1:
                v|=r<<sft; sft-=4; k+=1
                if k==3: break
        return v
    v=0; sft=16; k=0
    for r in range(12,-1,-1):
        if counts[r]==1:
            v|=r<<sft; sft-=4; k+=1
            if k==5: break
    return v
@njit(cache=True)
def eq_multi(holes,nh,board,nb,dead,nd):
    """exact equity share for nh players given nb board cards (3,4,5); dead cards excluded."""
    used=np.zeros(52,np.bool_)
    for i in range(nh): used[holes[i,0]]=True; used[holes[i,1]]=True
    for i in range(nb): used[board[i]]=True
    for i in range(nd): used[dead[i]]=True
    deck=np.zeros(52,np.int64); m=0
    for c in range(52):
        if not used[c]: deck[m]=c; m+=1
    out=np.zeros(nh); cs=np.zeros(7,np.int64); tot=0
    for i in range(nb): cs[2+i]=board[i]
    sc=np.zeros(nh,np.int64)
    if nb==5:
        best=-1;nbest=0
        for p in range(nh):
            cs[0]=holes[p,0]; cs[1]=holes[p,1]; sc[p]=rank7(cs,7)
            if sc[p]>best: best=sc[p]; nbest=1
            elif sc[p]==best: nbest+=1
        for p in range(nh):
            if sc[p]==best: out[p]=1.0/nbest
        return out
    if nb==4:
        for a in range(m):
            cs[6]=deck[a]; best=-1;nbest=0
            for p in range(nh):
                cs[0]=holes[p,0]; cs[1]=holes[p,1]; sc[p]=rank7(cs,7)
                if sc[p]>best: best=sc[p]; nbest=1
                elif sc[p]==best: nbest+=1
            for p in range(nh):
                if sc[p]==best: out[p]+=1.0/nbest
            tot+=1
        return out/tot
    for a in range(m):
        cs[5]=deck[a]
        for b in range(a+1,m):
            cs[6]=deck[b]; best=-1;nbest=0
            for p in range(nh):
                cs[0]=holes[p,0]; cs[1]=holes[p,1]; sc[p]=rank7(cs,7)
                if sc[p]>best: best=sc[p]; nbest=1
                elif sc[p]==best: nbest+=1
            for p in range(nh):
                if sc[p]==best: out[p]+=1.0/nbest
            tot+=1
    return out/tot
