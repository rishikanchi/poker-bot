import numpy as np
from numba import njit
@njit(cache=True)
def icm(stacks,payouts):
    n=len(stacks); out=np.zeros(n); tot=stacks.sum()
    for a in range(n):
        if stacks[a]<=0: continue
        pa=stacks[a]/tot; out[a]+=pa*payouts[0]; ra=tot-stacks[a]
        if ra<=0: continue
        for b in range(n):
            if b==a or stacks[b]<=0: continue
            pb=pa*stacks[b]/ra; out[b]+=pb*payouts[1]; rb=ra-stacks[b]
            if rb<=0: continue
            for c in range(n):
                if c==a or c==b or stacks[c]<=0: continue
                out[c]+=pb*stacks[c]/rb*payouts[2]
    return out
