import sys,os,collections,statistics as st,math,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp,ev,icm,numpy as np
from numba import njit
R=hp.all_tables(); me=R['me']; P=np.array([50.,30.,20.]); nt=len(R['tabs'])
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
rank=sorted((e for e in avg if e!=me),key=lambda e:-avg[e]); top=set(rank[:30])
def bbk(x): return 0 if x<4 else 1 if x<7 else 2 if x<10 else 3 if x<15 else 4 if x<25 else 5
BN=['<4','4-7','7-10','10-15','15-25','25+']
def role(p): return p if p in('SB','BB') else 'oth'
def pgrp(p): return p if p in('SB','BTN','CO') else 'EP'
@njit(cache=True)
def eq_vs(h0,h1,vill,ntr,seed):
    np.random.seed(seed); cs=np.zeros(7,np.int64); win=0.0; n=0; nv=len(vill)
    used=np.zeros(52,np.bool_)
    for _ in range(ntr):
        v=vill[np.random.randint(nv)]
        if v[0]==h0 or v[0]==h1 or v[1]==h0 or v[1]==h1: continue
        used[:]=False; used[h0]=True; used[h1]=True; used[v[0]]=True; used[v[1]]=True
        k=2
        while k<7:
            c=np.random.randint(52)
            if not used[c]: used[c]=True; cs[k]=c; k+=1
        cs[0]=h0; cs[1]=h1; a=ev.rank7(cs,7); cs[0]=v[0]; cs[1]=v[1]; b=ev.rank7(cs,7)
        win+= 1.0 if a>b else 0.5 if a==b else 0.0; n+=1
    return win/max(n,1)
def cls(c0,c1):
    r0,r1=c0>>2,c1>>2; hi,lo=max(r0,r1),min(r0,r1); return (hi,lo,(c0&3)==(c1&3) and hi!=lo)
def rep(k):
    hi,lo,s=k; return (hi*4, lo*4 if s else lo*4+1)
# ---------- collect field responses to open jams
callp=collections.defaultdict(lambda:[0,0]); callp_g=collections.defaultdict(lambda:[0,0])
call_combos=collections.defaultdict(list); jam_combos=collections.defaultdict(list)
jamfreq=collections.defaultdict(lambda:[0,0])
for t in R['tabs']:
    for h in t['hands']:
        pre=[a for a in h['acts'] if a['street']==0]
        # find open jam: first non-fold action is raise all-in with no limpers
        for i,a in enumerate(pre):
            if a['kind']=='fold': continue
            if a['kind']=='raise' and a.get('allin') and a['limpers']==0 and a['nraise']==0:
                js=a['seat']; jtot=a['to']; 
                for b in pre[i+1:]:
                    if b['seat']==js: break
                    ctot=b['stack']+ (b['cur']-b['to_call'] if False else 0)
                    size=min(jtot, b['stack']+ (h['bb'] if b['pos']=='BB' else h['sb'] if b['pos']=='SB' else 0))/h['bb']
                    key=(role(b['pos']),bbk(size)); called=b['kind'] in('call','raise')
                    e=t['seats'][b['seat']]
                    if e!=me: callp[key][0]+=called; callp[key][1]+=1
                    g='ME' if e==me else 'TOP30' if e in top else 'REST'
                    callp_g[(g,)+key][0]+=called; callp_g[(g,)+key][1]+=1
                    if called:
                        if b['seat'] in h['shows'] and e!=me: call_combos[bbk(size)].append([ev.card(c) for c in h['shows'][b['seat']]])
                        if js in h['shows'] and t['seats'][js]!=me:
                            jam_combos[(pgrp(a['pos']),bbk(min(jtot,size*h['bb'])/h['bb']))].append([ev.card(c) for c in h['shows'][js]])
                        break
            break
print('P(call) vs open-jam by role x jam size(bb, capped by caller stack): field (excl me) | ME | TOP30')
for r in ('BB','SB','oth'):
    for k in range(6):
        c=callp[(r,k)]; m=callp_g[('ME',r,k)]; tp=callp_g[('TOP30',r,k)]
        if c[1]>30: print('  %-3s %-6s field %.1f%% (n=%d) | ME %.1f%% (n=%d) | TOP30 %.1f%% (n=%d)'%(r,BN[k],100*c[0]/c[1],c[1],100*m[0]/max(1,m[1]),m[1],100*tp[0]/max(1,tp[1]),tp[1]))
print('caller combos',{BN[k]:len(v) for k,v in sorted(call_combos.items())}); print('jam combos',{(k[0],BN[k[1]]):len(v) for k,v in sorted(jam_combos.items())})
CC={k:np.array(v) for k,v in call_combos.items()}
JC={}
for k,v in jam_combos.items(): JC[k]=np.array(v)
def jam_range(pg,k):
    if k>=5: return None
    if (pg,k) in JC and len(JC[(pg,k)])>=80: return (pg,k)
    return None
cache={}
def eqc(hole,kind,key):
    k=(cls(*hole),kind,key)
    if k not in cache:
        r=rep(cls(*hole)); arr=CC[key] if kind=='c' else JC[key]
        cache[k]=eq_vs(r[0],r[1],arr,20000,12345)
    return cache[k]
def icm_of(stacks,s):
    if stacks[s]<=0:
        k=sum(1 for x in stacks if x>0); return float(P[k]) if k<3 else 0.0
    return icm.icm(np.array(stacks,float),P)[s]
# ---------- A: my RFI spots
res=collections.defaultdict(list); calib=[]
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        pre=[a for a in h['acts'] if a['street']==0]
        mine=[a for a in pre if a['seat']==s]
        if not mine: continue
        a=mine[0]
        if not(a['nraise']==0 and a['limpers']==0) or a['kind']=='check': continue
        mybb=(a['stack']+ (a['cur']-a['to_call']))/h['bb']   # total incl posted
        mytot=a['stack']+(a['cur']-a['to_call']); 
        if mytot/h['bb']>20: continue
        hole=[ev.card(c) for c in t['mycards'][h['hand']-1]]
        # stacks now (after blinds posted)
        cur={x:h['sb4'][x] for x in range(8)}; posted={x:0 for x in range(8)}
        for e2 in pre: pass
        bl={}
        for x,p in h['pos'].items():
            if p=='SB': bl[x]=min(h['sb'],h['sb4'][x])
            if p=='BB': bl[x]=min(h['bb'],h['sb4'][x])
        if h['n']==2:
            pass
        pot=sum(bl.values()); bbseat=[x for x,p in h['pos'].items() if p=='BB'][0]
        base=list(h['sb4'])
        # fold: my blind lost, BB gets pot
        f=list(base)
        for x,v in bl.items(): f[x]-=v
        f[bbseat]+=pot; ev_fold=icm_of(f,s)
        # behind players in order
        idx=pre.index(a); order=[]
        seen=set()
        alive_order=[x for x in h['alive']]
        # players yet to act: those after me in preflop order up to BB
        act_order=[b['seat'] for b in pre]
        # derive order from positions
        POS=hp.POSN[h['n']]; mypos=h['pos'][s]
        behind=[x for x,p in h['pos'].items() if POS.index(p)>POS.index(mypos)]
        behind.sort(key=lambda x:POS.index(h['pos'][x]))
        pnone=1.0; evj=0.0
        for x in behind:
            vtot=h['sb4'][x]; size=min(mytot,vtot)/h['bb']; k=bbk(size); c=callp[(role(h['pos'][x]),k)]
            pc=c[0]/c[1] if c[1]>20 else 0.1
            kk=k if k in CC and len(CC[k])>50 else min(CC,key=lambda z:abs(z-k))
            eq=eqc(hole,'c',kk); m=min(mytot,vtot)
            w=list(base); l=list(base)
            for y,v in bl.items():
                if y!=s and y!=x: w[y]-=v; l[y]-=v
            dead=sum(v for y,v in bl.items() if y!=s and y!=x)
            w[s]=base[s]+m+dead; w[x]=base[x]-m
            l[s]=base[s]-m; l[x]=base[x]+m+dead
            evj+=pnone*pc*(eq*icm_of(w,s)+(1-eq)*icm_of(l,s)); pnone*=1-pc
        w=list(base)
        for y,v in bl.items():
            if y!=s: w[y]-=v
        w[s]=base[s]+sum(v for y,v in bl.items() if y!=s)
        evj+=pnone*icm_of(w,s)
        act='jam' if (a['kind']=='raise' and a.get('allin')) else a['kind']
        d=evj-ev_fold; rb=bbk(mytot/h['bb'])
        rec=(d,pnone,h['eq1'][s]-h['eq0'][s],ev_fold-h['eq0'][s])
        for key in ((act,'all','all'),(act,mypos,'all'),(act,'all',BN[rb]),(act,mypos,BN[rb]),(act,'n=%d'%h['n'],'all')): res[key].append(rec)
        if act=='jam': calib.append((evj-h['eq0'][s],h['eq1'][s]-h['eq0'][s]))
print('\n== A. my first-in spots with <=20bb: counterfactual ICM EV of open-jamming vs folding (pts)')
print('calibration on my actual jams: predicted %+.3f  realized %+.3f pts/jam (n=%d)'%(st.mean(x[0] for x in calib),st.mean(x[1] for x in calib),len(calib)))
def pr(key):
    v=res[key]
    if len(v)<15: return
    pos=[x[0] for x in v if x[0]>0]; big=[x[0]-0.2 for x in v if x[0]>0.5]
    print('     with model haircut (need >+0.5pt, credit EV-0.2): %d spots (%.0f%%), %+.3f pts/table'%(len(big),100*len(big)/len(v),sum(big)/nt))
    print('  %-28s n=%5d  mean(jam-fold) %+.3f  jam better in %.0f%%  missed EV if all +EV jammed: %+.3f pts/table  avg P(all fold) %.0f%%'%(str(key),len(v),st.mean(x[0] for x in v),100*len(pos)/len(v),sum(pos)/nt,100*st.mean(x[1] for x in v)))
for act in ('fold','call','raise','jam'):
    pr((act,'all','all'))
    for p in ['UTG','UTG1','LJ','HJ','CO','BTN','SB']: pr((act,p,'all'))
    for b in BN: pr((act,'all',b))
    for n in range(2,9): pr((act,'n=%d'%n,'all'))
for p in ['CO','BTN','SB']:
    for b in BN: pr(('fold',p,b))
# realized for raise (non-jam) and limp at <=20bb
for act in ('raise','call'):
    v=res[(act,'all','all')]
    if v: print('  realized my %s at <=20bb: %+.3f pts/hand vs counterfactual jam %+.3f (rel. to fold %+.3f)'%(act,st.mean(x[2] for x in v),st.mean(x[0]+x[3] for x in v),st.mean(x[3] for x in v)))
# ---------- B: facing a single open jam
resB=collections.defaultdict(list)
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        pre=[a for a in h['acts'] if a['street']==0]
        mine=[a for a in pre if a['seat']==s]
        if not mine: continue
        a=mine[0]
        if a['nraise']!=1 or a['callers']!=0 or a['limpers']!=0: continue
        i=pre.index(a); rz=[b for b in pre[:i] if b['kind']=='raise']
        if not rz or not rz[0].get('allin'): continue
        j=rz[0]; js=j['seat']; mytot=a['stack']+(a['cur']-a['to_call']); m=min(mytot,j['to'])
        key=jam_range(pgrp(j['pos']),bbk(m/h['bb']))
        if key is None: continue
        hole=[ev.card(c) for c in t['mycards'][h['hand']-1]]; eq=eqc(hole,'j',key)
        bl={}
        for x,p in h['pos'].items():
            if p=='SB': bl[x]=min(h['sb'],h['sb4'][x])
            if p=='BB': bl[x]=min(h['bb'],h['sb4'][x])
        base=list(h['sb4']); dead=sum(v for y,v in bl.items() if y!=s and y!=js)
        f=list(base)
        for y,v in bl.items(): f[y]-=v
        f[js]=base[js]+sum(v for y,v in bl.items() if y!=js)
        w=list(base); l=list(base)
        for y,v in bl.items():
            if y!=s and y!=js: w[y]-=v; l[y]-=v
        w[s]=base[s]+m+dead; w[js]=base[js]-m; l[s]=base[s]-m; l[js]=base[js]+m+dead
        ef=icm_of(f,s); ec=eq*icm_of(w,s)+(1-eq)*icm_of(l,s)
        act=a['kind'] if a['kind']=='fold' else 'call'
        need=(m-bl.get(s,0))/(2*m+dead)  # chip pot odds
        closing=h['pos'][s]=='BB'
        rec=(ec-ef,eq,need,h['eq1'][s]-h['eq0'][s])
        for k2 in ((act,'all'),(act,h['pos'][s] if h['pos'][s] in('BB','SB') else 'oth'),(act,'jam '+BN[bbk(m/h['bb'])]),(act,'n=%d'%h['n'])): resB[k2].append(rec)
print('\n== B. my decisions facing a single open-jam (first to respond): ICM EV call - fold using field jam ranges')
for act in ('fold','call'):
    for k2 in sorted(k for k in resB if k[0]==act):
        v=resB[k2]
        if len(v)<15: continue
        pos=[x[0] for x in v if x[0]>0]; neg=[x[0] for x in v if x[0]<0]; big=[x[0] for x in v if x[0]>0.5]
        print('     clear (+0.5pt margin) calls: %d spots, %+.3f pts/table'%(len(big),sum(big)/nt))
        print('  %-22s n=%5d mean eq %.3f  chip-odds need %.3f  mean(call-fold) %+.3f  call better %.0f%%  sum(+)=%+.3f sum(-)=%+.3f pts/table'%(str(k2),len(v),st.mean(x[1] for x in v),st.mean(x[2] for x in v),st.mean(x[0] for x in v),100*len(pos)/len(v),sum(pos)/nt,sum(neg)/nt))
pickle.dump(dict(callp=dict(callp),res={k:v for k,v in res.items()},resB=dict(resB)),open('pf.pkl','wb'))
