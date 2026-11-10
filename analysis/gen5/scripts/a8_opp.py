import sys,os,collections,statistics as st,math,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp
R=hp.all_tables(); me=R['me']; names=R['names']; nt=len(R['tabs'])
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
F=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
def add(e,k,v): F[e][k][0]+=v; F[e][k][1]+=1
for t in R['tabs']:
    for h in t['hands']:
        pre=[a for a in h['acts'] if a['street']==0]; first={}
        openjam=None
        for i,a in enumerate(pre):
            s=a['seat']; e=t['seats'][s]; eff=a['eff']/a['bb']; k=a['kind']; jam=k=='raise' and a.get('allin',False)
            if s in first: 
                if first[s]=='raise' and a['nraise']>=2: add(e,'fold_to_3b',k=='fold')
                continue
            first[s]='jam' if jam else k
            add(e,'vpip',k in('call','raise')); add(e,'pfr',k=='raise')
            if a['nraise']==0 and a['limpers']==0 and k!='check':
                add(e,'rfi',k=='raise'); add(e,'limp',k=='call')
                if eff>=15: add(e,'openjam15+',jam)
                if eff<10: add(e,'openjam<10',jam)
                if a['pos'] in('BTN','SB','CO'): add(e,'steal',k=='raise')
            elif a['nraise']==1:
                # facing open: was it a jam?
                rz=[b for b in pre[:i] if b['kind']=='raise'][0]
                if rz.get('allin'):
                    add(e,'call_jam',k!='fold')
                    if a['pos']=='BB' and a['callers']==0: add(e,'bb_call_jam',k!='fold')
                else:
                    add(e,'3bet',k=='raise')
                    if a['pos']=='BB' and a['callers']==0 and rz['pos'] in('BTN','SB','CO'): add(e,'bb_fold_steal',k=='fold')
        # postflop
        post=[a for a in h['acts'] if a['street']>0]
        for a in post:
            e=t['seats'][a['seat']]
            if a['street']==1:
                if a['cur']==0 and a['seat']==h['pfr']: add(e,'cbet',a['kind']=='raise')
                if a['cur']>0 and a['nraise']==1: add(e,'fold_flop_bet',a['kind']=='fold')
            if a['cur']>0: add(e,'fold_any_bet',a['kind']=='fold'); add(e,'raise_any_bet',a['kind']=='raise')
            if a['cur']==0: add(e,'bet_any',a['kind']=='raise')
        if h['auction']:
            for s in h['bids']: 
                e=t['seats'][s]; F[e]['paid'][0]+=h['fc0'][s]-h['fc1'][s]; F[e]['paid'][1]+=1
                for s2 in h['saw'].get(1,()):
                    pass
        for s in h['saw'].get(1,()) if post else ():
            add(t['seats'][s],'wtsd',s in h['shows'])
KS=['vpip','pfr','rfi','limp','steal','openjam15+','openjam<10','3bet','fold_to_3b','bb_fold_steal','call_jam','bb_call_jam','cbet','fold_flop_bet','fold_any_bet','raise_any_bet','bet_any','wtsd','paid']
def val(e,k):
    c=F[e][k]; return c[0]/c[1] if c[1]>=(8 if k!='paid' else 5) else float('nan')
ents=[e for e in avg]
print('field distribution per entry (10/25/50/75/90 pct) and ME, + corr with avg payout')
def corr(a,b):
    ma,mb=st.mean(a),st.mean(b); va=sum((x-ma)**2 for x in a); vb=sum((y-mb)**2 for y in b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb) if va>0 and vb>0 else 0
for k in KS:
    xs=[(val(e,k),avg[e]) for e in ents if e!=me and val(e,k)==val(e,k)]
    v=sorted(x[0] for x in xs); q=[v[int(p*(len(v)-1))] for p in (.1,.25,.5,.75,.9)]
    fmt=(lambda x:'%5.1f'%x) if k=='paid' else (lambda x:'%4.0f%%'%(100*x))
    print('%-14s %s | ME %s | corr %+.2f'%(k,' '.join(fmt(x) for x in q),fmt(val(me,k)),corr([x[0] for x in xs],[x[1] for x in xs])))
# archetypes
print('\nextreme archetypes (count of entries, their avg payout):')
def cnt(name,cond):
    es=[e for e in ents if e!=me and cond(e)]
    print('  %-46s %3d entries  avg payout %.1f  e.g. %s'%(name,len(es),st.mean(avg[e] for e in es) if es else 0,', '.join(names[e] for e in es[:5])))
    return set(es)
man=cnt('maniac: open-jams >=30% at 15bb+',lambda e:val(e,'openjam15+')>=0.30)
cnt('semi: open-jams 8-30% at 15bb+',lambda e:0.08<=val(e,'openjam15+')<0.30)
nit=cnt('nit: VPIP < 12%',lambda e:val(e,'vpip')<0.12)
cnt('never raises: PFR < 3%',lambda e:val(e,'pfr')<0.03)
cnt('limper: limp >= 15% first in',lambda e:val(e,'limp')>=0.15)
cnt('BB over-folds to steals: >= 85%',lambda e:val(e,'bb_fold_steal')>=0.85)
cnt('BB never folds to steal: <= 35%',lambda e:val(e,'bb_fold_steal')<=0.35)
cnt('never calls jams: call_jam <= 3%',lambda e:val(e,'call_jam')<=0.03)
cnt('calls jams light: call_jam >= 20%',lambda e:val(e,'call_jam')>=0.20)
cnt('folds to any postflop bet >= 65%',lambda e:val(e,'fold_any_bet')>=0.65)
cnt('station: folds to postflop bet <= 30%',lambda e:val(e,'fold_any_bet')<=0.30)
cnt('folds to 3bet >= 80%',lambda e:val(e,'fold_to_3b')>=0.80)
cnt('never bids: paid/auction < 2',lambda e:val(e,'paid')<2)
cnt('big bidder: paid/auction >= 60',lambda e:val(e,'paid')>=60)
pickle.dump(dict(F={e:{k:list(v) for k,v in d.items()} for e,d in F.items()},maniac=man,nit=nit),open('opp.pkl','wb'))
# how fast can we identify? fraction of my table-opponents (seat-tables) falling in archetypes
print('\nper table, how many of my 7 opponents are: maniac %.2f  nit %.2f'%(sum(1 for t in R['tabs'] for e in t['seats'] if e in man)/nt,sum(1 for t in R['tabs'] for e in t['seats'] if e in nit)/nt))
# my results vs archetype presence
import numpy as np
mp=[];mn=[]
for t in R['tabs']:
    mine=t['hands'][-1]['eq1'][t['me']]; k=sum(1 for e in t['seats'] if e in man); mp.append((k,mine))
for k in range(0,5):
    v=[x[1] for x in mp if x[0]==k]
    if len(v)>20: print('  tables with %d maniacs: n=%d my avg payout %.2f'%(k,len(v),st.mean(v)))
# my payout vs avg strength of table
strength=[(st.mean(avg[e] for e in t['seats'] if e!=me),t['hands'][-1]['eq1'][t['me']]) for t in R['tabs']]
strength.sort()
for i in range(0,1001,250):
    v=strength[i:i+250]; print('  table-strength quartile (opp mean payout %.1f): my payout %.2f'%(st.mean(x[0] for x in v),st.mean(x[1] for x in v)))
