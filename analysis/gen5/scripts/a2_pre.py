import sys,os,collections,statistics as st,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp
R=hp.all_tables(); me=R['me']; names=R['names']
# groups by avg payout
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
rank=sorted((e for e in avg if e!=me),key=lambda e:-avg[e])
top=set(rank[:30]); bot=set(rank[-60:])
def grp(e): return 'ME' if e==me else 'TOP30' if e in top else 'BOT60' if e in bot else 'MID'
G=['ME','TOP30','MID','BOT60']
def reg(x): return '<10' if x<10 else '10-20' if x<20 else '20-40' if x<40 else '40+'
REG=['<10','10-20','20-40','40+']
cnt=collections.defaultdict(collections.Counter)   # key -> Counter(action)
eqd=collections.defaultdict(list)  # my equity delta by key
for t in R['tabs']:
    for h in t['hands']:
        first={}; 
        for a in h['acts']:
            if a['street']!=0: break
            s=a['seat']; g=grp(t['seats'][s]); effbb=a['eff']/a['bb']; r=reg(effbb)
            kind=a['kind']; jam=a.get('allin',False) and kind=='raise'
            act='jam' if jam else kind
            if s not in first:
                first[s]=act
                # situation
                if a['nraise']==0 and a['limpers']==0: sit='RFI'
                elif a['nraise']==0: sit='vsLimp'
                elif a['nraise']==1: sit='vsOpen'
                else: sit='vs3b+'
                # was the last raise an all-in?
                cnt[(g,sit,'all','all')][act]+=1
                cnt[(g,sit,a['pos'],'all')][act]+=1
                cnt[(g,sit,'all',r)][act]+=1
                cnt[(g,sit,a['pos'],r)][act]+=1
                cnt[(g,'VPIP','all','all')][kind in('call','raise')]+=1
                cnt[(g,'VPIP','all',r)][kind in('call','raise')]+=1
                cnt[(g,'VPIP','n%d'%min(h['n'],6),'all')][kind in('call','raise')]+=1
                cnt[(g,'PFR','all','all')][kind=='raise']+=1
                cnt[(g,'PFR','all',r)][kind=='raise']+=1
                cnt[(g,'PFR','n%d'%min(h['n'],6),'all')][kind=='raise']+=1
                if g=='ME':
                    d=h['eq1'][s]-h['eq0'][s]
                    eqd[(sit,act,r)].append(d); eqd[(sit,act,'all')].append(d); eqd[('pos',a['pos'],'all')].append(d); eqd[('pos',a['pos'],r)].append(d)
                    eqd[('reg',r,'all')].append(d); eqd[('n',h['n'],'all')].append(d)
            else:
                # second action preflop: opener facing 3bet etc
                if first[s] in('raise',) and a['nraise']>=2:
                    cnt[(g,'openThenFacing3b','all','all')][act]+=1; cnt[(g,'openThenFacing3b','all',r)][act]+=1
                    if g=='ME': eqd[('openF3b',act,'all')].append(h['eq1'][s]-h['eq0'][s])
                if first[s]=='call' and a['nraise']>=1:
                    cnt[(g,'limpcallThenRaised','all','all')][act]+=1
def show(sit,pos='all',r='all',acts=('fold','check','call','raise','jam')):
    out=[]
    for g in G:
        c=cnt[(g,sit,pos,r)]; n=sum(c.values())
        if n==0: out.append('%s n=0'%g); continue
        out.append('%s n=%d '%(g,n)+' '.join('%s=%.0f%%'%(a[:2],100*c[a]/n) for a in acts if c[a]))
    print('%-9s %-5s %-5s | '%(sit,pos,r)+' | '.join(out))
for k in('VPIP','PFR'):
    for r in ['all']+REG: 
        print(k,r,' '.join('%s=%.1f%%'%(g,100*cnt[(g,k,'all',r)][True]/max(1,sum(cnt[(g,k,'all',r)].values()))) for g in G))
    for n in range(2,7): print(k,'n=%d'%n,' '.join('%s=%.1f%%'%(g,100*cnt[(g,k,'n%d'%n,'all')][True]/max(1,sum(cnt[(g,k,'n%d'%n,'all')].values()))) for g in G))
print()
POS=['UTG','UTG1','LJ','HJ','CO','BTN','SB','BB']
for sit in('RFI','vsLimp','vsOpen','vs3b+'):
    show(sit)
    for r in REG: show(sit,'all',r)
    for p in POS: show(sit,p)
    print()
for r in REG:
    for p in POS: show('RFI',p,r)
for r in REG:
    for p in POS: show('vsOpen',p,r)
show('openThenFacing3b'); [show('openThenFacing3b','all',r) for r in REG]; show('limpcallThenRaised')
print('\n=== MY ICM equity delta (points; total per table = sum) ===')
nt=len(R['tabs'])
def pe(k):
    v=eqd[k]; 
    if len(v)<5: return
    m=st.mean(v); se=st.pstdev(v)/math.sqrt(len(v))
    print('%-34s n=%6d mean=%+.4f se=%.4f  total/table=%+.3f'%(str(k),len(v),m,se,sum(v)/nt))
for k in sorted(eqd,key=lambda k:(k[0],str(k[2]),str(k[1]))): pe(k)
