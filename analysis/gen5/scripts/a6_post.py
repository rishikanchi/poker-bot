import sys,os,collections,statistics as st,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.environ.get('GEN5_DIR','.'))  # folder holding the gen5 bot's core.py
import hp
import core  # gen5 core (copied)
R=hp.all_tables(); me=R['me']
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
rank=sorted((e for e in avg if e!=me),key=lambda e:-avg[e]); top=set(rank[:30]); bot=set(rank[-60:])
def grp(e): return 'ME' if e==me else 'TOP30' if e in top else 'BOT60' if e in bot else 'MID'
G=['ME','TOP30','MID','BOT60']; nt=len(R['tabs'])
C=collections.defaultdict(collections.Counter); sizes=collections.defaultdict(list)
myl=collections.defaultdict(list)
SN={1:'flop',2:'turn',3:'river'}
for t in R['tabs']:
    for h in t['hands']:
        if len(h['board'])<3: continue
        post=[a for a in h['acts'] if a['street']>0]
        if not post: continue
        pfr=h['pfr']; seen={}
        for g_ in set(grp(t['seats'][s]) for s in h['saw'][1]): pass
        # per street trackers
        for stt in (1,2,3):
            sa=[a for a in post if a['street']==stt]
            if not sa: continue
            first_bettor=None; nb=0
            prev_agg=h['street_agg'][stt-1] if stt>1 else pfr
            for a in sa:
                s=a['seat']; g=grp(t['seats'][s]); k=a['kind']
                hu='HU' if a['nlive']==2 else 'MW'
                if a['cur']==0:   # no bet yet: opportunity to bet
                    role='agg' if s==prev_agg else 'nonagg'
                    C[(g,SN[stt],'bet_opp',role,hu)][k=='raise']+=1
                    C[(g,SN[stt],'bet_opp',role,'all')][k=='raise']+=1
                    C[(g,SN[stt],'bet_opp','all','all')][k=='raise']+=1
                    if k=='raise': sizes[(g,SN[stt],'bet')].append(a['to']/max(1,a['pot']))
                else:  # facing bet
                    fr=a['to_call']/max(1,a['pot']-a['to_call']+0)  # bet as fraction of pot before bet (approx)
                    sz='small' if a['to_call']<=0.4*(a['pot']-a['to_call']) else 'big' if a['to_call']<=1.0*(a['pot']-a['to_call']) else 'over'
                    act='raise' if k=='raise' else k
                    C[(g,SN[stt],'facing',sz,'all')][act]+=1; C[(g,SN[stt],'facing','all','all')][act]+=1; C[(g,SN[stt],'facing','all',hu)][act]+=1
                    if a['nraise']>=2: C[(g,SN[stt],'facing_raise','all','all')][act]+=1
        # showdown stats
        for s in h['saw'][1]:
            g=grp(t['seats'][s]); wtsd=s in h['shows'] ; C[(g,'WTSD')][wtsd]+=1
            if wtsd: C[(g,'WSD')][h['wins'].get(s,0)>0]+=1
            C[(g,'WWSF')][h['wins'].get(s,0)>0]+=1
def rate(key,val=True):
    c=C[key]; n=sum(c.values()); return (100*c[val]/n if n else float('nan'),n)
print('== bet when no bet yet (cbet = street-before aggressor)')
for stt in ('flop','turn','river'):
    for role in ('agg','nonagg'):
        for hu in ('HU','MW','all'):
            print('%-5s %-6s %-3s '%(stt,role,hu)+' | '.join('%s %.0f%% (n=%d)'%(g,*rate((g,stt,'bet_opp',role,hu))) for g in G))
print('== bet sizes (fraction of pot) median / q25 / q75')
for stt in ('flop','turn','river'):
    out=[]
    for g in G:
        v=sorted(sizes[(g,stt,'bet')]); out.append('%s %.2f/%.2f/%.2f'%(g,v[len(v)//2],v[len(v)//4],v[3*len(v)//4]))
    print(stt,' | '.join(out))
print('== facing a bet: fold/call/raise')
for stt in ('flop','turn','river'):
    for sz in ('small','big','over','all'):
        out=[]
        for g in G:
            c=C[(g,stt,'facing',sz,'all')]; n=sum(c.values()); out.append('%s f=%.0f c=%.0f r=%.0f (n=%d)'%(g,100*c['fold']/max(1,n),100*c['call']/max(1,n),100*c['raise']/max(1,n),n))
        print('%-5s %-5s '%(stt,sz)+' | '.join(out))
    out=[]
    for g in G:
        c=C[(g,stt,'facing_raise','all','all')]; n=sum(c.values()); out.append('%s f=%.0f c=%.0f r=%.0f (n=%d)'%(g,100*c['fold']/max(1,n),100*c['call']/max(1,n),100*c['raise']/max(1,n),n))
    print('%-5s raise  '%stt+' | '.join(out))
for k in ('WTSD','WSD','WWSF'): print(k,' | '.join('%s %.1f%% (n=%d)'%(g,*rate((g,k))) for g in G))
# ---- my flop lines by strength bucket
SNAME=['air','weak pair','mid pair','TP weak','TP good','overpair','two pair','set/trips','straight/flush','nut flush/boat+']
rows=collections.defaultdict(list)
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        if len(h['board'])<3 or s not in h['saw'].get(1,()): continue
        post=[a for a in h['acts'] if a['street']>0]
        if not post: continue   # all-in preflop
        hole=[core.card(c) for c in t['mycards'][h['hand']-1]]; flop=[core.card(c) for c in h['board'][:3]]
        strength,eqr=core.features(hole,flop)
        mine=[a for a in post if a['seat']==s and a['street']==1]
        if not mine: continue
        a0=mine[0]; role='PFR' if h['pfr']==s else 'caller' if h['pfr'] is not None else 'limped'
        hu='HU' if len(h['saw'][1])==2 else 'MW'
        if a0['cur']==0: line='bet' if a0['kind']=='raise' else 'check'
        else: line='vsbet:'+a0['kind']
        d=h['eq1'][s]-h['eq0'][s]; bbw=(h['sa'][s]-h['sb4'][s])/h['bb']
        # subtract preflop investment to get postflop-only result: pot at flop share... keep total
        for key in ((strength,'all','all'),(strength,line,'all'),('all',line,role),('all','all',role),('all',line,hu),('ALL','ALL','ALL'),(strength,line,role)):
            rows[key].append((d,bbw))
print('\n== MY hands that saw a flop with betting (first flop action), result of whole hand: ICM pts and bb')
def pr(key):
    v=rows[key]
    if len(v)<25: return
    print('%-44s n=%5d  pts/hand %+.3f (se %.3f)  bb/hand %+.2f  pts/table %+.3f'%(str(key),len(v),st.mean(x[0] for x in v),st.pstdev([x[0] for x in v])/math.sqrt(len(v)),st.mean(x[1] for x in v),sum(x[0] for x in v)/nt))
pr(('ALL','ALL','ALL'))
for role in ('PFR','caller','limped'): pr(('all','all',role))
for line in ('bet','check','vsbet:fold','vsbet:call','vsbet:raise'):
    for x in ('PFR','caller','limped','HU','MW'): pr(('all',line,x))
for sidx in range(10):
    print(' --',SNAME[sidx]); pr((sidx,'all','all'))
    for line in ('bet','check','vsbet:fold','vsbet:call','vsbet:raise'): pr((sidx,line,'all'))
