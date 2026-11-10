import sys,os,collections,statistics as st,math,itertools
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp,ev,numpy as np
R=hp.all_tables(); me=R['me']; names=R['names']
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
rank=sorted((e for e in avg if e!=me),key=lambda e:-avg[e]); top=set(rank[:30]); bot=set(rank[-60:])
def grp(e): return 'ME' if e==me else 'TOP30' if e in top else 'BOT60' if e in bot else 'MID'
TR=list(itertools.combinations(range(5),3))
rows=[]
for t in R['tabs']:
    for h in t['hands']:
        if not h['auction'] or len(h['bids'])!=2: continue
        a,b=h['bids']
        if a not in h['shows'] or b not in h['shows']: continue
        # all-in before the flop? (no postflop actions)
        allin_pre=not any(x['street']>0 for x in h['acts'])
        holes=np.array([[ev.card(c) for c in h['shows'][a]],[ev.card(c) for c in h['shows'][b]]])
        auc=[ev.card(c) for c in h['auction']]; flop=tuple(sorted(h['auction'].index(c) for c in h['board'][:3]))
        eqs=[]
        for tr in TR:
            board=np.array([auc[i] for i in tr]+[0,0]); dead=np.array([auc[i] for i in range(5) if i not in tr])
            eqs.append(ev.eq_multi(holes,2,board,3,dead,2)[0])
        ia=TR.index(flop)
        potA=sum(x.get('amt',0) for x in h['acts'] if x['street']==0)+h['sb']+h['bb']
        # exact pot for allin: min matched
        rows.append(dict(t=t,h=h,a=a,b=b,eqs=eqs,ia=ia,allin=allin_pre,pa=h['fc0'][a]-h['fc1'][a],pb=h['fc0'][b]-h['fc1'][b],pot=potA))
print('HU auctions with both hands shown',len(rows),'all-in preflop',sum(r['allin'] for r in rows))
import pickle; pickle.dump([{k:v for k,v in r.items() if k not in('t','h')}|{'it':r['t']['it'],'hand':r['h']['hand']} for r in rows],open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'aucedge.pkl'),'wb'))
for name,sel in (('ALL-IN PREFLOP',[r for r in rows if r['allin']]),('SAW POSTFLOP BETTING + showdown',[r for r in rows if not r['allin']])):
    print('\n==',name,len(sel))
    spread=[max(r['eqs'])-min(r['eqs']) for r in sel]; print('equity spread across the 10 possible flops: mean %.3f q=%s'%(st.mean(spread),[round(x,2) for x in sorted(spread)[::max(1,len(spread)//6)]]))
    print('mean |eq(actual)-eq(default first3)| %.3f'%st.mean(abs(r['eqs'][r['ia']]-r['eqs'][0]) for r in sel))
    # player-level: gain vs default & vs mean, by paid more or less
    G=collections.defaultdict(list)
    for r in sel:
        for who,sign in (('a',1),('b',-1)):
            s=r[who]; e=r['t']['seats'][s]; g=grp(e)
            eq=[x if sign==1 else 1-x for x in r['eqs']]
            act=eq[r['ia']]; mine=r['pa'] if who=='a' else r['pb']; opp=r['pb'] if who=='a' else r['pa']
            rec=dict(g=g,gain_def=act-eq[0],gain_mean=act-st.mean(eq),best=max(eq),worst=min(eq),act=act,mine=mine,opp=opp,pot=r['pot'],bb=r['h']['bb'],
                     pct=(act-min(eq))/(max(eq)-min(eq)) if max(eq)>min(eq) else .5)
            G[g].append(rec); G['ALL'].append(rec)
    for g in ['ME','TOP30','MID','BOT60','ALL']:
        v=G[g]
        if not v: continue
        print('%-6s n=%5d paid=%.1f opp_paid=%.1f  eq_gain_vs_mean=%+.4f vs_default=%+.4f  capture(0=worst flop,1=best)=%.3f  potbb=%.1f'%(g,len(v),st.mean(x['mine'] for x in v),st.mean(x['opp'] for x in v),st.mean(x['gain_mean'] for x in v),st.mean(x['gain_def'] for x in v),st.mean(x['pct'] for x in v),st.mean(x['pot']/x['bb'] for x in v)))
    # capture by paid-difference bucket (all players)
    print(' capture & eq gain vs mean by (my paid - opp paid):')
    bk=collections.defaultdict(list)
    for x in G['ALL']:
        d=x['mine']-x['opp']; k='<=-100' if d<=-100 else '-100..-30' if d<=-30 else '-30..-5' if d<-5 else '-5..5' if d<=5 else '5..30' if d<30 else '30..100' if d<100 else '>=100'
        bk[k].append(x)
    for k in ['<=-100','-100..-30','-30..-5','-5..5','5..30','30..100','>=100']:
        v=bk[k]; 
        if v: print('   %-10s n=%5d capture=%.3f gain_vs_mean=%+.4f  gain*pot(bb)=%+.2f bb'%(k,len(v),st.mean(x['pct'] for x in v),st.mean(x['gain_mean'] for x in v),st.mean(x['gain_mean']*x['pot']/x['bb'] for x in v)))
    # by ratio of paid share
    print(' capture by my share of total paid:')
    bk=collections.defaultdict(list)
    for x in G['ALL']:
        tot=x['mine']+x['opp']
        if tot==0: bk['none paid'].append(x); continue
        sh=x['mine']/tot; bk['0' if sh==0 else '<25%' if sh<.25 else '25-50%' if sh<.5 else '50-75%' if sh<.75 else '75-<100%' if sh<1 else '100%'].append(x)
    for k in ['none paid','0','<25%','25-50%','50-75%','75-<100%','100%']:
        v=bk[k]
        if v: print('   %-10s n=%5d capture=%.3f gain_vs_mean=%+.4f'%(k,len(v),st.mean(x['pct'] for x in v),st.mean(x['gain_mean'] for x in v)))
