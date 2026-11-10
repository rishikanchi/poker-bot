import sys,os,collections,statistics as st,math,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp
R=hp.all_tables(); me=R['me']
def q(v,qs=(.1,.25,.5,.75,.9)):
    v=sorted(v); return [v[min(len(v)-1,int(x*len(v)))] for x in qs] if v else []
nt=len(R['tabs'])
# (1) all-in swing
AE=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'aucedge.pkl'),'rb'))
idx={(t['it'],h['hand']):(t,h) for t in R['tabs'] for h in t['hands']}
win=[];lose=[];cnt=0; gains=[]; rows=[]
for r in AE:
    t,h=idx[(r['it'],r['hand'])]; s=t['me']
    if s not in (r['a'],r['b']) : continue
    eq=[x if s==r['a'] else 1-x for x in r['eqs']]
    mine=r['pa'] if s==r['a'] else r['pb']; opp=r['pb'] if s==r['a'] else r['pa']
    d=h['eq1'][s]-h['eq0'][s]; chips=h['sa'][s]-h['sb4'][s]
    pe=t['priv'].get(h['hand'],[]); bid=sum(e['amount'] for e in pe if e['kind']=='bid'); ncards=sum(1 for e in pe if e['kind']=='bid')
    rows.append(dict(allin=r['allin'],act=eq[r['ia']],mean=st.mean(eq),best=max(eq),d=d,chips=chips,mine=mine,opp=opp,bid=bid,ncards=ncards,pot=r['pot'],bb=h['bb'],n=h['n'],hand=h['hand'],fc=h['fc0'][s],
                     risk=min(h['sb4'][s],h['sb4'][r['b'] if s==r['a'] else r['a']])))
A=[x for x in rows if x['allin']]
w=[x['d'] for x in A if x['chips']>0]; l=[x['d'] for x in A if x['chips']<0]
S=st.mean(w)-st.mean(l)
print('my HU all-in-preflop showdowns: %d (%.2f/table)  ICM pts when win %+.2f  lose %+.2f  swing %.2f pts'%(len(A),len(A)/nt,st.mean(w),st.mean(l),S))
print('  my avg equity on actual flop %.3f | avg over 10 flops %.3f | best flop %.3f ; won %.1f%%'%(st.mean(x['act'] for x in A),st.mean(x['mean'] for x in A),st.mean(x['best'] for x in A),100*len(w)/len(A)))
# precise: sum over hands of delta-equity * (per-hand swing approximated by S scaled by risk?) use per-hand: value = (act-mean)
print('  current auction result: %+.3f equity/all-in => %+.3f pts/table'%(st.mean(x['act']-x['mean'] for x in A),st.mean(x['act']-x['mean'] for x in A)*S*len(A)/nt))
for tgt in (0.0,0.05,0.10): print('  if we averaged %+.2f equity vs flop-mean (TOP30 get +0.02, outbidders +0.10): %+.2f pts/table vs now'%(tgt,(tgt-st.mean(x['act']-x['mean'] for x in A))*S*len(A)/nt))
print('  my bid total in all-in pots: mean %.1f q=%s ; zero %.0f%%; chips available mean %.0f; opp paid mean %.1f q=%s'%(st.mean(x['bid'] for x in A),q([x['bid'] for x in A]),100*sum(1 for x in A if x['bid']==0)/len(A),st.mean(x['fc'] for x in A),st.mean(x['opp'] for x in A),q([x['opp'] for x in A])))
print('  all-in pots by my bid bucket: n, capture-equity vs mean, opp paid')
bk=collections.defaultdict(list)
for x in A: bk['0' if x['bid']==0 else '1-10' if x['bid']<=10 else '11-35' if x['bid']<=35 else '36-110' if x['bid']<=110 else '111+'].append(x)
for k in ['0','1-10','11-35','36-110','111+']:
    v=bk[k]
    if v: print('    bid %-7s n=%4d  eq-vs-mean %+.3f  opp_paid %.0f  pot %.0fbb  my paid %.0f'%(k,len(v),st.mean(x['act']-x['mean'] for x in v),st.mean(x['opp'] for x in v),st.mean(x['pot']/x['bb'] for x in v),st.mean(x['mine'] for x in v)))
# (2) all my auctions from private events
allb=[]; 
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        if not h['auction'] or s not in h['bids']: continue
        pe=t['priv'].get(h['hand'],[]); bids={e['cards'][0]:e['amount'] for e in pe if e['kind']=='bid'}
        flop=h['board'][:3]; got=sum(1 for c in bids if c in flop)
        allin=not any(x['street']>0 for x in h['acts'])
        potbb=(sum(x.get('amt',0) for x in h['acts'] if x['street']==0)+h['sb']+h['bb'])/h['bb']
        opp=sum(h['fc0'][o]-h['fc1'][o] for o in h['bids'] if o!=s)
        allb.append(dict(n=len(bids),tot=sum(bids.values()),got=got,allgot=got==len(bids),allin=allin,potbb=potbb,opp=opp,fc=h['fc0'][s],hand=h['hand'],d=h['eq1'][s]-h['eq0'][s],nb=len(h['bids'])))
print('\nmy auctions',len(allb),'per table %.1f'%(len(allb)/nt))
print('cards bid on:',sorted(collections.Counter(x['n'] for x in allb).items()))
print('bid total q',q([x['tot'] for x in allb],(.1,.25,.5,.75,.9,.95,.99)),'mean %.1f'%st.mean(x['tot'] for x in allb))
for n in (1,2,3):
    v=[x for x in allb if x['n']==n]
    if v: print(' bid on %d cards: n=%d all selected %.0f%%  mean cards got %.2f  mean total bid %.1f  opp paid %.1f'%(n,len(v),100*sum(x['allgot'] for x in v)/len(v),st.mean(x['got'] for x in v),st.mean(x['tot'] for x in v),st.mean(x['opp'] for x in v)))
print(' P(all my cards selected) by my per-card bid:')
bk=collections.defaultdict(list)
for x in allb:
    if x['n']==0: continue
    pc=x['tot']/x['n']; bk['<=3' if pc<=3 else '4-6' if pc<=6 else '7-12' if pc<=12 else '13-35' if pc<=35 else '36-110' if pc<=110 else '111+'].append(x)
for k in ['<=3','4-6','7-12','13-35','36-110','111+']:
    v=bk[k]
    if v: print('    per-card %-7s n=%5d  all-selected %.0f%%  frac of my cards on flop %.2f  opp paid mean %.0f  potbb %.1f'%(k,len(v),100*sum(x['allgot'] for x in v)/len(v),st.mean(x['got']/x['n'] for x in v),st.mean(x['opp'] for x in v),st.mean(x['potbb'] for x in v)))
print(' my bid vs pot: chips bid per bb of pot: q',[round(z,2) for z in q([x['tot']/x['potbb'] for x in allb])],' opp paid per bb of pot q',[round(z,2) for z in q([x['opp']/x['potbb'] for x in allb])])
print(' bid as fraction of my remaining flop chips: q',[round(z,3) for z in q([x['tot']/max(1,x['fc']) for x in allb],(.5,.75,.9,.99))])
