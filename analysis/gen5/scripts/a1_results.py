import sys,collections,statistics as st,math
sys.path.insert(0,__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from load import load
D=load(sys.argv[1]); me=D['me']; names=D['names']; tabs=D['tabs']
C=collections.Counter
pay=collections.defaultdict(list); place=collections.defaultdict(list)
myplace=[];mypay=[];bust_hand=[];bust_frac=[]
for t in tabs:
    hs=t['hands']; final=hs[-1]['equity_after']; n=8
    # bust order
    busted={}
    for h in hs:
        for s in range(8):
            if h['stacks_before'][s]>0 and h['stacks_after'][s]==0: busted[s]=(h['hand'],h['stacks_before'][s])
    order=sorted(range(8),key=lambda s:(busted.get(s,(10**9,0))[0],busted.get(s,(0,0))[1]))  # earliest bust first
    for rank,s in enumerate(order):
        pl=8-rank; e=t['seats'][s]; place[e].append(pl); pay[e].append(final[s])
        if e==me:
            myplace.append(pl); mypay.append(final[s]); bust_hand.append(busted.get(s,(len(hs),0))[0]); bust_frac.append(bust_hand[-1]/len(hs))
n=len(tabs); print('tables',n)
print('my avg payout %.3f  (field avg 12.5)  se %.3f'%(st.mean(mypay), st.pstdev(mypay)/math.sqrt(n)))
pc=C(myplace); print('place dist', {p:'%.1f%%'%(100*pc[p]/n) for p in range(1,9)})
print('ITM %.1f%% (exp 37.5)  win %.1f%% (12.5)'%(100*sum(pc[p] for p in(1,2,3))/n,100*pc[1]/n))
print('bust hand quantiles', [sorted(bust_hand)[int(q*n)] for q in (.1,.25,.5,.75,.9)])
# by tenth of sample: trend
for i in range(0,n,200): print('iters',i,i+200,'avg pay %.2f'%st.mean(mypay[i:i+200]))
# leaderboard among those with >=20 tables
rows=[(st.mean(v),len(v),names.get(e,e[:8]),e) for e,v in pay.items() if len(v)>=15]
rows.sort(reverse=True)
print('entries seen',len(pay),'with>=15',len(rows))
for i,r in enumerate(rows[:25]): 
    pl=C(place[r[3]]); print(i+1,'%.2f n=%d %s  win%%=%.0f itm%%=%.0f 8th%%=%.0f'%(r[0],r[1],r[2],100*pl[1]/r[1],100*sum(pl[p] for p in(1,2,3))/r[1],100*pl[8]/r[1]))
print('...'); 
for r in rows[-8:]: print('%.2f n=%d %s'%r[:3])
myrank=[i for i,r in enumerate(rows) if r[3]==me]; print('my rank among them',myrank, 'of',len(rows))
print('opp tables count dist', sorted(len(v) for v in pay.values())[::20])
