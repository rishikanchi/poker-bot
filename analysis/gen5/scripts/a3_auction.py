import sys,os,collections,statistics as st,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp
R=hp.all_tables(); me=R['me']; names=R['names']
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
rank=sorted((e for e in avg if e!=me),key=lambda e:-avg[e]); top=set(rank[:30]); bot=set(rank[-60:])
def grp(e): return 'ME' if e==me else 'TOP30' if e in top else 'BOT60' if e in bot else 'MID'
G=['ME','TOP30','MID','BOT60']
def q(v,qs=(.1,.25,.5,.75,.9)):
    v=sorted(v); return [v[min(len(v)-1,int(x*len(v)))] for x in qs] if v else []
# 1. rule check with my private events
ok=bad=0; tie_first=0; n_auc=0; default_flop=0; n_bidders=collections.Counter()
for t in R['tabs']:
    for h in t['hands']:
        if not h['auction']: continue
        n_auc+=1; n_bidders[len(h['bids'])]+=1
        flop=h['board'][:3]
        if flop==h['auction'][:3]: default_flop+=1
        pe=t['priv'].get(h['hand'],[])
        mb={e['cards'][0]:e['amount'] for e in pe if e['kind']=='bid'}
        paid=sum(e['amount'] for e in pe if e['kind']=='flop_chips_paid')
        if mb:
            exp=sum(a for c,a in mb.items() if c in flop)
            if exp==paid and paid==h['fc0'][t['me']]-h['fc1'][t['me']]: ok+=1
            else:
                bad+=1
                if bad<4: print('BAD',mb,flop,paid,h['fc0'][t['me']]-h['fc1'][t['me']])
print('auctions',n_auc,'rule ok',ok,'bad',bad,'flop==first3 %.1f%%'%(100*default_flop/n_auc),'bidders',sorted(n_bidders.items()))
# is flop order = auction order subset?
# 2. total paid per auction & relation to pot
rows=[]
for t in R['tabs']:
    for h in t['hands']:
        if not h['auction']: continue
        paid={s:h['fc0'][s]-h['fc1'][s] for s in h['bids']}
        pot=sum(h['put'][s] for s in h['alive'])  # final; need pot at auction
        potA=0
        # pot at auction = sum of preflop money
        pre=sum(a.get('amt',0) for a in h['acts'] if a['street']==0)+sum(0 for _ in [])
        rows.append((t,h,paid))
tot=[sum(p.values()) for t,h,p in rows]
print('total paid per auction quantiles',q(tot,(.1,.25,.5,.75,.9,.95,.99)),'mean %.1f'%st.mean(tot),'zero%%=%.1f'%(100*sum(1 for x in tot if x==0)/len(tot)))
# per-seat paid by group
pg=collections.defaultdict(list); zg=collections.Counter(); ng=collections.Counter()
for t,h,p in rows:
    for s,v in p.items():
        g=grp(t['seats'][s]); pg[g].append(v); ng[g]+=1; zg[g]+= v==0
for g in G: print(g,'auctions',ng[g],'paid mean %.1f'%st.mean(pg[g]),'zero %.0f%%'%(100*zg[g]/ng[g]),'q',q(pg[g],(.5,.75,.9,.95,.99)))
# 3. chips left at end of table / at bust
left=collections.defaultdict(list); leftbust=collections.defaultdict(list); 
for t in R['tabs']:
    last=t['hands'][-1]
    for s,e in enumerate(t['seats']):
        g=grp(e); left[g].append(last['fc1'][s])
print('\nflop chips unspent at table end:')
for g in G: print(g,'mean %.0f'%st.mean(left[g]),'q',q(left[g]))
# by entry: unspent vs avg payout correlation
ent_left=collections.defaultdict(list); ent_paid_per=collections.defaultdict(list)
for t in R['tabs']:
    last=t['hands'][-1]
    for s,e in enumerate(t['seats']): ent_left[e].append(last['fc1'][s])
for t,h,p in rows:
    for s,v in p.items(): ent_paid_per[t['seats'][s]].append(v)
xs=[];ys=[];zs=[]
for e in avg:
    if e in ent_paid_per and len(ent_paid_per[e])>50: xs.append(st.mean(ent_left[e])); ys.append(avg[e]); zs.append(st.mean(ent_paid_per[e]))
def corr(a,b):
    ma,mb=st.mean(a),st.mean(b); return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
print('corr(unspent, avg payout) across entries %.2f ; corr(mean paid per auction, payout) %.2f'%(corr(xs,ys),corr(zs,ys)))
print('top-15 entries: payout, unspent, mean paid/auction, auctions per table')
for e in rank[:15]: print('  %-24s %.1f  unspent %.0f  paid/auc %.1f  zero%%=%.0f n=%d'%(names[e][:24],avg[e],st.mean(ent_left[e]),st.mean(ent_paid_per[e]),100*sum(1 for v in ent_paid_per[e] if v==0)/len(ent_paid_per[e]),len(ent_paid_per[e])))
print('  ME %.1f unspent %.0f paid/auc %.1f'%(avg[me],st.mean(ent_left[me]),st.mean(ent_paid_per[me])))
# 4. spending pace: my chips remaining by hand number
pace=collections.defaultdict(list); pace_top=collections.defaultdict(list)
for t in R['tabs']:
    for h in t['hands']:
        if h['hand']%20==0:
            for s,e in enumerate(t['seats']):
                if h['sb4'][s]>0:
                    (pace if e==me else pace_top if e in top else collections.defaultdict(list))[h['hand']].append(h['fc0'][s])
print('\nchips remaining (alive players) by hand#: ME vs TOP30')
for k in sorted(pace): print(k,'ME %.0f (n=%d)'%(st.mean(pace[k]),len(pace[k])),'TOP30 %.0f'%(st.mean(pace_top[k]) if pace_top[k] else -1))
# 5. paid as function of pot (bb) & n bidders; in units of chips
byp=collections.defaultdict(list)
for t,h,p in rows:
    pre=sum(a.get('amt',0) for a in h['acts'] if a['street']==0)
    blinds=h['sb']+h['bb']
    potbb=(pre+blinds)/h['bb']  # approx (blinds counted if posted)
    b='<3' if potbb<3 else '3-6' if potbb<6 else '6-12' if potbb<12 else '12-25' if potbb<25 else '25+'
    stage='h<40' if h['hand']<=40 else 'h41-80' if h['hand']<=80 else 'h81-120' if h['hand']<=120 else 'h121+'
    for s,v in p.items():
        if t['seats'][s]!=me: byp[(b,stage)].append(v); byp[(b,'all')].append(v); byp[('all',stage)].append(v)
        else: byp[('ME',b,stage)].append(v); byp[('ME',b,'all')].append(v); byp[('ME','all',stage)].append(v)
print('\nopponent paid per auction by pot(bb) x stage: mean, P(0), q50,q75,q90,q95 | ME mean')
for b in ['<3','3-6','6-12','12-25','25+','all']:
    for sg in ['h<40','h41-80','h81-120','h121+','all']:
        v=byp.get((b,sg)); m=byp.get(('ME',b,sg))
        if v and len(v)>30: print('%-6s %-8s n=%6d mean=%6.1f zero=%3.0f%% q=%s | ME n=%d mean=%.1f zero=%.0f%% q=%s'%(b,sg,len(v),st.mean(v),100*sum(1 for x in v if x==0)/len(v),q(v,(.5,.75,.9,.95)),len(m or []),st.mean(m) if m else 0,100*sum(1 for x in m if x==0)/len(m) if m else 0,q(m or [0],(.5,.75,.9,.95))))
