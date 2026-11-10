import sys,os,collections,statistics as st,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import hp
R=hp.all_tables(); me=R['me']; nt=len(R['tabs'])
pay=collections.defaultdict(list)
for t in R['tabs']:
    fin=t['hands'][-1]['eq1']
    for s,e in enumerate(t['seats']): pay[e].append(fin[s])
avg={e:st.mean(v) for e,v in pay.items()}
# 1 bust analysis
B=collections.Counter(); Bbb=collections.defaultdict(list); stage=collections.Counter()
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        if h['sb4'][s]>0 and h['sa'][s]==0:
            mine=[a for a in h['acts'] if a['seat']==s]; pre=[a for a in mine if a['street']==0]; post=[a for a in mine if a['street']>0]
            bb=h['sb4'][s]/h['bb']
            if post and not any(a.get('allin') for a in pre): how='postflop stack-off'
            elif pre and pre[-1]['kind']=='raise' and pre[-1].get('allin'):
                how='pre: I jammed first-in' if pre[-1]['nraise']==0 else 'pre: I re-jammed/3b-jam'
            elif pre and pre[-1]['kind']=='call': how='pre: I called all-in'
            else: how='blinded/forced'
            B[how]+=1; Bbb[how].append(bb); stage['n=%d'%h['n']]+=1
tot=sum(B.values()); print('busts',tot)
for k,v in B.most_common(): print('  %-28s %4d (%.0f%%)  median stack %.1fbb  stack<5bb: %.0f%%'%(k,v,100*v/tot,st.median(Bbb[k]),100*sum(1 for x in Bbb[k] if x<5)/v))
print('  by players left',sorted(stage.items()))
# stack trajectory: my avg stack in bb & share when alive at hand N vs field
print('\nmy stack (bb) when alive, at start of each level; % alive; and blinds folded cost')
for hn in (1,21,41,61,81,101,121,141,161):
    v=[];al=0;fa=[]
    for t in R['tabs']:
        if len(t['hands'])>=hn:
            h=t['hands'][hn-1]; s=t['me']
            if h['sb4'][s]>0: v.append(h['sb4'][s]/h['bb']); al+=1
            fa.append(sum(1 for x in h['sb4'] if x>0))
    if v: print('  hand %3d: alive %.0f%% (field avg alive %.0f%%)  my median stack %.1fbb  mean %.1fbb  <10bb: %.0f%%'%(hn,100*al/len(fa),100*st.mean(fa)/8,st.median(v),st.mean(v),100*sum(1 for x in v if x<10)/len(v)))
# 4 time trend
print('\ntrend by iteration block: my payout, mean opp strength, my unspent')
for i in range(0,1001,200):
    tb=[t for t in R['tabs'] if i<=t['it']<i+200]
    if tb: print('  it %4d-%4d n=%d payout %.2f  opp strength %.2f'%(i,i+199,len(tb),st.mean(t['hands'][-1]['eq1'][t['me']] for t in tb),st.mean(avg[e] for t in tb for e in t['seats'] if e!=me)))
# 5 turn call -> river
tc=collections.Counter(); res=collections.defaultdict(list)
rv=collections.Counter(); rres=collections.defaultdict(list)
for t in R['tabs']:
    s=t['me']
    for h in t['hands']:
        mine=[a for a in h['acts'] if a['seat']==s]
        tcalls=[a for a in mine if a['street']==2 and a['kind']=='call' and not a.get('allin')]
        bbw=(h['sa'][s]-h['sb4'][s])/h['bb']
        if tcalls:
            a=tcalls[-1]; sz='small' if a['to_call']<=0.4*(a['pot']-a['to_call']) else 'big'
            riv=[x for x in mine if x['street']==3]
            if not riv: out='no river action'
            else:
                facing=[x for x in riv if x['cur']>0]
                if facing: out='river: faced bet -> '+facing[-1]['kind']
                else: out='river: checked through' if all(x['kind']=='check' for x in riv) else 'river: I bet'
            tc[(sz,out)]+=1; res[(sz,out)].append(bbw); res[(sz,'ALL')].append(bbw); tc[(sz,'ALL')]+=1
        # river as turn aggressor
        if h['street_agg'][2]==s and len(h['board'])==5:
            riv=[x for x in mine if x['street']==3]
            if riv and riv[0]['cur']==0:
                k='bet' if riv[0]['kind']=='raise' else 'check'
                won=h['wins'].get(s,0)>0; sd=s in h['shows']
                rv[k]+=1; rres[k].append((bbw,won,sd))
print('\nafter I call a turn bet (not all-in): what happens, whole-hand bb result')
for sz in ('small','big'):
    for k,v in sorted(tc.items()):
        if k[0]==sz: print('  %-5s %-34s n=%4d (%.0f%%)  bb/hand %+.2f'%(sz,k[1],v,100*v/tc[(sz,'ALL')],st.mean(res[k])))
print('\nriver, I was the turn bettor and act with no bet in front:')
for k in ('bet','check'):
    v=rres[k]; sd=[x for x in v if x[2]]
    print('  %-5s n=%4d  won pot %.0f%%  reached showdown %.0f%%  won at showdown %.0f%%  bb/hand %+.2f'%(k,len(v),100*st.mean(x[1] for x in v),100*len(sd)/len(v),100*st.mean(x[1] for x in sd) if sd else 0,st.mean(x[0] for x in v)))
# blinds: cost of folding in blinds per table by stage
# 6 fold-to-steal exploit: field BB fold vs my steals; EV of min-raise any two from BTN/SB/CO given observed fold rates
st_c=collections.defaultdict(lambda:[0,0])
for t in R['tabs']:
    for h in t['hands']:
        pre=[a for a in h['acts'] if a['street']==0]
        for i,a in enumerate(pre):
            if a['kind']=='fold': continue
            if a['kind']=='raise' and not a.get('allin') and a['nraise']==0 and a['limpers']==0 and a['pos'] in('CO','BTN','SB') and t['seats'][a['seat']]!=me:
                rest=pre[i+1:]; nxt=[b for b in rest if b['seat']!=a['seat']]
                # all fold?
                allfold=all(b['kind']=='fold' for b in nxt[:{'CO':3,'BTN':2,'SB':1}[a['pos']]]) and len(nxt)>={'CO':3,'BTN':2,'SB':1}[a['pos']]
                effb='<20' if a['eff']/a['bb']<20 else '20-40' if a['eff']/a['bb']<40 else '40+'
                st_c[(a['pos'],effb)][0]+=allfold; st_c[(a['pos'],effb)][1]+=1; st_c[(a['pos'],'size')][0]+=a['to']/a['bb']; st_c[(a['pos'],'size')][1]+=1
            break
print('\nfield non-all-in steals: P(everyone folds) and breakeven (raise to x bb needs x-ish/(x+1.5))')
for p in ('CO','BTN','SB'):
    sz=st_c[(p,'size')][0]/st_c[(p,'size')][1]
    print('  %-3s avg open %.2fbb (bluff breakeven fold%% = %.0f%%): '%(p,sz,100*(sz-(0.5 if p=='SB' else 0))/(sz+1.5-(0.5 if p=='SB' else 0)))+'  '.join('%s: %.0f%% (n=%d)'%(e,100*st_c[(p,e)][0]/max(1,st_c[(p,e)][1]),st_c[(p,e)][1]) for e in('<20','20-40','40+')))
