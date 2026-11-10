"""Parse hands into per-action records."""
import os,sys,pickle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import load
POSN={2:['SB','BB'],3:['BTN','SB','BB'],4:['CO','BTN','SB','BB'],5:['HJ','CO','BTN','SB','BB'],6:['LJ','HJ','CO','BTN','SB','BB'],
      7:['UTG1','LJ','HJ','CO','BTN','SB','BB'],8:['UTG','UTG1','LJ','HJ','CO','BTN','SB','BB']}
def parse_hand(t,h):
    sb4=h['stacks_before']; alive=[s for s in range(8) if sb4[s]>0]; n=len(alive)
    ev=h['events']; bl=[e for e in ev if e['kind']=='blind']
    bbv=h['blinds'][1]
    bbseat=max(bl,key=lambda e:e['amount'])['seat'] if bl else None
    sbseat=None
    if len(bl)>1: sbseat=min(bl,key=lambda e:e['amount'])['seat']
    if len(bl)==2 and bl[0]['amount']==bl[1]['amount']: sbseat,bbseat=bl[0]['seat'],bl[1]['seat']
    # order: seats after BB in turn order
    order=[]
    if bbseat is not None:
        i=alive.index(bbseat); order=alive[i+1:]+alive[:i+1]  # first to act preflop ... BB last
    pos={}
    if n>=2 and len(order)==n:
        labels=POSN[n]
        if n==2: pos={order[0]:'SB',order[1]:'BB'}
        else:
            for s,l in zip(order,labels): pos[s]=l
    stack={s:sb4[s] for s in alive}; put={s:0 for s in alive}; sbet={s:0 for s in alive}
    infold=set(); allin=set(); street=0; pot=0; cur=0; acts=[]; nraise=0; last_agg=None; pfr=None
    limpers=0; callers=0; board=[]; auction=None; bids=[]; shows={}; wins={}
    street_agg={0:None,1:None,2:None,3:None}; saw={0:set(alive)}; nbet_street=0
    for i,e in enumerate(ev):
        k=e['kind']
        if k=='blind':
            s=e['seat']; a=e['amount']; stack[s]-=a; put[s]+=a; sbet[s]+=a; pot+=a; cur=max(cur,a)
            if stack[s]==0: allin.add(s)
        elif k in('fold','check','call','raise'):
            s=e['seat']; to_call=cur-sbet[s]; live=[x for x in alive if x not in infold]
            rec=dict(i=i,seat=s,street=street,kind=k,to_call=min(to_call,stack[s]),pot=pot,stack=stack[s],nlive=len(live),
                     nraise=nraise,limpers=limpers,callers=callers,cur=cur,agg=last_agg,pos=pos.get(s),bb=bbv,
                     eff=min(stack[s]+sbet[s], max([stack[x]+sbet[x] for x in live if x!=s],default=0)),
                     nact=len([x for x in live if x not in allin]))
            if k=='fold': infold.add(s)
            elif k=='call':
                a=e['amount']; stack[s]-=a; put[s]+=a; sbet[s]+=a; pot+=a; rec['amt']=a
                if street==0:
                    if nraise==0: limpers+=1
                    else: callers+=1
            elif k=='raise':
                a=e['amount']; add=a-sbet[s]; rec['amt']=add; rec['to']=a; rec['prev']=cur
                stack[s]-=add; put[s]+=add; sbet[s]=a; pot+=add; cur=a; nraise+=1; last_agg=s; callers=0
                if street==0: pfr=s
                street_agg[street]=s
            if stack[s]==0 and k in('call','raise'): allin.add(s); rec['allin']=True
            acts.append(rec)
        elif k=='auction': auction=e['cards']; saw['auc']=set(x for x in alive if x not in infold)
        elif k=='bid': bids.append(e['seat'])
        elif k=='board':
            board+=e['cards']; street={3:1,4:2,5:3}[len(board)]; sbet={s:0 for s in alive}; cur=0; nraise=0; last_agg=None
            saw[street]=set(x for x in alive if x not in infold)
        elif k=='show': shows[e['seat']]=e['cards']
        elif k=='win': wins[e['seat']]=wins.get(e['seat'],0)+e['amount']
    return dict(hand=h['hand'],n=n,alive=alive,pos=pos,bb=bbv,sb=h['blinds'][0],acts=acts,board=board,auction=auction,bids=bids,shows=shows,wins=wins,
                put=put,pfr=pfr,saw=saw,sb4=sb4,sa=h['stacks_after'],eq0=h['equity_before'],eq1=h['equity_after'],
                fc0=h['flop_chips_before'],fc1=h['flop_chips_after'],infold=infold,allin=allin,button=h['button'],cards=h['cards'],street_agg=street_agg)
def all_tables():
    PK=os.path.join(os.path.dirname(os.path.abspath(__file__)),'parsed.pkl')
    if os.path.exists(PK): return pickle.load(open(PK,'rb'))
    D=load(sys.argv[1] if len(sys.argv)>1 else None); out=[]
    for t in D['tabs']:
        y=t['you']; hs=[parse_hand(t,h) for h in t['hands']]
        priv={}
        for e in y['private_events']: priv.setdefault(e['hand'],[]).append(e)
        out.append(dict(it=t['iteration'],seats=t['seats'],me=y['seat'],mycards=y['cards'],priv=priv,hands=hs,auto=y['automatic']))
    R=dict(me=D['me'],names=D['names'],tabs=out); pickle.dump(R,open(PK,'wb'),-1); return R
if __name__=='__main__':
    R=all_tables(); 
    # sanity: chip conservation
    bad=0;tot=0
    for t in R['tabs']:
        for h in t['hands']:
            tot+=1
            for s in h['alive']:
                if h['sb4'][s]-h['put'][s]+h['wins'].get(s,0)!=h['sa'][s]: bad+=1; break
    print('hands',tot,'chip mismatch',bad)
    import collections
    print(collections.Counter(h['n'] for t in R['tabs'] for h in t['hands']))
    print(collections.Counter(len(h['pos']) == h['n'] for t in R['tabs'] for h in t['hands']))
