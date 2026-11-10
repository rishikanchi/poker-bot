import json,pickle,os,sys
SC=os.path.dirname(os.path.abspath(__file__)); PK=os.path.join(SC,'tabs.pkl')
def load(path=None):
    if os.path.exists(PK): return pickle.load(open(PK,'rb'))
    f=open(path); hdr=json.loads(f.readline()); tabs=[]
    while True:
        l=f.readline()
        if not l: break
        t=json.loads(l); y=json.loads(f.readline())['you']; t['you']=y; tabs.append(t)
    names={e['entry_id']:e['team_name'] for e in hdr['entries']}
    d=dict(me=hdr['you'],names=names,tabs=tabs); pickle.dump(d,open(PK,'wb'),-1); return d
