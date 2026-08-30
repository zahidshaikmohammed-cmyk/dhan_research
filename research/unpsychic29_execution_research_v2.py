from pathlib import Path
import pandas as pd,numpy as np,json
OUT=Path('output'); OUT.mkdir(exist_ok=True)
SYMS=['MPHASIS','HCLTECH','POLYMED','JSL','AFFLE','SONACOMS','VEDL','KEI','TATATECH','LTM','UNOMINDA','SHREECEM','APTUS','MOTILALOFS','NATCOPHARM','PERSISTENT']
ENTRY=['09:20','09:21','09:22','09:23','09:24','09:25','09:26','09:27','09:28','09:29','09:30','09:35','09:40','09:45','09:50','10:00']; EM={e:int(e[:2])*60+int(e[3:]) for e in ENTRY}
F=['mom_3','mom_5','mom_10','mom_15','eff_5','eff_10','r2_10','r2_20','relvol_5_20','dist_vwap_pct','open_range_5','open_range_10']
def eff(v):
 v=np.asarray(v,float)
 if len(v)<2:return np.nan
 p=np.abs(np.diff(v)/v[:-1]).sum(); return abs(v[-1]/v[0]-1)/p if p else 0.0
def r2(v):
 v=np.asarray(v,float)
 if len(v)<3 or np.any(v<=0):return np.nan
 x=np.arange(len(v)); y=np.log(v); b=np.polyfit(x,y,1); fit=np.polyval(b,x); den=((y-y.mean())**2).sum(); return 1-((y-fit)**2).sum()/max(den,1e-12)
def load(sym):
 ps=sorted(Path('data').rglob(f'raw_1min__{sym}.parquet.chunk*.csv')); assert ps,f'missing {sym}'
 d=pd.concat([pd.read_csv(p) for p in ps],ignore_index=True); d['ts']=pd.to_datetime(d.timestamp,utc=True).dt.tz_convert('Asia/Kolkata'); d=d.sort_values('ts').drop_duplicates('ts'); d['day']=d.ts.dt.date; d['minute']=d.ts.dt.hour*60+d.ts.dt.minute; return d.reset_index(drop=True)
def feat(z,i):
 p=float(z.close.iloc[i]); f={}
 for n in [3,5,10,15]: f[f'mom_{n}']=100*(p/float(z.close.iloc[i-n])-1) if i>=n else np.nan
 for n in [5,10]:
  j=max(0,i-n+1); f[f'eff_{n}']=eff(z.close.iloc[j:i+1]); f[f'r2_{n}']=r2(z.close.iloc[j:i+1])
 v5=z.volume.iloc[max(0,i-4):i+1].mean(); v20=z.volume.iloc[max(0,i-19):i+1].mean(); f['relvol_5_20']=v5/v20 if v20>0 else np.nan
 vv=z.volume.iloc[:i+1]; vw=(((z.high+z.low+z.close)/3)*z.volume).iloc[:i+1].sum()/vv.sum() if vv.sum()>0 else np.nan; f['dist_vwap_pct']=100*(p/vw-1) if pd.notna(vw) else np.nan
 op=float(z.open.iloc[0]); f['open_range_5']=100*(z.high.iloc[:min(i+1,5)].max()-z.low.iloc[:min(i+1,5)].min())/op; f['open_range_10']=100*(z.high.iloc[:min(i+1,10)].max()-z.low.iloc[:min(i+1,10)].min())/op; return f
E=[]; raw={}
for sym in SYMS:
 d=load(sym); raw[sym]=d
 for day,g0 in d.groupby('day',sort=True):
  z=g0[(g0.minute>=555)&(g0.minute<=930)].sort_values('ts').reset_index(drop=True); idx={m:i for i,m in enumerate(z.minute)}
  if 555 not in idx or not all(m in idx for m in EM.values()): continue
  for e,m in EM.items():
   i=idx[m]; f=feat(z,i); f.update(symbol=sym,day=pd.Timestamp(day),entry=e,signal_i=i,fwd60=100*(float(z.close.iloc[i+60])/float(z.close.iloc[i])-1) if i+60<len(z) else np.nan); E.append(f)
E=pd.DataFrame(E); days=np.array(sorted(E.day.unique())); cut1=days[int(.60*len(days))]; cut2=days[int(.80*len(days))]; E['split']=np.where(E.day<cut1,'TRAIN',np.where(E.day<cut2,'VALIDATION','TEST'))
# Exact Trigger-C discovery algorithm reproduced from repository source.
C=[]
for (sym,e),g in E.groupby(['symbol','entry']):
 tr=g[g.split=='TRAIN']; va=g[g.split=='VALIDATION']
 for side in ['LONG','SHORT']:
  sign=1 if side=='LONG' else -1; cols=[]
  for ff in F:
   q=tr[[ff,'fwd60']].dropna(); q['y']=sign*q.fwd60
   if len(q)>=50:
    c=q[ff].corr(q.y)
    if np.isfinite(c) and c>0: cols.append((ff,float(c)))
  if len(cols)<2: continue
  w=dict(sorted(cols,key=lambda x:x[1],reverse=True)[:5])
  def score(df):
   s=pd.Series(0.0,index=df.index)
   for ff,c in w.items(): s=s+c*np.sign(df[ff])*df[ff].abs()/max(float(tr[ff].abs().quantile(.75)),1e-9)
   return s
  th=float(score(tr).quantile(.75)); y=(va.loc[score(va)>=th,'fwd60']*sign).dropna()
  if len(y)>=20:C.append({'symbol':sym,'entry':e,'side':side,'score_threshold':th,'n':len(y),'win':float((y>0).mean()),'mean':float(y.mean()),'median':float(y.median()),'weights':json.dumps(w),'score':float(y.median()*((y>0).mean()-.5)*np.sqrt(len(y)))})
C=pd.DataFrame(C); C.to_csv(OUT/'EXECUTION_TRIGGER_C_REBUILT_V2.csv',index=False)
B=C.sort_values('score',ascending=False).groupby(['symbol','side','entry'],as_index=False).head(1)
# Event-driven execution: signal at completed bar; fill next bar open. Stop = opposite side of causal 5/10-minute opening range. Target = 1R or 2R. Same-bar stop+target => stop first.
rows=[]
for _,r in B.iterrows():
 d=raw[r.symbol]; side=r.side; sign=1 if side=='LONG' else -1; e=r.entry; m=EM[e]; w=json.loads(r.weights); trsc=E[(E.symbol==r.symbol)&(E.split=='TRAIN')]
 for orb in [5,10]:
  for rr in [1.0,2.0]:
   for day,g0 in d.groupby('day',sort=True):
    z=g0[(g0.minute>=555)&(g0.minute<=930)].sort_values('ts').reset_index(drop=True); idx={x:i for i,x in enumerate(z.minute)}
    if m not in idx: continue
    i=idx[m]; f=feat(z,i); sc=0
    for ff,c in w.items(): sc+=c*np.sign(f[ff])*abs(f[ff])/max(float(trsc[ff].abs().quantile(.75)),1e-9)
    if sc<r.score_threshold or i+1>=len(z): continue
    entry=float(z.open.iloc[i+1]); hi0=float(z.high.iloc[:min(i+1,orb)].max()); lo0=float(z.low.iloc[:min(i+1,orb)].min()); stop=lo0 if side=='LONG' else hi0; risk=abs(100*(stop/entry-1)); target=entry*(1+sign*rr*risk/100); end=min(len(z)-1,i+61); exitp=float(z.close.iloc[end]); reason='TIME'
    for j in range(i+1,end+1):
     hi=float(z.high.iloc[j]); lo=float(z.low.iloc[j]); hs=(lo<=stop) if side=='LONG' else (hi>=stop); ht=(hi>=target) if side=='LONG' else (lo<=target)
     if hs: exitp=stop; reason='STOP'; break
     if ht: exitp=target; reason='TARGET'; break
    gross=sign*100*(exitp/entry-1); rows.append({'symbol':r.symbol,'entry':e,'side':side,'orb':orb,'rr':rr,'day':pd.Timestamp(day),'split':'TRAIN' if day<cut1 else 'VALIDATION' if day<cut2 else 'TEST','gross_pct':gross,'entry_price':entry,'stop_price':stop,'target_price':target,'risk_pct':risk,'reason':reason,'score':sc})
T=pd.DataFrame(rows); T.to_csv(OUT/'EXECUTION_TRADES_V2.csv',index=False)
S=[]
for keys,g in T.groupby(['symbol','entry','side']):
 v=g[g.split=='VALIDATION']; t=g[g.split=='TEST']; opts=[]
 for (orb,rr),x in v.groupby(['orb','rr']):
  if len(x)>=10 and x.gross_pct.mean()>0: opts.append((x.gross_pct.mean(),x.gross_pct.median(),orb,rr))
 if not opts: continue
 _,_,orb,rr=max(opts); tt=t[(t.orb==orb)&(t.rr==rr)]
 if tt.empty: continue
 testmean=float(tt.gross_pct.mean()); testwin=float((tt.gross_pct>0).mean()); pf=float(tt.loc[tt.gross_pct>0,'gross_pct'].sum()/abs(tt.loc[tt.gross_pct<0,'gross_pct'].sum())) if (tt.gross_pct<0).any() else np.inf
 S.append({'symbol':keys[0],'entry':keys[1],'side':keys[2],'orb_stop':orb,'target_R':rr,'validation_trades':int(len(v[(v.orb==orb)&(v.rr==rr)])),'validation_mean':float(v[(v.orb==orb)&(v.rr==rr)].gross_pct.mean()),'test_trades':int(len(tt)),'test_win':testwin,'test_mean':testmean,'test_median':float(tt.gross_pct.median()),'test_profit_factor':pf,'break_even_roundtrip_bps':max(0,testmean*100)})
S=pd.DataFrame(S); S.to_csv(OUT/'EXECUTION_SELECTED_RULES_V2.csv',index=False)
summary={'status':'COMPLETE','candidate_definitions':len(C),'selected_execution_rules':len(S),'entry_execution':'next 1-minute OPEN after completed signal bar','stop_variants':'opposite 5-minute or 10-minute opening-range boundary','targets':'1R/2R selected on validation only','same_bar_ambiguity':'STOP FIRST','costs':'gross only plus exact break-even round-trip bps; no broker fee fabricated','test_usage':'test is never used to select the execution variant','raw_universe':'16 stocks represented by Trigger-C artifact'}
(OUT/'EXECUTION_RESEARCH_PROTOCOL_V2.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(S.to_string(index=False))
