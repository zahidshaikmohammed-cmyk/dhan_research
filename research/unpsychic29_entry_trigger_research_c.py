from pathlib import Path
import pandas as pd,numpy as np,json
O=Path('output'); E=pd.read_parquet(O/'ENTRY_EVENT_DATA.parquet'); E.trade_date=pd.to_datetime(E.trade_date); E['d']=E.trade_date.dt.normalize(); days=np.array(sorted(E.d.unique())); trd=set(days[:int(.60*len(days))]); vad=set(days[int(.60*len(days)):int(.80*len(days))]); ted=set(days[int(.80*len(days)):])
F=['mom_3','mom_5','mom_10','mom_15','eff_5','eff_10','r2_10','r2_20','relvol_5_20','dist_vwap_pct','open_range_5','open_range_10']; rows=[]
for (s,e),g in E.groupby(['symbol','entry']):
 tr=g[g.d.isin(trd)]; va=g[g.d.isin(vad)]
 for side in ['LONG','SHORT']:
  sign=1 if side=='LONG' else -1
  for f in F:
   x=tr[[f,'fwd_60']].dropna(); x['y']=sign*x.fwd_60
   if len(x)<50: continue
   corr=x[f].corr(x.y)
   if not np.isfinite(corr): continue
   # Directional score combines several causal features whose TRAIN correlation agrees with the side.
   score_cols=[]
   for ff in F:
    z=tr[[ff,'fwd_60']].dropna(); z['y']=sign*z.fwd_60
    c=z[ff].corr(z.y) if len(z)>=50 else np.nan
    if np.isfinite(c) and c>0: score_cols.append((ff,float(c)))
   if len(score_cols)<2: continue
   weights={ff:c for ff,c in sorted(score_cols,key=lambda q:q[1],reverse=True)[:5]}
   def score(df):
    vals=[]
    for ff,c in weights.items():
     med=float(tr[ff].abs().median()); mad=float(tr[ff].abs().quantile(.75)); scale=max(mad,1e-9); vals.append(c*np.sign(df[ff])*df[ff].abs()/scale)
    return pd.concat(vals,axis=1).sum(axis=1)
   vs=score(va); threshold=float(np.quantile(score(tr),.75)); m=vs>=threshold; y=(va.loc[m,'fwd_60']*sign).dropna()
   if len(y)>=20: rows.append([s,e,side,threshold,len(y),float((y>0).mean()),float(y.mean()),float(y.median()),json.dumps(weights)])
R=pd.DataFrame(rows,columns=['symbol','entry','side','score_threshold','n','win','mean','median','weights']); R['score']=R['median']*(R.win-.5)*np.sqrt(R.n); R.to_csv(O/'TRIGGER_C_CANDIDATES.csv',index=False); B=R.sort_values('score',ascending=False).groupby(['symbol','side','entry'],as_index=False).head(1); out=[]
for _,r in B.iterrows():
 te=E[(E.symbol==r.symbol)&(E.entry==r.entry)&E.d.isin(ted)]; w=json.loads(r.weights); sc=0
 for ff,c in w.items(): sc+=c*np.sign(te[ff])*te[ff].abs()/max(float(trd and E[E.d.isin(trd)][ff].abs().quantile(.75)),1e-9)
 m=sc>=r.score_threshold; y=(te.loc[m,'fwd_60']*(1 if r.side=='LONG' else -1)).dropna(); d=r.to_dict(); d.update(test_n=len(y),test_win=(y>0).mean() if len(y) else np.nan,test_mean=y.mean() if len(y) else np.nan,test_median=y.median() if len(y) else np.nan); out.append(d)
Z=pd.DataFrame(out); Z.to_csv(O/'TRIGGER_C_BEST.csv',index=False); (O/'TRIGGER_C_PROTOCOL.json').write_text(json.dumps({'method':'correlation-weighted causal multi-feature score; train weights and scale; validation threshold; untouched test','rows':len(Z)},indent=2)); print(Z.sort_values('score',ascending=False).head(50).to_string(index=False))
