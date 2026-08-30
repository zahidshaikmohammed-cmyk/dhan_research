from pathlib import Path
import pandas as pd,numpy as np,json
O=Path('output'); E=pd.read_parquet(O/'ENTRY_EVENT_DATA.parquet'); E.trade_date=pd.to_datetime(E.trade_date); E['d']=E.trade_date.dt.normalize(); days=np.array(sorted(E.d.unique())); trd=set(days[:int(.60*len(days))]); vad=set(days[int(.60*len(days)):int(.80*len(days))]); ted=set(days[int(.80*len(days)):])
features=['mom_3','mom_5','mom_10','mom_15','eff_5','eff_10','r2_10','r2_20','relvol_5_20','dist_vwap_pct','open_range_5','open_range_10']; rows=[]
for (s,e),g in E.groupby(['symbol','entry']):
 tr=g[g.d.isin(trd)]; va=g[g.d.isin(vad)]
 for side in ['LONG','SHORT']:
  for f in features:
   qv=tr[f].abs().dropna()
   for q in [.60,.70,.80,.90]:
    if len(qv)<40: continue
    th=float(qv.quantile(q)); direction=(va.mom_10>0) if side=='LONG' else (va.mom_10<0); m=direction&(va[f].abs()>=th); y=(va.loc[m,'fwd_60'] if side=='LONG' else -va.loc[m,'fwd_60']).dropna()
    if len(y)>=20: rows.append([s,e,side,f,q,th,len(y),float((y>0).mean()),float(y.mean()),float(y.median())])
R=pd.DataFrame(rows,columns=['symbol','entry','side','feature','q','threshold','n','win','mean','median']); R['score']=R['median']*(R.win-.5)*np.sqrt(R.n); R.to_csv(O/'TRIGGER_A_CANDIDATES.csv',index=False); B=R.sort_values('score',ascending=False).groupby(['symbol','side','entry'],as_index=False).head(1); out=[]
for _,r in B.iterrows():
 te=E[(E.symbol==r.symbol)&(E.entry==r.entry)&E.d.isin(ted)]; direction=(te.mom_10>0) if r.side=='LONG' else (te.mom_10<0); m=direction&(te[r.feature].abs()>=r.threshold); y=(te.loc[m,'fwd_60'] if r.side=='LONG' else -te.loc[m,'fwd_60']).dropna(); d=r.to_dict(); d.update(test_n=len(y),test_win=(y>0).mean() if len(y) else np.nan,test_mean=y.mean() if len(y) else np.nan,test_median=y.median() if len(y) else np.nan); out.append(d)
Z=pd.DataFrame(out); Z.to_csv(O/'TRIGGER_A_BEST.csv',index=False); (O/'TRIGGER_A_PROTOCOL.json').write_text(json.dumps({'method':'single-factor causal quantile; validation selection; untouched test','rows':len(Z)},indent=2)); print(Z.sort_values('score',ascending=False).head(50).to_string(index=False))
