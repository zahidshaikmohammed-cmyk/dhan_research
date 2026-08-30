from pathlib import Path
import numpy as np,pandas as pd,json
OUT=Path('output'); OUT.mkdir(exist_ok=True)
SYMS=['360ONE','MPHASIS','HCLTECH','POLYMED','JSL','AFFLE','SONACOMS','VEDL','NATIONALUM','JKCEMENT','ZENSARTECH','KEI','TATATECH','LTM','PVRINOX','UNOMINDA','ENDURANCE','RAINBOW','SHREECEM','APTUS','MOTILALOFS','ESCORTS','NATCOPHARM','SPLPETRO','PERSISTENT','TRAVELFOOD','UBL','INDIACEM','BHARTIHEXA']
E=pd.read_parquet(OUT/'ENTRY_EVENT_DATA.parquet'); E.trade_date=pd.to_datetime(E.trade_date); E['day']=E.trade_date.dt.normalize(); E['entry']=E.entry.astype(str).str[:5]
assert set(E.symbol)==set(SYMS),f'Universe mismatch: {set(SYMS)-set(E.symbol)}'
days=np.array(sorted(E.day.unique())); blocks=np.array_split(days,5); test_days=set(blocks[-1])
prims=['mom_3','mom_5','mom_10','mom_15','eff_5','eff_10','eff_20','r2_10','r2_20','slope_10','slope_20','relvol_5_20','dist_vwap_pct','open_range_5','open_range_10']
results=[]
for sym in SYMS:
 for entry,g in E[E.symbol==sym].groupby('entry'):
  train=g[~g.day.isin(test_days)]; val=train[train.day.isin(set(blocks[-2]))]
  if len(train)<100 or len(val)<30: continue
  for side in ('LONG','SHORT'):
   direction=(val.mom_10>0) if side=='LONG' else (val.mom_10<0)
   for feat in prims:
    vals=train[feat].dropna().abs()
    if len(vals)<50: continue
    for q in (.60,.70,.80,.90):
     th=float(vals.quantile(q))
     if not np.isfinite(th) or th<=0: continue
     m=direction&(val[feat].abs()>=th); y=(val.loc[m,'fwd_60'] if side=='LONG' else -val.loc[m,'fwd_60']).dropna()
     if len(y)<20: continue
     results.append(dict(symbol=sym,entry=entry,side=side,feature=feat,threshold=th,q=q,val_n=len(y),val_win=float((y>0).mean()),val_mean=float(y.mean()),val_median=float(y.median()),score=float(y.median()*((y>0).mean()-.5)*np.sqrt(len(y)))))
R=pd.DataFrame(results)
if R.empty: raise RuntimeError('No evaluable causal trigger candidates')
R.to_csv(OUT/'V4_TRIGGER_CANDIDATES.csv',index=False)
B=R.sort_values(['score','val_median','val_mean'],ascending=False).groupby(['symbol','side','entry'],as_index=False).head(1); B=B[(B.val_median>0)&(B.val_mean>0)&(B.val_win>=.50)].copy()
out=[]
for _,r in B.iterrows():
 x=E[(E.symbol==r.symbol)&(E.entry==r.entry)&E.day.isin(test_days)]; direction=(x.mom_10>0) if r.side=='LONG' else (x.mom_10<0); m=direction&(x[r.feature].abs()>=r.threshold); y=(x.loc[m,'fwd_60'] if r.side=='LONG' else -x.loc[m,'fwd_60']).dropna(); d=r.to_dict(); d.update(test_n=len(y),test_win=float((y>0).mean()) if len(y) else np.nan,test_mean=float(y.mean()) if len(y) else np.nan,test_median=float(y.median()) if len(y) else np.nan,test_positive=bool(len(y)>=20 and y.mean()>0 and y.median()>0)); out.append(d)
O=pd.DataFrame(out); O.to_csv(OUT/'V4_TRIGGER_OOS.csv',index=False); BEST=O.sort_values(['score','val_median'],ascending=False).groupby('symbol',as_index=False).head(1).sort_values('symbol'); BEST.to_csv(OUT/'V4_BEST_TRIGGER_PER_STOCK.csv',index=False); SURV=BEST[BEST.test_positive].copy(); SURV.to_csv(OUT/'V4_OOS_SURVIVORS.csv',index=False)
summary={'status':'COMPLETE','version':'V4_CAUSAL_TRIGGER_DISCOVERY','stocks':len(SYMS),'best_trigger_rows':len(BEST),'oos_survivors':len(SURV),'long':int((BEST.side=='LONG').sum()),'short':int((BEST.side=='SHORT').sum()),'test_block_days':len(test_days),'lookahead':'Features are causal; forward returns are labels only.','stops_targets_costs':'excluded from entry discovery'}
(OUT/'V4_PROTOCOL.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2)); print(BEST.to_string(index=False))
