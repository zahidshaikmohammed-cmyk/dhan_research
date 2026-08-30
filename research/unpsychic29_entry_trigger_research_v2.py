from pathlib import Path
import json, numpy as np, pandas as pd
OUT=Path('output'); E=pd.read_parquet(OUT/'ENTRY_EVENT_DATA.parquet').copy()
ENTRY=['09:20','09:21','09:22','09:23','09:24','09:25','09:26','09:27','09:28','09:29','09:30','09:35','09:40','09:45','09:50','10:00']; SIDES=['LONG','SHORT']; H=60
F=[('BREAK5_EFF','breakout_5','eff_10'),('BREAK10_EFF','breakout_10','eff_10'),('BREAK5_R2','breakout_5','r2_20'),('BREAK10_R2','breakout_10','r2_20'),('BREAK5_MOM','breakout_5','mom_10'),('BREAK10_MOM','breakout_10','mom_10'),('RANGE_EFF','open_range_10','eff_10'),('RANGE_R2','open_range_10','r2_20'),('MOM_EFF','mom_10','eff_10'),('MOM_R2','mom_10','r2_20'),('VWAP_EFF','dist_vwap_pct','eff_10'),('VWAP_R2','dist_vwap_pct','r2_20')]
E['day']=pd.to_datetime(E.trade_date).dt.normalize(); E['entry_ts']=E.day+pd.to_timedelta(E.entry.astype(str)+':00'); E['label_end']=E.entry_ts+pd.Timedelta(minutes=H)
dates=np.array(sorted(E.day.unique())); blocks=np.array_split(dates,5)
if len(dates)<100: raise RuntimeError('Insufficient dates')
folds=[(np.concatenate(blocks[:2]),blocks[2]),(np.concatenate(blocks[:3]),blocks[3])]; final=set(blocks[4])
def signed(y,s): return y if s=='LONG' else -y
def stats(y,s):
 y=signed(pd.Series(y).dropna().astype(float),s)
 if len(y)==0:return None
 return dict(n=len(y),win=float((y>0).mean()),mean=float(y.mean()),median=float(y.median()),q25=float(y.quantile(.25)),q75=float(y.quantile(.75)))
def mask(d,s,a,ath,b,bth):
 direction=d.mom_10>0 if s=='LONG' else d.mom_10<0
 ga=d[a]==(1 if s=='LONG' else -1) if a.startswith('breakout_') else d[a].abs()>=ath
 return direction & ga & (d[b].abs()>=bth)
rows=[]
for fi,(td,vd) in enumerate(folds,1):
 tr=E[(E.day.isin(td))&(E.label_end<=pd.Timestamp(vd[0]))]; va=E[E.day.isin(vd)]
 for sym in sorted(E.symbol.unique()):
  for ent in ENTRY:
   for s in SIDES:
    a=tr[(tr.symbol==sym)&(tr.entry==ent)]; v=va[(va.symbol==sym)&(va.entry==ent)]
    if len(a)<50 or len(v)<25: continue
    for fam,x,y in F:
     av=[('EVENT',None)] if x.startswith('breakout_') else [(f'Q{q:.2f}',float(a[x].dropna().abs().quantile(q))) for q in (.60,.70,.80,.90)]
     bv=[(f'Q{q:.2f}',float(a[y].dropna().abs().quantile(q))) for q in (.60,.70,.80,.90)]
     for ar,at in av:
      for br,bt in bv:
       st=stats(v.loc[mask(v,s,x,at,y,bt),'fwd_60'],s)
       if st and st['n']>=25 and st['median']>0 and st['win']>=.52:
        rows.append(dict(symbol=sym,entry=ent,side=s,family=fam,gate_a=x,gate_a_rule=ar,gate_a_threshold=at,gate_b=y,gate_b_rule=br,gate_b_threshold=bt,fold=fi,validation_n=st['n'],validation_win=st['win'],validation_mean=st['mean'],validation_median=st['median'],validation_q25=st['q25'],validation_q75=st['q75'],score=st['median']*100*(st['win']-.5)*np.sqrt(st['n'])))
C=pd.DataFrame(rows)
if C.empty: raise RuntimeError('No candidates survived')
C.to_csv(OUT/'ENTRY_TRIGGER_CANDIDATES_ALL.csv',index=False)
keys=['symbol','entry','side','family','gate_a','gate_a_rule','gate_a_threshold','gate_b','gate_b_rule','gate_b_threshold']
A=C.groupby(keys,dropna=False).agg(folds=('fold','nunique'),validation_n=('validation_n','sum'),validation_win=('validation_win','mean'),validation_mean=('validation_mean','mean'),validation_median=('validation_median','mean')).reset_index(); A=A[A.folds==2].copy()
if A.empty: raise RuntimeError('No trigger survived both chronological validation folds')
A['robust_score']=A.validation_median*100*(A.validation_win-.5)*np.sqrt(A.validation_n); A=A.sort_values(['symbol','side','robust_score'],ascending=[True,True,False]); best=A.groupby(['symbol','side'],as_index=False).head(1).copy()
te=E[E.day.isin(final)]; out=[]
for _,r in best.iterrows():
 x=te[(te.symbol==r.symbol)&(te.entry==r.entry)]; st=stats(x.loc[mask(x,r.side,r.gate_a,r.gate_a_threshold,r.gate_b,r.gate_b_threshold),'fwd_60'],r.side); d=r.to_dict()
 d.update(test_n=st['n'] if st else 0,test_win=st['win'] if st else np.nan,test_mean=st['mean'] if st else np.nan,test_median=st['median'] if st else np.nan,test_q25=st['q25'] if st else np.nan,test_q75=st['q75'] if st else np.nan); out.append(d)
BEST=pd.DataFrame(out); BEST.to_csv(OUT/'ENTRY_TRIGGERS_BEST_BY_STOCK.csv',index=False); OOS=BEST[(BEST.test_n>=30)&(BEST.test_median>0)&(BEST.test_mean>0)].copy(); OOS.to_csv(OUT/'ENTRY_TRIGGERS_OOS_SURVIVORS.csv',index=False)
protocol={'research_version':'V2_NESTED_WALK_FORWARD','features_available_at_entry_only':True,'direction_gate':'10m momentum sign','trigger':'two-gate structural/range/VWAP condition plus independent trend-quality confirmation','thresholds':'learned from prior TRAIN only','validation':'two expanding chronological folds; 60m label purge','selection':'must survive both folds; final block untouched','execution_reference':'tested 1-minute bar close; not guaranteed fill','stops_targets':'excluded'}; (OUT/'ENTRY_TRIGGER_PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
summary={'research_version':'V2_NESTED_WALK_FORWARD','candidate_rows':len(C),'robust_candidate_definitions':len(A),'best_rows':len(BEST),'oos_survivor_rows':len(OOS),'long_best':int((BEST.side=='LONG').sum()),'short_best':int((BEST.side=='SHORT').sum()),'oos_long':int((OOS.side=='LONG').sum()),'oos_short':int((OOS.side=='SHORT').sum()),'warning':'No stops/targets/costs yet.'}; (OUT/'ENTRY_TRIGGER_RESEARCH_SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(BEST.to_string(index=False)); print(OOS.to_string(index=False))
