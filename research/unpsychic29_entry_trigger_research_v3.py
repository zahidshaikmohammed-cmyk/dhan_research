from pathlib import Path
import json, numpy as np, pandas as pd
OUT=Path('output'); OUT.mkdir(exist_ok=True)
E=pd.read_parquet(OUT/'ENTRY_EVENT_DATA.parquet').copy()
E['day']=pd.to_datetime(E.trade_date).dt.normalize(); E['entry']=E.entry.astype(str).str[:5]
ENTRY=['09:20','09:21','09:22','09:23','09:24','09:25','09:26','09:27','09:28','09:29','09:30','09:35','09:40','09:45','09:50','10:00']
SIDES=['LONG','SHORT']
F=[('BREAK5_EFF','breakout_5','eff_10'),('BREAK10_EFF','breakout_10','eff_10'),('BREAK5_R2','breakout_5','r2_20'),('BREAK10_R2','breakout_10','r2_20'),('BREAK5_MOM','breakout_5','mom_10'),('BREAK10_MOM','breakout_10','mom_10'),('RANGE_EFF','open_range_10','eff_10'),('RANGE_R2','open_range_10','r2_20'),('MOM_EFF','mom_10','eff_10'),('MOM_R2','mom_10','r2_20'),('VWAP_EFF','dist_vwap_pct','eff_10'),('VWAP_R2','dist_vwap_pct','r2_20')]
dates=np.array(sorted(E.day.unique()))
if len(dates)<100: raise RuntimeError(f'Insufficient dates: {len(dates)}')
blocks=np.array_split(dates,5)
folds=[(np.concatenate(blocks[:1]),blocks[1]),(np.concatenate(blocks[:2]),blocks[2]),(np.concatenate(blocks[:3]),blocks[3])]
final_dates=set(blocks[4])
def signed(y,side):
    y=pd.Series(y).dropna().astype(float); return y if side=='LONG' else -y
def stat(y,side):
    y=signed(y,side)
    if len(y)==0:return None
    return dict(n=len(y),win=float((y>0).mean()),mean=float(y.mean()),median=float(y.median()),q25=float(y.quantile(.25)),q75=float(y.quantile(.75)))
def mask(d,side,x,at,y,bt):
    direction=d.mom_10>0 if side=='LONG' else d.mom_10<0
    if x.startswith('breakout_'): ga=d[x]==(1 if side=='LONG' else -1)
    elif x in ('mom_10','dist_vwap_pct'): ga=(d[x]>=at) if side=='LONG' else (d[x]<=-at)
    else: ga=d[x].abs()>=at
    gb=d[y].abs()>=bt
    return direction & ga & gb
rows=[]
for fold_id,(train_dates,val_dates) in enumerate(folds,1):
    tr=E[E.day.isin(train_dates)]; va=E[E.day.isin(val_dates)]
    for sym in sorted(E.symbol.unique()):
      for ent in ENTRY:
        a=tr[(tr.symbol==sym)&(tr.entry==ent)]; v=va[(va.symbol==sym)&(va.entry==ent)]
        if len(a)<25 or len(v)<15: continue
        for family,x,y in F:
          ax=a[x].dropna().abs(); by=a[y].dropna().abs()
          if len(ax)<25 or len(by)<25: continue
          arules=[('EVENT',0.0)] if x.startswith('breakout_') else [(f'Q{q:.2f}',float(ax.quantile(q))) for q in (.60,.70,.80,.90)]
          brules=[(f'Q{q:.2f}',float(by.quantile(q))) for q in (.60,.70,.80,.90)]
          for ar,at in arules:
            for br,bt in brules:
              for side in SIDES:
                st=stat(v.loc[mask(v,side,x,at,y,bt),'fwd_60'],side)
                if st is None or st['n']<15: continue
                rows.append(dict(fold=fold_id,symbol=sym,entry=ent,side=side,family=family,gate_a=x,gate_a_rule=ar,gate_a_threshold=at,gate_b=y,gate_b_rule=br,gate_b_threshold=bt,validation_n=st['n'],validation_win=st['win'],validation_mean=st['mean'],validation_median=st['median'],validation_q25=st['q25'],validation_q75=st['q75'],validation_score=st['median']*(st['win']-.5)*np.sqrt(st['n'])))
C=pd.DataFrame(rows); C.to_csv(OUT/'ENTRY_TRIGGER_CANDIDATES_ALL.csv',index=False)
if C.empty: raise RuntimeError('No causal trigger candidates; event data or feature coverage is invalid')
keys=['symbol','entry','side','family','gate_a','gate_a_rule','gate_a_threshold','gate_b','gate_b_rule','gate_b_threshold']
A=C.groupby(keys,dropna=False).agg(folds=('fold','nunique'),validation_n=('validation_n','sum'),validation_win=('validation_win','mean'),validation_mean=('validation_mean','mean'),validation_median=('validation_median','mean'),validation_q25=('validation_q25','mean'),validation_q75=('validation_q75','mean'),robust_score=('validation_score','mean')).reset_index()
A=A[A.folds==len(folds)].copy(); A.to_csv(OUT/'ENTRY_TRIGGER_ROBUST_CANDIDATES.csv',index=False)
if A.empty: raise RuntimeError('No trigger definition survived all three chronological validation folds')
A=A.sort_values(['symbol','robust_score','validation_median','validation_win'],ascending=[True,False,False,False])
best=A.groupby('symbol',as_index=False).head(1).copy(); best.to_csv(OUT/'ENTRY_TRIGGERS_BEST_VALIDATED_29.csv',index=False)
te=E[E.day.isin(final_dates)]; out=[]
for _,r in best.iterrows():
    x=te[(te.symbol==r.symbol)&(te.entry==r.entry)]; st=stat(x.loc[mask(x,r.side,r.gate_a,r.gate_a_threshold,r.gate_b,r.gate_b_threshold),'fwd_60'],r.side)
    d=r.to_dict(); d.update(test_n=st['n'] if st else 0,test_win=st['win'] if st else np.nan,test_mean=st['mean'] if st else np.nan,test_median=st['median'] if st else np.nan,test_q25=st['q25'] if st else np.nan,test_q75=st['q75'] if st else np.nan)
    d['oos_status']='SURVIVES_OOS' if st and st['n']>=15 and st['median']>0 and st['mean']>0 else 'FAIL_OOS'; out.append(d)
BEST=pd.DataFrame(out); BEST.to_csv(OUT/'ENTRY_TRIGGERS_BEST_BY_STOCK.csv',index=False)
OOS=BEST[BEST.oos_status=='SURVIVES_OOS']; OOS.to_csv(OUT/'ENTRY_TRIGGERS_OOS_SURVIVORS.csv',index=False)
protocol={'research_version':'V3_NESTED_WALK_FORWARD','universe':'locked UNPSYCHIC29','entry_times':ENTRY,'label':'signed 60-minute close-to-close return after simulated entry minute','folds':'3 expanding chronological train/validation folds; fifth chronological block untouched OOS','thresholds':'learned only from each fold TRAIN absolute distributions','trigger':'directional 10-minute momentum + one structural/range/VWAP condition + independent trend-quality condition','directional_features':'mom_10 and dist_vwap_pct use side-specific sign; breakout uses side-specific breakout; non-directional gates use magnitude','minimums':'25 TRAIN events and 15 validation events per stock/time/definition; 15 OOS events minimum','selection':'definition must occur in all 3 validation folds; best robust validation definition selected independently per stock','leakage':'no future prices/volumes are used as features; forward returns are labels only','execution':'research trigger is evaluated at the observed entry-minute close; slippage, stops, targets and costs deliberately deferred'}
(OUT/'ENTRY_TRIGGER_PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
summary={'research_version':'V3_NESTED_WALK_FORWARD','status':'COMPLETE','candidate_rows':len(C),'robust_candidate_definitions':len(A),'best_stock_rows':len(BEST),'oos_survivor_rows':len(OOS),'unique_best_stocks':int(BEST.symbol.nunique()),'long_best':int((BEST.side=='LONG').sum()),'short_best':int((BEST.side=='SHORT').sum()),'oos_long':int((OOS.side=='LONG').sum()),'oos_short':int((OOS.side=='SHORT').sum()),'folds':3,'final_oos_dates':len(final_dates)}
(OUT/'ENTRY_TRIGGER_RESEARCH_SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(BEST.to_string(index=False)); print('\nOOS SURVIVORS'); print(OOS.to_string(index=False))
