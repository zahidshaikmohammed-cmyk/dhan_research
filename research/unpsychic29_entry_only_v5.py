from pathlib import Path
import numpy as np,pandas as pd,json,duckdb
OUT=Path('output'); OUT.mkdir(exist_ok=True)
SYMS=['360ONE','MPHASIS','HCLTECH','POLYMED','JSL','AFFLE','SONACOMS','VEDL','NATIONALUM','JKCEMENT','ZENSARTECH','KEI','TATATECH','LTM','PVRINOX','UNOMINDA','ENDURANCE','RAINBOW','SHREECEM','APTUS','MOTILALOFS','ESCORTS','NATCOPHARM','SPLPETRO','PERSISTENT','TRAVELFOOD','UBL','INDIACEM','BHARTIHEXA']
TIMES=['09:20','09:21','09:22','09:23','09:24','09:25','09:26','09:27','09:28','09:29','09:30','09:35','09:40','09:45','09:50','10:00']; TMIN=[int(x[:2])*60+int(x[3:]) for x in TIMES]
con=duckdb.connect(); con.execute('PRAGMA threads=4'); rows=[]; cov=[]
for sym in SYMS:
 p=sorted(Path('data').rglob(f'raw_1min__{sym}.parquet.chunk*.csv')) or sorted(Path('data').rglob(f'raw_1min__{sym}.parquet'))
 if not p: raise RuntimeError('MISSING_RAW:'+sym)
 arr=','.join("'"+str(x).replace("'","''")+"'" for x in p)
 q=f"SELECT CAST(timestamp AS TIMESTAMPTZ) AT TIME ZONE 'Asia/Kolkata' ts,CAST(open AS DOUBLE)o,CAST(high AS DOUBLE)h,CAST(low AS DOUBLE)l,CAST(close AS DOUBLE)c,CAST(volume AS DOUBLE)v FROM read_csv_auto([{arr}],header=true,union_by_name=true,ignore_errors=false) ORDER BY ts"
 d=con.execute(q).df(); d['date']=d.ts.dt.date; d['m']=d.ts.dt.hour*60+d.ts.dt.minute; used=0
 for dt,z0 in d.groupby('date',sort=True):
  z=z0[(z0.m>=555)&(z0.m<=930)].sort_values('ts').reset_index(drop=True); idx={int(m):i for i,m in enumerate(z.m)}
  if not all(m in idx for m in TMIN) or 555 not in idx: continue
  prior=d[d.date<dt]
  if prior.empty: continue
  used+=1; prev=float(prior.iloc[-1].c)
  for em,t in zip(TMIN,TIMES):
   i=idx[em]; p0=float(z.c.iloc[i]); nxt=float(z.o.iloc[i+1]) if i+1<len(z) else np.nan
   def mom(n): return 100*(p0/float(z.c.iloc[i-n])-1) if i>=n else np.nan
   def eff(n):
    a=z.c.iloc[max(0,i-n+1):i+1].to_numpy(float); path=np.abs(np.diff(a)/a[:-1]).sum()*100 if len(a)>1 else 0; return abs(100*(a[-1]/a[0]-1))/path if path else 0
   def r2(n):
    a=z.c.iloc[max(0,i-n+1):i+1].to_numpy(float)
    if len(a)<4 or np.any(a<=0): return np.nan
    y=np.log(a); x=np.arange(len(a)); b=np.polyfit(x,y,1); fit=np.polyval(b,x); den=((y-y.mean())**2).sum(); return 1-((y-fit)**2).sum()/max(den,1e-12)
   f={'symbol':sym,'trade_date':str(dt),'entry':t,'price_close':p0,'next_open':nxt,'gap_pct':100*(float(z.o.iloc[idx[555]])/prev-1)}
   for n in [3,5,10,15]: f[f'mom{n}']=mom(n); f[f'eff{n}']=eff(n); f[f'r2{n}']=r2(n)
   for n in [5,10]:
    j=max(0,i-n+1); f[f'range{n}']=100*(float(z.h.iloc[j:i+1].max())-float(z.l.iloc[j:i+1].min()))/p0; f[f'break{n}']=1 if p0>float(z.h.iloc[max(0,i-n):i].max()) else -1 if p0<float(z.l.iloc[max(0,i-n):i].min()) else 0
   v5=float(z.v.iloc[max(0,i-4):i+1].mean()); v20=float(z.v.iloc[max(0,i-19):i+1].mean()); f['relvol']=v5/v20 if v20>0 else np.nan; vol=z.v.iloc[:i+1]; vw=float((((z.h+z.l+z.c)/3)*z.v).iloc[:i+1].sum()/vol.sum()) if vol.sum()>0 else np.nan; f['vwap_dist']=100*(p0/vw-1) if vw>0 else np.nan
   for n in [3,5,10]:
    j=max(0,i-n+1); f[f'long_invalid_{n}']=float(z.l.iloc[j:i+1].min()); f[f'short_invalid_{n}']=float(z.h.iloc[j:i+1].max())
   for h in [30,60]: f[f'fwd{h}']=100*(float(z.c.iloc[i+h])/p0-1) if i+h<len(z) else np.nan
   rows.append(f)
 cov.append({'symbol':sym,'sessions':used,'raw_rows':len(d)})
E=pd.DataFrame(rows); E.to_parquet(OUT/'ENTRY_ONLY_EVENTS_V5.parquet',index=False); pd.DataFrame(cov).to_csv(OUT/'ENTRY_ONLY_COVERAGE_V5.csv',index=False)
days=np.array(sorted(E.trade_date.unique())); blocks=np.array_split(days,5)
features=['mom3','mom5','mom10','mom15','eff5','eff10','r25','r210','relvol','vwap_dist','range5','range10','break5','break10']; C=[]
for sym in SYMS:
 for ent in TIMES:
  g=E[(E.symbol==sym)&(E.entry==ent)]
  if len(g)<80: continue
  for side in ['LONG','SHORT']:
   sign=1 if side=='LONG' else -1
   for feat in features:
    for q in [.60,.70,.80]:
     fs=[]
     for k in range(1,4):
      tr=g[g.trade_date.isin(np.concatenate(blocks[:k]))]; va=g[g.trade_date.isin(blocks[k])]
      if len(tr)<25 or len(va)<15: fs=[]; break
      vals=tr[feat].dropna().abs();
      if len(vals)<25: fs=[]; break
      th=float(vals.quantile(q)) if not feat.startswith('break') else .5
      mask=(va.mom10*sign>0)&((va[feat]==sign) if feat.startswith('break') else (va[feat].abs()>=th)); y=(va.loc[mask,'fwd60']*sign).dropna()
      if len(y)<10: fs=[]; break
      fs.append((len(y),float((y>0).mean()),float(y.mean()),float(y.median())))
     if len(fs)==3 and all(x[1]>=.5 and x[2]>0 and x[3]>0 for x in fs): C.append({'symbol':sym,'entry':ent,'side':side,'feature':feat,'q':q,'fold_n':[x[0] for x in fs],'fold_win':[x[1] for x in fs],'fold_mean':[x[2] for x in fs],'fold_median':[x[3] for x in fs],'robust_win':np.mean([x[1] for x in fs]),'robust_mean':np.mean([x[2] for x in fs]),'robust_median':np.mean([x[3] for x in fs])})
C=pd.DataFrame(C)
if C.empty: raise RuntimeError('NO_ROBUST_CANDIDATES')
C.to_csv(OUT/'ENTRY_ONLY_ROBUST_CANDIDATES_V5.csv',index=False); B=C.sort_values(['robust_median','robust_mean','robust_win'],ascending=False).groupby('symbol',as_index=False).head(1)
final=[]
for _,r in B.iterrows():
 g=E[(E.symbol==r.symbol)&(E.entry==r.entry)]; sign=1 if r.side=='LONG' else -1; train=g[g.trade_date.isin(np.concatenate(blocks[:3]))]; th=float(train[r.feature].dropna().abs().quantile(r.q)) if not r.feature.startswith('break') else .5; te=g[g.trade_date.isin(blocks[4])]; mask=(te.mom10*sign>0)&((te[r.feature]==sign) if r.feature.startswith('break') else (te[r.feature].abs()>=th)); y=(te.loc[mask,'fwd60']*sign).dropna(); rr=r.to_dict(); rr.update(threshold=th,test_n=len(y),test_win=float((y>0).mean()) if len(y) else np.nan,test_mean=float(y.mean()) if len(y) else np.nan,test_median=float(y.median()) if len(y) else np.nan,executable_entry='NEXT_1MINUTE_OPEN',invalidation='LONG below completed 5-bar low / SHORT above completed 5-bar high',invalidation_field='long_invalid_5 / short_invalid_5',status='OOS_POSITIVE' if len(y)>=10 and y.mean()>0 and y.median()>0 else 'OOS_WEAK_OR_NEGATIVE'); final.append(rr)
F=pd.DataFrame(final).sort_values(['status','symbol']); F.to_csv(OUT/'UNPSYCHIC29_ENTRY_TRIGGERS_FINAL_V5.csv',index=False); summary={'status':'COMPLETE','version':'V5_ENTRY_ONLY','locked_stocks':29,'stocks_with_robust_rules':int(F.symbol.nunique()),'oos_positive':int((F.status=='OOS_POSITIVE').sum()),'excluded':['SL','targets','trailing exits','cost optimization'],'execution':'next 1-minute bar OPEN after completed signal minute','invalidation':'causal completed 5-bar low/high','features':features}; (OUT/'ENTRY_ONLY_PROTOCOL_V5.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2)); print(F.to_string(index=False))
