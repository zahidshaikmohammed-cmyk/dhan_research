from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb

OUT=Path('output'); OUT.mkdir(exist_ok=True)
SYMS=['360ONE','MPHASIS','HCLTECH','POLYMED','JSL','AFFLE','SONACOMS','VEDL','NATIONALUM','JKCEMENT','ZENSARTECH','KEI','TATATECH','LTM','PVRINOX','UNOMINDA','ENDURANCE','RAINBOW','SHREECEM','APTUS','MOTILALOFS','ESCORTS','NATCOPHARM','SPLPETRO','PERSISTENT','TRAVELFOOD','UBL','INDIACEM','BHARTIHEXA']
ENTRY=['09:20','09:21','09:22','09:23','09:24','09:25','09:26','09:27','09:28','09:29','09:30','09:35','09:40','09:45','09:50','10:00']
ENTRY_MIN=[int(x[:2])*60+int(x[3:]) for x in ENTRY]
H=[15,30,60,120,300]
con=duckdb.connect(); con.execute('PRAGMA threads=4')

def r2_slope(v):
    v=np.asarray(v,float)
    if len(v)<3 or np.any(v<=0): return np.nan,np.nan
    x=np.arange(len(v)); y=np.log(v); b=np.polyfit(x,y,1); fit=np.polyval(b,x)
    den=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-fit)**2)/max(den,1e-12)),float(100*b[0])

def efficiency(v):
    v=np.asarray(v,float)
    if len(v)<2:return np.nan
    path=np.abs(np.diff(v)/v[:-1]).sum()*100
    return float(abs(100*(v[-1]/v[0]-1))/path) if path>0 else 0.0

events=[]; integrity=[]
for sym in SYMS:
    paths=sorted(Path('data').rglob(f'raw_1min__{sym}.parquet.chunk*.csv'))
    if not paths: paths=sorted(Path('data').rglob(f'raw_1min__{sym}.parquet'))
    if not paths: raise RuntimeError(f'No raw data found for locked symbol {sym}')
    arr=','.join("'"+str(p).replace("'","''")+"'" for p in paths)
    q=f'''SELECT CAST(timestamp AS TIMESTAMPTZ) AT TIME ZONE 'Asia/Kolkata' ts,CAST(open AS DOUBLE)o,CAST(high AS DOUBLE)h,CAST(low AS DOUBLE)l,CAST(close AS DOUBLE)cl,CAST(volume AS DOUBLE)v FROM read_csv_auto([{arr}],header=true,union_by_name=true,ignore_errors=false) ORDER BY ts'''
    d=con.execute(q).df(); d['date']=d.ts.dt.date; d['minute']=d.ts.dt.hour*60+d.ts.dt.minute
    used=[]
    for dt,z0 in d.groupby('date',sort=True):
        z=z0[(z0.minute>=555)&(z0.minute<=930)].sort_values('ts').reset_index(drop=True)
        if len(z)<360: continue
        idx={m:i for i,m in enumerate(z.minute)}
        if not all(m in idx for m in ENTRY_MIN): continue
        prior=d[d.date<dt]
        if prior.empty: continue
        prev_close=float(prior.sort_values('ts').iloc[-1].cl); used.append(dt)
        for em,elabel in zip(ENTRY_MIN,ENTRY):
            ei=idx[em]; p=float(z.cl.iloc[ei]); gap=100*(float(z.o.iloc[idx[555]])/prev_close-1)
            def ret(n):
                j=ei-n; return 100*(p/float(z.cl.iloc[j])-1) if j>=0 else np.nan
            def rng(n):
                j=max(0,ei-n+1); return 100*(float(z.h.iloc[j:ei+1].max())-float(z.l.iloc[j:ei+1].min()))/p
            def eff(n):
                j=max(0,ei-n+1); return efficiency(z.cl.iloc[j:ei+1].to_numpy())
            def trend(n):
                j=max(0,ei-n+1); return r2_slope(z.cl.iloc[j:ei+1].to_numpy())
            f={'symbol':sym,'trade_date':dt,'entry':elabel,'entry_minute':em,'price':p,'gap_pct':gap}
            for n in [1,3,5,10,15,20,30]: f[f'mom_{n}']=ret(n)
            for n in [5,10,15,20,30]: f[f'range_{n}']=rng(n); f[f'eff_{n}']=eff(n); f[f'r2_{n}'],f[f'slope_{n}']=trend(n)
            for n in [3,5,10]: f[f'upbars_{n}']=float((z.cl.iloc[ei-n+1:ei+1].to_numpy()>z.o.iloc[ei-n+1:ei+1].to_numpy()).mean())
            v5=float(z.v.iloc[max(0,ei-4):ei+1].mean()); v20=float(z.v.iloc[max(0,ei-19):ei+1].mean()); vp=float(z.v.iloc[max(0,ei-39):max(0,ei-19)].mean())
            f['relvol_5_20']=v5/v20 if v20>0 else np.nan; f['relvol_20_prior']=v20/vp if vp>0 else np.nan
            vol=z.v.iloc[:ei+1]; vw=(((z.h+z.l+z.cl)/3)*z.v).iloc[:ei+1].sum()/vol.sum() if vol.sum()>0 else np.nan
            f['open_range_5']=100*(z.h.iloc[:min(ei+1,5)].max()-z.l.iloc[:min(ei+1,5)].min())/float(z.o.iloc[0]); f['open_range_10']=100*(z.h.iloc[:min(ei+1,10)].max()-z.l.iloc[:min(ei+1,10)].min())/float(z.o.iloc[0]); f['dist_vwap_pct']=100*(p/vw-1) if pd.notna(vw) and vw>0 else np.nan
            for n in [5,10,20]:
                j=max(0,ei-n); hi=float(z.h.iloc[j:ei].max()) if ei>j else np.nan; lo=float(z.l.iloc[j:ei].min()) if ei>j else np.nan
                f[f'breakout_{n}']=1 if p>hi else -1 if p<lo else 0
            for h in H:
                j=ei+h
                if j<len(z):
                    f[f'fwd_{h}']=100*(float(z.cl.iloc[j])/p-1); f[f'fwd_high_{h}']=100*(float(z.h.iloc[ei+1:j+1].max())/p-1); f[f'fwd_low_{h}']=100*(float(z.l.iloc[ei+1:j+1].min())/p-1)
                else: f[f'fwd_{h}']=np.nan; f[f'fwd_high_{h}']=np.nan; f[f'fwd_low_{h}']=np.nan
            j=min(len(z)-1,ei+300); f['fwd_close']=100*(float(z.cl.iloc[j])/p-1)
            events.append(f)
    integrity.append({'symbol':sym,'sessions_used':len(used),'first_session':str(min(used)) if used else None,'last_session':str(max(used)) if used else None,'raw_rows':len(d),'timestamp_min':str(d.ts.min()),'timestamp_max':str(d.ts.max())})
    print(sym,len(used),flush=True)

E=pd.DataFrame(events); E.to_parquet(OUT/'ENTRY_EVENT_DATA.parquet',index=False); pd.DataFrame(integrity).to_csv(OUT/'DATA_COVERAGE_29.csv',index=False)
dates=np.array(sorted(pd.to_datetime(E.trade_date).unique())); cut1=dates[int(len(dates)*.60)]; cut2=dates[int(len(dates)*.80)]; E['split']=np.where(pd.to_datetime(E.trade_date)<cut1,'TRAIN',np.where(pd.to_datetime(E.trade_date)<cut2,'VALIDATION','TEST'))

# Raw entry-time sweep. This is descriptive, not a fitted model.
sweep=[]
for ent,g in E.groupby('entry'):
    for feat in ['mom_3','mom_5','mom_10','mom_15','mom_20','mom_30','eff_5','eff_10','eff_20','r2_10','r2_20','slope_10','slope_20','relvol_5_20','dist_vwap_pct']:
        for side in ['LONG','SHORT']:
            if feat.startswith(('eff_','r2_','relvol_')): sig=(g.mom_10>0) if side=='LONG' else (g.mom_10<0)
            else: sig=(g[feat]>0) if side=='LONG' else (g[feat]<0)
            for h in [30,60,120]:
                y=g.loc[sig,f'fwd_{h}'].dropna();
                if len(y)<50: continue
                sr=y if side=='LONG' else -y
                sweep.append({'entry':ent,'feature':feat,'side':side,'horizon':h,'signals':len(sr),'win_rate':float((sr>0).mean()),'median_signed_return':float(sr.median()),'mean_signed_return':float(sr.mean()),'q25':float(sr.quantile(.25)),'q75':float(sr.quantile(.75))})
pd.DataFrame(sweep).to_csv(OUT/'ENTRY_TIME_FEATURE_SWEEP.csv',index=False)

# Threshold discovery: thresholds are estimated only from TRAIN+VALIDATION; TEST is untouched until final reporting.
features=['mom_3','mom_5','mom_10','mom_15','mom_20','mom_30','eff_5','eff_10','eff_20','r2_10','r2_20','slope_10','slope_20','relvol_5_20','relvol_20_prior','dist_vwap_pct','open_range_5','open_range_10','breakout_5','breakout_10','breakout_20']
cand=[]
for ent in ENTRY:
    base=E[E.entry==ent]
    for feat in features:
        tr=base[base.split!='TEST'][feat].dropna()
        if len(tr)<200: continue
        for q in [.50,.60,.70,.80,.90]:
            th=float(tr.abs().quantile(q)) if not feat.startswith('breakout_') else .5
            for side in ['LONG','SHORT']:
                direction=(base.mom_10>0) if side=='LONG' else (base.mom_10<0)
                filt=(base[feat].abs()>=th) if not feat.startswith('breakout_') else (base[feat]==(1 if side=='LONG' else -1))
                sig=base[direction & filt]; vv=sig[sig.split=='VALIDATION']; tt=sig[sig.split=='TEST']
                if len(vv)<50 or len(tt)<50: continue
                vr=(vv.fwd_60 if side=='LONG' else -vv.fwd_60).dropna(); trt=(tt.fwd_60 if side=='LONG' else -tt.fwd_60).dropna()
                cand.append({'entry':ent,'feature':feat,'q':q,'threshold':th,'side':side,'validation_n':len(vr),'validation_win':float((vr>0).mean()),'validation_mean':float(vr.mean()),'validation_median':float(vr.median()),'test_n':len(trt),'test_win':float((trt>0).mean()),'test_mean':float(trt.mean()),'test_median':float(trt.median())})
C=pd.DataFrame(cand); C['val_score']=C.validation_mean*C.validation_win; C=C.sort_values(['val_score','validation_mean','validation_win'],ascending=False); C.to_csv(OUT/'SIGNAL_DISCOVERY_CANDIDATES.csv',index=False); C.head(50).to_csv(OUT/'TOP_SIGNAL_CANDIDATES.csv',index=False)

# Per-stock personality: entry time is chosen from validation only by median absolute 60m movement, then test is reported separately.
pers=[]
for sym,g in E.groupby('symbol'):
    rows=[]
    for ent,x in g[g.split=='VALIDATION'].groupby('entry'):
        y=x.fwd_60.dropna()
        if len(y)>=30: rows.append((float(y.abs().median()),ent,float(y.median()),float((y>0).mean()),len(y)))
    if rows:
        _,ent,med,up,n=max(rows); z=g[g.entry==ent]; tz=z[z.split=='TEST']; pers.append({'symbol':sym,'best_validation_entry':ent,'validation_median_60m_return':med,'validation_up_rate':up,'validation_events':n,'all_events':len(z),'median_gap_pct':z.gap_pct.median(),'median_mom10_pct':z.mom_10.median(),'median_eff10':z.eff_10.median(),'median_r2_20':z.r2_20.median(),'median_relvol_5_20':z.relvol_5_20.median(),'test_60m_median':tz.fwd_60.median(),'test_60m_up_rate':(tz.fwd_60>0).mean(),'test_events':len(tz)})
pd.DataFrame(pers).sort_values('symbol').to_csv(OUT/'STOCK_PERSONALITIES_29.csv',index=False)

# Robustness of the chosen discovery direction across chronological quarters.
blocks=np.array_split(dates,4); bd=[]
for bi,b in enumerate(blocks,1):
    x=E[pd.to_datetime(E.trade_date).isin(b)]
    for sym,z in x.groupby('symbol'):
        bd.append({'symbol':sym,'block':bi,'ret60_median':z.fwd_60.median(),'up_rate':(z.fwd_60>0).mean(),'abs60_median':z.fwd_60.abs().median(),'eff10_median':z.eff_10.median(),'r2_20_median':z.r2_20.median(),'late120_same_sign':((z.mom_10*z.fwd_120)>0).mean()})
pd.DataFrame(bd).to_csv(OUT/'CHRONOLOGICAL_STABILITY_4BLOCKS.csv',index=False)

summary={'locked_universe':SYMS,'events':int(len(E)),'sessions_by_stock':pd.DataFrame(integrity).set_index('symbol').sessions_used.to_dict(),'entry_times_tested':ENTRY,'horizons_tested_minutes':H,'chronological_split':{'train_end':str(cut1),'validation_end':str(cut2),'test_start':str(cut2)},'lookahead_policy':'All feature columns are constructed only from bars at or before the simulated entry minute. Forward returns and excursions are labels only. No future price or volume is used as an entry feature.','market_sector_context':'Not present in supplied repository dataset; deliberately not fabricated.','stop_target_phase':'Deferred to optimized event-level simulator; this run intentionally does not execute per-event database queries.','outputs':['DATA_COVERAGE_29.csv','ENTRY_EVENT_DATA.parquet','ENTRY_TIME_FEATURE_SWEEP.csv','SIGNAL_DISCOVERY_CANDIDATES.csv','TOP_SIGNAL_CANDIDATES.csv','STOCK_PERSONALITIES_29.csv','CHRONOLOGICAL_STABILITY_4BLOCKS.csv']}
(OUT/'DEEP_RESEARCH_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str))
print(json.dumps(summary,indent=2)); print('\nTOP 20 VALIDATION-DISCOVERED CONDITIONS'); print(C.head(20).to_string(index=False))
