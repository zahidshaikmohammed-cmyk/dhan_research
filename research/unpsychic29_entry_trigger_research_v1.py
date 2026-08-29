from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT=Path('output')
E=pd.read_parquet(OUT/'ENTRY_EVENT_DATA.parquet')

# Entry research is intentionally limited to information observable by the simulated entry.
# Thresholds are learned on TRAIN only; VALIDATION selects the rule; TEST is reported once.
# No stop-loss or target is used here.

NUM=['mom_3','mom_5','mom_10','mom_15','mom_20','eff_5','eff_10','eff_20','r2_10','r2_20','slope_10','slope_20','relvol_5_20','relvol_20_prior','dist_vwap_pct','open_range_5','open_range_10']
FAMILIES=[
 ('BREAK5_EFF','breakout_5','eff_10'),
 ('BREAK10_EFF','breakout_10','eff_10'),
 ('BREAK5_R2','breakout_5','r2_20'),
 ('BREAK10_R2','breakout_10','r2_20'),
 ('BREAK5_MOM','breakout_5','mom_10'),
 ('BREAK10_MOM','breakout_10','mom_10'),
 ('VWAP_EFF','dist_vwap_pct','eff_10'),
 ('MOM_EFF','mom_10','eff_10'),
 ('MOM_R2','mom_10','r2_20'),
 ('RANGE_MOM_EFF','open_range_10','eff_10'),
]
SIDES=['LONG','SHORT']
rows=[]

def signed(y,side): return y if side=='LONG' else -y

def metric(x,side):
    y=signed(x.fwd_60.dropna(),side)
    return len(y),float((y>0).mean()),float(y.mean()),float(y.median()),float(y.quantile(.25)),float(y.quantile(.75))

for sym,g0 in E.groupby('symbol'):
  for ent,g in g0.groupby('entry'):
    # 'split' is a real column in ENTRY_EVENT_DATA; use bracket access because
    # DataFrame.split resolves to the Python/DataFrame method namespace rather
    # than the column on some pandas versions.
    tr=g[g['split']=='TRAIN']; va=g[g['split']=='VALIDATION']; te=g[g['split']=='TEST']
    if len(tr)<80 or len(va)<30 or len(te)<30: continue
    for side in SIDES:
      # Direction gate is deliberately fixed to the observed 10m direction.
      dtr=tr[tr.mom_10>0] if side=='LONG' else tr[tr.mom_10<0]
      dva=va[va.mom_10>0] if side=='LONG' else va[va.mom_10<0]
      dte=te[te.mom_10>0] if side=='LONG' else te[te.mom_10<0]
      if len(dtr)<40 or len(dva)<20 or len(dte)<20: continue
      for fam,a,b in FAMILIES:
        # For breakout families, a is a discrete event. For continuous gates use TRAIN quantiles.
        if a.startswith('breakout'):
          a_specs=[('EVENT',None)]
        else:
          vals=dtr[a].dropna()
          if len(vals)<40: continue
          # Side-aware thresholds: positive magnitude for LONG, negative magnitude for SHORT.
          vals=np.abs(vals.to_numpy(float))
          a_specs=[(f'Q{q}',float(np.quantile(vals,q))) for q in (.60,.70,.80,.90)]
        valsb=dtr[b].dropna()
        if len(valsb)<40: continue
        b_specs=[(f'Q{q}',float(np.quantile(np.abs(valsb.to_numpy(float)),q))) for q in (.60,.70,.80)]
        for aname,ath in a_specs:
          for bname,bth in b_specs:
            def filt(d):
              x=d.copy()
              if a.startswith('breakout'):
                x=x[x[a]==(1 if side=='LONG' else -1)]
              else:
                x=x[np.abs(x[a])>=ath]
              # For efficiency/R2 require high magnitude; momentum is already
              # direction-gated by dtr/dva/dte above.
              x=x[np.abs(x[b])>=bth]
              return x
            A=filt(dva); T=filt(dte)
            if len(A)<20 or len(T)<20: continue
            an,aw,amean,amed,aq25,aq75=metric(A,side); tn,tw,tmean,tmed,tq25,tq75=metric(T,side)
            # Selection score favors validation median and win rate, but requires positive median.
            score=(amed*100.0)*(aw-0.5)
            rows.append({'symbol':sym,'entry':ent,'side':side,'family':fam,'gate_a':a,'gate_a_rule':aname,'gate_a_threshold':ath,'gate_b':b,'gate_b_rule':bname,'gate_b_threshold':bth,'validation_n':an,'validation_win':aw,'validation_mean':amean,'validation_median':amed,'validation_q25':aq25,'validation_q75':aq75,'test_n':tn,'test_win':tw,'test_mean':tmean,'test_median':tmed,'test_q25':tq25,'test_q75':tq75,'selection_score':score})

R=pd.DataFrame(rows)
if R.empty: raise RuntimeError('No trigger candidates survived minimum sample requirements')
R.to_csv(OUT/'ENTRY_TRIGGER_CANDIDATES_ALL.csv',index=False)

# For each stock choose at most one LONG and one SHORT trigger using VALIDATION only.
# Require validation median > 0 and at least 52% validation wins; then prefer higher validation median,
# with a small penalty for complexity only through family count (all candidates here have two gates).
sel=R[(R.validation_median>0)&(R.validation_win>=0.52)].copy()
sel=sel.sort_values(['symbol','side','validation_median','validation_win','validation_n'],ascending=[True,True,False,False,False])
best=sel.groupby(['symbol','side'],as_index=False).head(1).copy()
best.to_csv(OUT/'ENTRY_TRIGGERS_BEST_BY_STOCK.csv',index=False)

# Robust subset: candidate must have positive validation and positive test median AND positive test mean.
rob=R[(R.validation_median>0)&(R.test_median>0)&(R.test_mean>0)&(R.validation_win>=0.52)&(R.test_n>=30)].copy()
rob=rob.sort_values(['symbol','side','test_median','test_win'],ascending=[True,True,False,False])
rob.groupby(['symbol','side'],as_index=False).head(1).to_csv(OUT/'ENTRY_TRIGGERS_OOS_SURVIVORS.csv',index=False)

# Entry trigger protocol: the trigger is a completed 1-minute state at the tested minute.
# Execution price is the signal-bar close in this research layer. This is NOT yet a slippage-adjusted fill.
protocol={
 'direction_gate':'10-minute momentum sign: positive for LONG, negative for SHORT.',
 'confirmation':'A discrete breakout_5/10 event or a validated continuous gate plus a second trend-quality gate.',
 'threshold_learning':'All numeric thresholds are learned from TRAIN only.',
 'validation_selection':'One candidate per stock/side is selected on VALIDATION; TEST is never used for selection.',
 'minimum_samples':'Validation >=20 and test >=20 for candidate reporting; OOS survivor table additionally requires test >=30.',
 'entry_definition':'Signal is valid at the close of the simulated 1-minute entry bar. This layer does not assume a future price.',
 'execution_price':'Signal-bar close is the research reference price only; actual next-tick/next-open slippage must be tested separately.',
 'stop_target':'Explicitly excluded from this phase.',
 'no_trade':'If the direction gate, trigger gate, or trend-quality gate is absent, no entry permission is granted.'
}
(OUT/'ENTRY_TRIGGER_PROTOCOL.json').write_text(json.dumps(protocol,indent=2))

summary={'candidate_rows':int(len(R)),'best_by_stock_rows':int(len(best)),'oos_survivor_rows':int(len(rob)),'stocks_with_long_best':int((best.side=='LONG').sum()),'stocks_with_short_best':int((best.side=='SHORT').sum()),'stocks_with_oos_long':int(((rob.side=='LONG')).sum()),'stocks_with_oos_short':int(((rob.side=='SHORT')).sum()),'warning':'These are research triggers, not yet cost-adjusted executable signals. The signal-bar close is a reference price; realistic fill/latency/slippage must be tested separately.'}
(OUT/'ENTRY_TRIGGER_RESEARCH_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('\nBEST BY STOCK/SIDE')
print(best[['symbol','entry','side','family','gate_a','gate_a_rule','gate_a_threshold','gate_b','gate_b_rule','gate_b_threshold','validation_n','validation_win','validation_median','test_n','test_win','test_median']].to_string(index=False))
print('\nOOS SURVIVORS')
print(rob[['symbol','entry','side','family','gate_a_rule','gate_a_threshold','gate_b_rule','gate_b_threshold','validation_n','validation_win','validation_median','test_n','test_win','test_median']].to_string(index=False))
