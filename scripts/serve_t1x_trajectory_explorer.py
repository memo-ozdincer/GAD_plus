#!/usr/bin/env python3
"""Fast local GADplus run explorer with a DuckDB index and lazy A/B traces."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq
from flask import Flask, Response, jsonify, request
from plotly.offline import get_plotlyjs

from gadplus.logging.wandb_export import (
    enrich_rows,
    event_preserving_indices,
    kabsch_rmsd,
    load_bundle,
)

FIELDS = (
    "id", "surface", "method", "panel", "noise", "noise_unit", "sample_id",
    "seed", "formula", "rxn", "initial_n_neg", "final_n_neg", "final_force",
    "final_lambda1", "final_lambda2", "evaluations", "budget", "local_ts",
    "strict_ts", "endpoint_minima", "native_topology", "failure", "has_trace",
    "trace_kind", "trace_path", "hparams",
)
SORTABLE = frozenset(FIELDS[:23])


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def topology_rows(root: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for path in root.glob("task_*.json"):
        row = json.loads(path.read_text())
        if "sample_id" in row:
            result[int(row["sample_id"])] = row
    return result


def build_index(manifest: Path, lj_results: Path, database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    records: list[tuple[Any, ...]] = []
    for spec in json.loads(manifest.read_text()):
        method, noise, budget = spec["family"], float(spec["noise_angstrom"]), int(spec["budget_updates"])
        root, topo = Path(spec["raw_root"]), topology_rows(Path(spec["topology_root"]))
        if method in {"competitive", "competitive_subspace"}:
            for path in sorted((root / "tasks").glob("task_*.json")):
                row = json.loads(path.read_text()); sid = int(row["sample_id"]); end = topo.get(sid, {})
                fmax, nneg = finite(row.get("final_fmax")), int(row.get("final_n_neg", -1))
                trace = str(row.get("trajectory_bundle", "")); error = str(row.get("error", ""))
                params = {"variant": row.get("gate_variant"), "spectral_temperature": row.get("spectral_temperature"), "step_fraction": row.get("step_fraction")}
                records.append((f"gxtb:{method}:{noise}:{sid}", "Transition1x / g-xTB", method, "matched test", noise, "Å", sid, row.get("seed"), row.get("formula", ""), row.get("rxn", ""), row.get("initial_n_neg"), nneg, fmax, finite(row.get("final_lambda1")), finite(row.get("final_lambda2")), int(row.get("n_evaluations", 0)), budget, nneg == 1 and fmax is not None and fmax < .03, nneg == 1 and fmax is not None and fmax < .01, bool(end.get("endpoint_minima")), bool(end.get("native_endpoint_topology")), error, bool(trace and Path(trace).is_dir()), "bundle", trace, json.dumps(params, sort_keys=True)))
        else:
            for path in sorted(root.glob("task_*/summary_*.parquet")):
                row = pq.read_table(path).to_pylist()[0]; sid = int(row["sample_id"]); end = topo.get(sid, {})
                fmax, nneg = finite(row.get("final_force_max")), int(row.get("final_n_neg", -1))
                if method == "sella":
                    trace, kind = str(row.get("sella_trace_path", "")), "sella_npz"
                    params = {"coordinates": "Cartesian", "projection": "Eckart"}
                else:
                    trace, kind = str(next(iter(path.parent.glob("traj_*.parquet")), "")), "regular_parquet"
                    params = {"dt": .003, "mode": "instantaneous lowest", "projection": "Eckart"}
                failure = str(row.get("failure_type", "") or row.get("final_eval_error", ""))
                records.append((f"gxtb:{method}:{noise}:{sid}", "Transition1x / g-xTB", method, "matched test", noise, "Å", sid, None, row.get("formula", ""), row.get("rxn", ""), None, nneg, fmax, finite(row.get("final_eig0")), None, int(row.get("total_steps", 0)), budget, nneg == 1 and fmax is not None and fmax < .03, nneg == 1 and fmax is not None and fmax < .01, bool(end.get("endpoint_minima")), bool(end.get("native_endpoint_topology")), failure, bool(trace and Path(trace).is_file()), kind, trace, json.dumps(params, sort_keys=True)))
    if lj_results.is_file():
        lj_params = {
            "ordinary_gad": {"dt": .005, "cap": .005, "budget": 8000},
            "hard_gate": {"gate": "lambda2 >= 0", "dt": .005, "cap": .005, "budget": 8000},
            "historical_lambda2": {"gate": "sigmoid(50 lambda2)", "dt": .005, "cap": .005, "budget": 8000},
            "intrinsic": {"spectral_temperature": .01, "step_fraction": .05, "budget": 200},
        }
        for row in json.loads(lj_results.read_text()):
            method, sid = row["method"], int(row["sample_id"]); params = lj_params[method]
            fmax, nneg = finite(row.get("final_force_max")), int(row.get("final_n_neg", -1))
            records.append((f"lj7:{method}:{row['noise']}:{row['panel']}:{sid}", "analytic reduced LJ7", method, row["panel"], float(row["noise"]), "σ", sid, row.get("seed"), "LJ7", row["panel"], row.get("initial_n_neg"), nneg, fmax, finite(row.get("final_lambda1")), finite(row.get("final_lambda2")), int(row.get("n_evaluations", 0)), int(params["budget"]), bool(row.get("converged")), bool(row.get("converged")), bool(row.get("downhill_valid")), bool(row.get("correct_event")), str(row.get("failure_type", "") or row.get("error", "")), False, "none", "", json.dumps(params, sort_keys=True)))
    con = duckdb.connect(str(database)); con.execute("DROP TABLE IF EXISTS runs")
    con.execute("CREATE TABLE runs AS SELECT * FROM (VALUES " + ",".join(["(" + ",".join(["?"] * len(FIELDS)) + ")"] * len(records)) + ") v(" + ",".join(FIELDS) + ")", [item for row in records for item in row])
    con.execute("CREATE INDEX runs_id ON runs(id)"); con.close()


def decimate(rows: list[dict[str, Any]], maximum: int = 600) -> list[dict[str, Any]]:
    return [rows[index] for index in event_preserving_indices(rows, max_rows=maximum)]


def bundle_trace(path: str) -> dict[str, Any]:
    rows, coords, _ = load_bundle(path); rows = decimate(enrich_rows(rows, coords, force_threshold=.01))
    return {key: [row.get(key) for row in rows] for key in ("evaluation", "force_max", "n_neg", "lambda1", "lambda2", "lambda3", "energy_from_start", "step_cart_rms", "max_atom_displacement", "distance_to_terminal", "distance_to_labelled_ts", "lambda2_gate", "effective_gate", "activity_fraction", "lowest_reflection")}


def sella_trace(path: str) -> dict[str, Any]:
    with np.load(path) as a:
        coords = a["coordinates"]; terminal = coords[-1]
        step = np.zeros(len(coords)); step[1:] = np.sqrt(np.mean(np.sum((coords[1:] - coords[:-1]) ** 2, axis=2), axis=1))
        distance = [kabsch_rmsd(xyz, terminal) for xyz in coords]; energy = a["energy"] - a["energy"][0]
        return {"evaluation":a["evaluation"].tolist(), "force_max":a["force_max"].tolist(), "n_neg":a["n_neg"].tolist(), "lambda1":a["lambda1"].tolist(), "lambda2":a["lambda2"].tolist(), "lambda3":a["lambda3"].tolist(), "energy_from_start":energy.tolist(), "step_cart_rms":step.tolist(), "distance_to_terminal":distance}


def regular_trace(path: str) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist(); coords = np.asarray([row["coords_flat"] for row in rows]).reshape(len(rows), -1, 3); terminal = coords[-1]
    spectrum = [row.get("bottom_spectrum") or [] for row in rows]
    return {"evaluation":[row["step"] for row in rows], "force_max":[row["force_max"] for row in rows], "n_neg":[row["n_neg"] for row in rows], "lambda1":[value[0] if value else None for value in spectrum], "lambda2":[value[1] if len(value)>1 else None for value in spectrum], "lambda3":[value[2] if len(value)>2 else None for value in spectrum], "energy_from_start":[row["energy"]-rows[0]["energy"] for row in rows], "step_cart_rms":[row.get("disp_from_last") for row in rows], "distance_to_terminal":[kabsch_rmsd(xyz,terminal) for xyz in coords], "distance_to_labelled_ts":[row.get("dist_to_known_ts") for row in rows], "mode_overlap":[row.get("mode_overlap") for row in rows], "grad_v0_overlap":[row.get("grad_v0_overlap") for row in rows]}


def load_trace(kind: str, path: str) -> dict[str, Any]:
    if not path: return {}
    return {"bundle":bundle_trace, "sella_npz":sella_trace, "regular_parquet":regular_trace}[kind](path)


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite diagnostics with standards-compliant nulls."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@lru_cache(maxsize=64)
def cached_trace(kind: str, path: str) -> dict[str, Any]:
    """Cache decoded/decimated traces; files are immutable campaign artifacts."""
    return json_safe(load_trace(kind, path))


PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GADplus Explorer</title><script src="/plotly.min.js"></script><style>
:root{--bg:#f4f6fa;--card:#fff;--ink:#182230;--muted:#667085;--blue:#2563eb;--blue2:#dbeafe;--orange:#e76f00;--orange2:#ffedd5;--line:#d8dee9;--good:#087443;--bad:#b42318}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px Inter,ui-sans-serif,system-ui,sans-serif}.top{padding:18px 28px;background:#111827;color:#fff;display:flex;align-items:center;justify-content:space-between}.top h1{font-size:20px;margin:0}.top span{color:#b8c0d4}.wrap{max-width:1680px;margin:auto;padding:18px 26px 60px}.card{background:var(--card);border:1px solid var(--line);border-radius:11px;box-shadow:0 1px 3px #1018280c;margin-bottom:14px}.section-title{padding:15px 16px 7px;display:flex;align-items:end;justify-content:space-between;gap:16px}.section-title h2{font-size:16px;margin:0}.section-title p{color:var(--muted);margin:4px 0 0}.tabs{display:flex;gap:6px;padding:5px;background:#e9edf4;border-radius:9px}.tab{border:0;border-radius:6px;padding:8px 18px;background:transparent;font-weight:650;color:#475467;cursor:pointer}.tab.active{background:white;color:#175cd3;box-shadow:0 1px 3px #1018281f}button,input,select{font:inherit}.aggregate-wrap,.tablebox{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:9px 11px;border-bottom:1px solid #edf0f5;text-align:left}th{background:#f0f3f8;color:#475467;font-size:11px;text-transform:uppercase;letter-spacing:.035em;position:sticky;top:0;z-index:1}td.metric{font-variant-numeric:tabular-nums}.method{font-weight:700}.yes{color:var(--good)}.no{color:var(--bad)}.muted{color:var(--muted)}.examples{padding:10px 15px 16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.example-card{border:1px solid var(--line);border-radius:9px;padding:12px;background:#fbfcfe}.example-card h3{margin:0 0 9px;font-size:14px}.example-actions{display:grid;grid-template-columns:1fr 1fr;gap:7px}.example-btn{border:1px solid var(--line);border-radius:7px;padding:9px;background:#fff;text-align:left;cursor:pointer}.example-btn:hover:not(:disabled){border-color:#84adff;background:#f5f8ff}.example-btn:disabled{opacity:.45;cursor:not-allowed}.example-btn.good{border-left:4px solid var(--good)}.example-btn.bad{border-left:4px solid var(--bad)}.example-btn b,.example-btn small{display:block}.example-btn small{color:var(--muted);margin-top:2px}.slots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.slot{padding:12px 14px;border-left:5px solid var(--blue);min-height:78px}.slot.b{border-color:var(--orange)}.slot strong{font-size:14px}.slot code{display:block;color:var(--muted);margin-top:5px;white-space:normal}.clear{float:right;border:0;background:none;color:var(--muted);cursor:pointer}.plot-status{padding:24px;text-align:center;color:var(--muted)}.plot-status.error{color:var(--bad);background:#fff5f4}.plots{display:block}.plot{height:375px;width:100%;margin-bottom:12px;background:#fff;border:1px solid var(--line);border-radius:9px}.browser summary{cursor:pointer;padding:14px 16px;font-weight:700;font-size:14px}.browser-note{padding:0 16px 10px;color:var(--muted)}.filters{padding:0 12px 12px;display:grid;grid-template-columns:2fr repeat(4,1fr);gap:8px}.filters input,.filters select,.pager button{border:1px solid var(--line);border-radius:6px;padding:8px;background:white}.tablebox{height:390px}.tablebox tbody tr:hover{background:#eef5ff;cursor:pointer}.tablebox tr.selA{background:var(--blue2)}.tablebox tr.selB{background:var(--orange2)}.stats{padding:0 12px 9px;color:var(--muted)}.pager{display:flex;gap:8px;align-items:center;padding:9px}.spacer{margin-left:auto}@media(max-width:900px){.wrap{padding:12px}.slots{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}.top span{display:none}}
</style></head><body>
<header class="top"><h1>GADplus trajectory explorer</h1><span>Fast local index · curated diagnostics · A/B mechanism comparison</span></header>
<main class="wrap">
  <section class="card">
    <div class="section-title"><div><h2>Method overview</h2><p>All campaign outcomes, grouped by noise. Click a surface tab to switch datasets.</p></div><div class="tabs"><button class="tab active" data-surface="gxtb">g-xTB</button><button class="tab" data-surface="lj7">LJ7</button></div></div>
    <div class="aggregate-wrap"><table><thead><tr><th>Method</th><th>Noise</th><th>Starts</th><th>Local TS</th><th>Strict TS</th><th>IRC endpoints</th><th>IRC/native event</th><th>End index 0</th><th>End index &gt;1</th><th>Median evals</th></tr></thead><tbody id="aggregateBody"><tr><td colspan="10">Loading…</td></tr></tbody></table></div>
  </section>
  <section class="card">
    <div class="section-title"><div><h2>Representative trajectories</h2><p>Failures are shown only here, as deliberate diagnostic examples; they never enter the raw browser.</p></div></div>
    <div id="examples" class="examples"><div class="muted">Loading examples…</div></div>
  </section>
  <section class="slots"><div id="slotA" class="card slot"><strong>A — choose an example or successful run</strong></div><div id="slotB" class="card slot b"><strong>B — choose another trace to compare</strong></div></section>
  <div id="plotStatus" class="card plot-status">Choose one or two trajectories. Full diagnostics render one graph per row.</div><div id="plots" class="plots"></div>
  <details class="card browser"><summary>Successful-run browser <span class="muted">(secondary)</span></summary><div class="browser-note">Only successful runs with saved traces appear here. Use the curated buttons above to inspect failures.</div>
    <div class="filters"><input id="search" placeholder="Search ID, formula, reaction…"><select id="surface"><option value="">All surfaces</option></select><select id="method"><option value="">All methods</option></select><select id="noise"><option value="">All noise</option></select><select id="nneg"><option value="">Any final index</option><option>0</option><option>1</option><option>2+</option></select></div><div id="stats" class="stats"></div>
    <div class="tablebox"><table><thead id="head"></thead><tbody id="runBody"></tbody></table></div><div class="pager"><button id="prev">← Previous</button><span id="page"></span><button id="next">Next →</button><span class="spacer">Click a row to add it to A/B</span></div>
  </details>
</main><script>
const labels={competitive_subspace:'CS²-GAD',competitive:'Competitive GAD',regular_gad:'Regular GAD',sella:'Sella',ordinary_gad:'Regular GAD',hard_gate:'Hard λ₂ gate',historical_lambda2:'Smooth λ₂ gate',intrinsic:'Pointwise intrinsic GAD'};
const methodOrder=['regular_gad','ordinary_gad','hard_gate','historical_lambda2','intrinsic','competitive','competitive_subspace','sella'];
const methodLabel=x=>labels[x]||x, q=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let activeSurface='gxtb', sort='final_n_neg',dir='asc',offset=0,limit=100,total=0,A=null,B=null,rows=[],traceRequest=null;
const columns=[['surface','Surface'],['method','Method'],['panel','Panel'],['noise','Noise'],['sample_id','Sample'],['formula','Formula'],['initial_n_neg','Start idx'],['final_n_neg','End idx'],['final_force','Final fmax'],['evaluations','Evals'],['strict_ts','Strict'],['endpoint_minima','2 endpoints'],['native_topology','Native/event']];
function fmt(v,k){if(v===null||v===undefined)return '—';if(typeof v==='boolean')return '<span class="'+(v?'yes':'no')+'">'+(v?'✓':'—')+'</span>';if(k==='method')return esc(methodLabel(v));if(k==='final_force'&&typeof v==='number')return v.toExponential(2);return esc(v)}
function ratio(n,d){return '<b>'+n+'/'+d+'</b> <span class="muted">('+(d?100*n/d:0).toFixed(1)+'%)</span>'}
async function getJSON(url,signal){let r=await fetch(url,{signal});if(!r.ok)throw new Error(r.status+' '+r.statusText);return r.json()}
async function loadDashboard(){q('aggregateBody').innerHTML='<tr><td colspan="10">Loading…</td></tr>';q('examples').innerHTML='<div class="muted">Loading examples…</div>';try{let [a,e]=await Promise.all([getJSON('/api/aggregates?surface='+activeSurface),getJSON('/api/examples?surface='+activeSurface)]);renderAggregates(a.rows);renderExamples(e.rows)}catch(e){q('aggregateBody').innerHTML='<tr><td colspan="10" class="no">'+esc(e.message)+'</td></tr>';q('examples').innerHTML='<div class="no">'+esc(e.message)+'</div>'}}
function renderAggregates(data){let ordered=[...data].sort((a,b)=>a.noise-b.noise||methodOrder.indexOf(a.method)-methodOrder.indexOf(b.method));q('aggregateBody').innerHTML=ordered.length?ordered.map(r=>'<tr><td class="method">'+esc(methodLabel(r.method))+'</td><td>'+r.noise+' '+esc(r.noise_unit)+'</td><td class="metric">'+r.starts+'</td><td class="metric">'+ratio(r.local_count,r.starts)+'</td><td class="metric">'+ratio(r.strict_count,r.starts)+'</td><td class="metric">'+ratio(r.endpoint_count,r.starts)+'</td><td class="metric">'+ratio(r.native_count,r.starts)+'</td><td class="metric">'+ratio(r.index0_count,r.starts)+'</td><td class="metric">'+ratio(r.high_index_count,r.starts)+'</td><td class="metric">'+Math.round(r.median_evaluations)+'</td></tr>').join(''):'<tr><td colspan="10">No indexed runs.</td></tr>'}
function renderExamples(data){if(!data.length){q('examples').innerHTML='<div class="muted">This historical LJ7 sweep contains aggregate outcomes but no saved per-step traces. No trajectory is fabricated.</div>';return}let groups={};for(let r of data)(groups[r.method]??={})[r.local_ts?'success':'failure']=r;let keys=Object.keys(groups).sort((a,b)=>methodOrder.indexOf(a)-methodOrder.indexOf(b));q('examples').innerHTML=keys.map(m=>{let g=groups[m];return '<article class="example-card"><h3>'+esc(methodLabel(m))+'</h3><div class="example-actions">'+exampleButton(g.success,true)+exampleButton(g.failure,false)+'</div></article>'}).join('');document.querySelectorAll('.example-btn[data-id]').forEach(b=>b.onclick=()=>select(JSON.parse(decodeURIComponent(b.dataset.row))))}
function exampleButton(r,success){if(!r)return '<button class="example-btn '+(success?'good':'bad')+'" disabled><b>'+(success?'Success':'Failure')+'</b><small>No saved example</small></button>';let detail='noise '+r.noise+r.noise_unit+' · sample '+r.sample_id+(success?'':' · index '+r.final_n_neg);return '<button class="example-btn '+(success?'good':'bad')+'" data-id="'+esc(r.id)+'" data-row="'+encodeURIComponent(JSON.stringify(r))+'"><b>'+(success?'Success':'Failure')+'</b><small>'+esc(detail)+'</small></button>'}
function slot(el,r,label){el.innerHTML=r?'<button class="clear" data-slot="'+label+'">Clear</button><strong>'+label+' · '+esc(methodLabel(r.method))+' · '+(r.local_ts?'<span class="yes">success</span>':'<span class="no">failure</span>')+'</strong><div>noise '+r.noise+r.noise_unit+' · sample '+r.sample_id+' · final index '+r.final_n_neg+' · fmax '+fmt(r.final_force,'final_force')+'</div><code>'+esc(r.hparams)+'</code>':'<strong>'+label+' — choose '+(label==='A'?'an example or successful run':'another trace to compare')+'</strong>';let b=el.querySelector('.clear');if(b)b.onclick=()=>{if(label==='A')A=null;else B=null;updateSelection()}}
function select(r){if(A&&A.id===r.id)A=null;else if(B&&B.id===r.id)B=null;else if(!A)A=r;else if(!B)B=r;else{A=B;B=r}updateSelection()}
function updateSelection(){slot(q('slotA'),A,'A');slot(q('slotB'),B,'B');renderRuns();plot()}
async function plot(){let selected=[A,B].filter(Boolean);if(traceRequest)traceRequest.abort();for(let el of document.querySelectorAll('.plot'))try{Plotly.purge(el)}catch(_){}q('plots').innerHTML='';q('plotStatus').className='card plot-status';if(!selected.length){q('plotStatus').style.display='block';q('plotStatus').textContent='Choose one or two trajectories. Full diagnostics render one graph per row.';return}traceRequest=new AbortController();q('plotStatus').style.display='block';q('plotStatus').textContent='Loading '+selected.length+' selected trace'+(selected.length>1?'s':'')+'…';let started=performance.now();try{let d=await getJSON('/api/traces?ids='+selected.map(x=>encodeURIComponent(x.id)).join(','),traceRequest.signal);let errors=d.filter(x=>x.error).map(x=>x.error);if(errors.length)throw new Error(errors.join(' · '));if(d.some(x=>!x.trace||!(x.trace.evaluation||[]).length))throw new Error('No saved trajectory samples for this run.');q('plotStatus').textContent='Rendering diagnostics…';await drawPlots(d,selected);q('plotStatus').style.display='none';q('plotStatus').textContent='Loaded in '+Math.round(performance.now()-started)+' ms'}catch(e){if(e.name==='AbortError')return;q('plotStatus').style.display='block';q('plotStatus').className='card plot-status error';q('plotStatus').textContent='Could not load trace: '+e.message}}
function traces(d,selected,key,transform){let fn=transform||function(x){return x};return d.map((z,i)=>({x:z.trace.evaluation||[],y:(z.trace[key]||[]).map(v=>v===null||v===undefined?null:fn(v)),name:(i?'B · ':'A · ')+methodLabel(selected[i].method),type:'scattergl',mode:'lines',line:{color:i?'#e76f00':'#2563eb',width:2}})).filter(x=>x.x.length&&x.y.some(v=>v!==null&&v!==undefined))}
async function drawPlots(d,s){let specs=[['force_max','Force max',true],['spectrum','Signed lowest eigenvalues',false],['spectrum_abs','Lowest-eigenvalue magnitudes',true],['n_neg','Projected Morse index',false],['step_cart_rms','Per-step displacement',true],['distance','Hindsight closeness to TS / terminal',true],['energy_from_start','Energy from start',false],['mechanism','Method-specific dynamics',false]],panels=[];for(let spec of specs){let key=spec[0],t=[];if(key==='spectrum'||key==='spectrum_abs'){let fn=key==='spectrum_abs'?Math.abs:x=>x;t=['lambda1','lambda2','lambda3'].flatMap((k,j)=>traces(d,s,k,fn).map(x=>({...x,name:x.name+' · '+(key==='spectrum_abs'?'|λ':'λ')+(j+1)+(key==='spectrum_abs'?'|':''),line:{...x.line,dash:['solid','dash','dot'][j]}})))}else if(key==='distance')t=['distance_to_terminal','distance_to_labelled_ts'].flatMap(k=>traces(d,s,k).map(x=>({...x,name:x.name+' · '+(k==='distance_to_terminal'?'terminal':'labelled TS')})));else if(key==='mechanism')t=['effective_gate','lambda2_gate','activity_fraction','lowest_reflection','mode_overlap','grad_v0_overlap'].flatMap(k=>traces(d,s,k).map(x=>({...x,name:x.name+' · '+k})));else t=traces(d,s,key);if(t.length)panels.push({spec,traces:t})}if(!panels.length)throw new Error('Trace has no plottable diagnostic channels.');q('plots').innerHTML=panels.map((_,i)=>'<div class="plot" id="p'+i+'"></div>').join('');for(let i=0;i<panels.length;i++){let p=panels[i];await Plotly.newPlot('p'+i,p.traces,{title:{text:p.spec[1],font:{size:14}},margin:{l:62,r:18,t:42,b:45},hovermode:'x unified',legend:{orientation:'h',y:1.14},xaxis:{title:'evaluation',rangeslider:{visible:false}},yaxis:{type:p.spec[2]?'log':'linear',automargin:true},shapes:p.spec[0]==='force_max'?[{type:'line',x0:0,x1:1,xref:'paper',y0:.01,y1:.01,line:{dash:'dot',color:'#555'}}]:[]},{responsive:true,displaylogo:false,scrollZoom:true})}}
async function options(){let o=await getJSON('/api/options');for(let k of ['surface','method','noise'])for(let v of o[k])q(k).insertAdjacentHTML('beforeend','<option value="'+esc(v)+'">'+(k==='method'?esc(methodLabel(v)):esc(v))+'</option>')}
const filters=['search','surface','method','noise','nneg'];function params(){let p=new URLSearchParams({limit,offset,sort,dir});for(let k of filters)if(q(k).value)p.set(k,q(k).value);return p}
async function loadRuns(){q('runBody').innerHTML='<tr><td colspan="13">Loading…</td></tr>';try{let d=await getJSON('/api/runs?'+params());rows=d.rows;total=d.total;renderRuns()}catch(e){q('runBody').innerHTML='<tr><td colspan="13" class="no">'+esc(e.message)+'</td></tr>'}}
function renderRuns(){q('head').innerHTML='<tr>'+columns.map(([k,t])=>'<th data-k="'+k+'">'+t+(sort===k?(dir==='asc'?' ▲':' ▼'):'')+'</th>').join('')+'</tr>';q('runBody').innerHTML=rows.map(r=>'<tr data-id="'+esc(r.id)+'" class="'+(A&&A.id===r.id?'selA':B&&B.id===r.id?'selB':'')+'">'+columns.map(([k])=>'<td>'+fmt(r[k],k)+'</td>').join('')+'</tr>').join('');q('stats').textContent=total.toLocaleString()+' successful runs with saved trajectories';q('page').textContent=total?((offset+1)+'–'+Math.min(offset+limit,total)+' of '+total):'0 runs';document.querySelectorAll('#head th[data-k]').forEach(e=>e.onclick=()=>{let k=e.dataset.k;if(sort===k)dir=dir==='asc'?'desc':'asc';else{sort=k;dir='asc'}offset=0;loadRuns()});document.querySelectorAll('#runBody tr[data-id]').forEach(e=>e.onclick=()=>select(rows.find(r=>r.id===e.dataset.id)))}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{activeSurface=b.dataset.surface;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));loadDashboard()});for(let k of filters)q(k).addEventListener(k==='search'?'input':'change',()=>{offset=0;clearTimeout(window.filterTimer);window.filterTimer=setTimeout(loadRuns,150)});q('prev').onclick=()=>{offset=Math.max(0,offset-limit);loadRuns()};q('next').onclick=()=>{if(offset+limit<total){offset+=limit;loadRuns()}};
loadDashboard();options().then(loadRuns);
</script></body></html>'''


def create_app(database: Path) -> Flask:
    app = Flask(__name__)
    def connection(): return duckdb.connect(str(database), read_only=True)
    @app.get("/")
    def home(): return PAGE
    @app.get("/plotly.min.js")
    def plotly_js(): return Response(get_plotlyjs(), mimetype="application/javascript")
    @app.get("/api/options")
    def options():
        con=connection(); result={key:[row[0] for row in con.execute(f"SELECT DISTINCT {key} FROM runs ORDER BY {key}").fetchall()] for key in ("surface","method","noise")};con.close();return jsonify(result)
    @app.get("/api/aggregates")
    def aggregates():
        surface = "analytic reduced LJ7" if request.args.get("surface") == "lj7" else "Transition1x / g-xTB"
        con = connection()
        cur = con.execute(
            """SELECT method, noise, noise_unit, count(*) AS starts,
                      sum(CAST(local_ts AS INTEGER)) AS local_count,
                      sum(CAST(strict_ts AS INTEGER)) AS strict_count,
                      sum(CAST(endpoint_minima AS INTEGER)) AS endpoint_count,
                      sum(CAST(native_topology AS INTEGER)) AS native_count,
                      sum(CAST(final_n_neg=0 AS INTEGER)) AS index0_count,
                      sum(CAST(final_n_neg>1 AS INTEGER)) AS high_index_count,
                      median(evaluations) AS median_evaluations,
                      sum(CAST(has_trace AS INTEGER)) AS saved_traces
               FROM runs WHERE surface=?
               GROUP BY method, noise, noise_unit ORDER BY noise, method""",
            [surface],
        )
        rows = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
        con.close()
        return jsonify({"surface": surface, "rows": rows})
    @app.get("/api/examples")
    def examples():
        surface = "analytic reduced LJ7" if request.args.get("surface") == "lj7" else "Transition1x / g-xTB"
        con = connection()
        cur = con.execute(
            """SELECT * EXCLUDE(example_rank) FROM (
                   SELECT *, row_number() OVER (
                       PARTITION BY method, local_ts
                       ORDER BY noise DESC, (final_n_neg>=0) DESC,
                                evaluations DESC, sample_id
                   ) AS example_rank
                   FROM runs WHERE surface=? AND has_trace
               ) WHERE example_rank=1 ORDER BY method, local_ts DESC""",
            [surface],
        )
        rows = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
        con.close()
        return jsonify({"surface": surface, "rows": rows})
    @app.get("/api/runs")
    def runs():
        # The secondary browser is deliberately a clean success-only trace browser.
        # Failures remain represented in aggregates and in curated diagnostic examples.
        clauses=["local_ts", "has_trace"]; values=[]
        for key in ("surface","method","noise"):
            if request.args.get(key): clauses.append(f"{key}=?"); values.append(request.args[key])
        if request.args.get("search"):
            clauses.append("lower(id || ' ' || formula || ' ' || rxn) LIKE ?"); values.append("%"+request.args["search"].lower()+"%")
        nneg=request.args.get("nneg")
        if nneg: clauses.append("final_n_neg>=2" if nneg=="2+" else "final_n_neg=?"); values.extend([] if nneg=="2+" else [int(nneg)])
        outcome=request.args.get("outcome"); mapping={"local":"local_ts","strict":"strict_ts","endpoint":"endpoint_minima","native":"native_topology"}
        if outcome: clauses.append(mapping[outcome])
        where=" WHERE "+" AND ".join(clauses) if clauses else ""; sort=request.args.get("sort","final_n_neg"); sort=sort if sort in SORTABLE else "final_n_neg"; direction="DESC" if request.args.get("dir")=="desc" else "ASC"; limit=min(int(request.args.get("limit",100)),250); offset=max(int(request.args.get("offset",0)),0)
        con=connection(); total=con.execute("SELECT count(*) FROM runs"+where,values).fetchone()[0]; cur=con.execute(f"SELECT * FROM runs{where} ORDER BY {sort} {direction}, id LIMIT ? OFFSET ?",values+[limit,offset]); rows=[dict(zip([d[0] for d in cur.description],row)) for row in cur.fetchall()];con.close();return jsonify({"total":total,"rows":rows})
    @app.get("/api/traces")
    def traces():
        ids=request.args.get("ids","").split(",")[:2]; con=connection(); result=[]
        for ident in ids:
            cur=con.execute("SELECT * FROM runs WHERE id=?",[ident]); row=cur.fetchone()
            if row is None:
                result.append({"error":f"Unknown run: {ident}","trace":{}}); continue
            meta=dict(zip([d[0] for d in cur.description],row))
            try:
                trace=cached_trace(meta["trace_kind"],meta["trace_path"]) if meta["has_trace"] else {}
                result.append({"meta":meta,"trace":trace})
            except Exception as exc:
                result.append({"meta":meta,"trace":{},"error":f"{type(exc).__name__}: {exc}"})
        con.close();return jsonify(result)
    return app


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--manifest",type=Path,default=Path("experiments/t1x_gxtb_matched_noise_grid_manifest.json"));parser.add_argument("--lj-results",type=Path,default=Path("/scratch/memoozd/gadplus/runs/lj-method-progression-1946071/all_results.json"));parser.add_argument("--database",type=Path,default=Path("/scratch/memoozd/gadplus/analysis/trajectory-explorer/index.duckdb"));parser.add_argument("--port",type=int,default=8767);parser.add_argument("--rebuild",action="store_true");args=parser.parse_args()
    if args.rebuild or not args.database.exists(): build_index(args.manifest,args.lj_results,args.database)
    create_app(args.database).run(host="127.0.0.1",port=args.port,debug=False,threaded=True)


if __name__=="__main__": main()
