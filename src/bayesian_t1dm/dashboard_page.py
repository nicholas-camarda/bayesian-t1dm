"""The front page: one self-contained HTML file (no network, no libraries) with tabs and interactive charts."""

from __future__ import annotations

from .dashboard import to_script_json

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glucose dashboard</title>
<style>
:root{color-scheme:light;--bg:#f4f4f1;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#76756f;--line:#dcdbd5;--grid:#ebeae5;
--blue:#2a78d6;--orange:#eb6834;--red:#e34948;--band:#cde2fb;--band2:#9ec5f4;--ok:#1b7f3a;--okbg:#e3f3e7;--warn:#8a5a00;--warnbg:#fdf0d0;--act:#a52422;--actbg:#fbe3e2;--focus:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#97968d;--line:#3a3a37;--grid:#2a2a28;
--blue:#3987e5;--orange:#d95926;--red:#e66767;--band:#17406f;--band2:#215a9b;--ok:#6fd08c;--okbg:#18301f;--warn:#f0c15a;--warnbg:#38300f;--act:#f29a98;--actbg:#3d1a19;--focus:#6da7ec}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--ink3:#97968d;--line:#3a3a37;--grid:#2a2a28;
--blue:#3987e5;--orange:#d95926;--red:#e66767;--band:#17406f;--band2:#215a9b;--ok:#6fd08c;--okbg:#18301f;--warn:#f0c15a;--warnbg:#38300f;--act:#f29a98;--actbg:#3d1a19;--focus:#6da7ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}
.top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:12px 0 4px;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-weight:650}.sub{color:var(--ink3);font-size:13px}
nav{display:flex;gap:2px;overflow-x:auto;padding-bottom:0}
nav button{background:none;border:0;border-bottom:3px solid transparent;color:var(--ink2);font:inherit;padding:8px 12px;cursor:pointer;white-space:nowrap}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--blue);font-weight:600}
nav button:focus-visible,.btn:focus-visible,summary:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
main{padding:18px 0 48px}section.tab{display:none}section.tab.on{display:block}
h2{font-size:17px;margin:26px 0 4px}h2:first-child{margin-top:0}.lead{color:var(--ink2);margin:0 0 12px;max-width:70ch}
.grid{display:grid;gap:12px}.g4{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.g2{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink3);font-size:13px}.tile .v{font-size:28px;font-weight:650;line-height:1.15;font-variant-numeric:tabular-nums}.tile .n{color:var(--ink2);font-size:13px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:650;border-radius:999px;padding:2px 9px}
.b-act{background:var(--actbg);color:var(--act)}.b-watch{background:var(--warnbg);color:var(--warn)}.b-ok{background:var(--okbg);color:var(--ok)}
.problem{display:flex;gap:12px;align-items:flex-start}.problem h3{margin:0 0 2px;font-size:15px}.problem p{margin:0;color:var(--ink2)}
.btn{background:none;border:1px solid var(--line);color:var(--ink);border-radius:8px;padding:5px 10px;font:inherit;font-size:13px;cursor:pointer;margin-top:8px}.btn:hover{border-color:var(--blue)}
svg text{font:12px system-ui,sans-serif;fill:var(--ink2)}.chart{width:100%;display:block}.axis path,.axis line{stroke:var(--line)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:4px 0 6px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;vertical-align:top}
th:first-child,td:first-child,td.l,th.l{text-align:left}td:first-child{white-space:nowrap}th{color:var(--ink3);font-weight:600}.scroll{overflow-x:auto}
#tip{position:fixed;pointer-events:none;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.18);display:none;z-index:20;max-width:260px}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.seg{display:flex;gap:6px;margin:6px 0}.seg button{border:1px solid var(--line);background:var(--card);color:var(--ink2);border-radius:8px;padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}.seg button[aria-pressed="true"]{border-color:var(--blue);color:var(--ink);font-weight:600}
.note{color:var(--ink3);font-size:12.5px;margin-top:6px;max-width:80ch}.chip{border-radius:6px;padding:1px 7px;font-size:12px;background:var(--grid)}
@media (max-width:520px){.g2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="wrap">
<div class="top"><h1>Glucose dashboard</h1><div class="sub" id="meta"></div></div>
<nav role="tablist" id="nav"></nav>
</div></header>
<main class="wrap">
<section class="tab" id="t-today"></section>
<section class="tab" id="t-glucose"></section>
<section class="tab" id="t-meals"></section>
<section class="tab" id="t-basal"></section>
<section class="tab" id="t-settings"></section>
<section class="tab" id="t-about"></section>
</main>
<div id="tip" role="status"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s, r=document) => r.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const f1 = v => v == null ? 'n/a' : v.toFixed(1), f0 = v => v == null ? 'n/a' : Math.round(v).toString(), pct = v => v == null ? 'n/a' : v.toFixed(1) + '%';
const hm = m => String(Math.floor(m/60)).padStart(2,'0') + ':' + String(m%60).padStart(2,'0');
const SETTING = {carb_ratio_g_per_unit:'I:C (g/U)', isf_mgdl_per_unit:'Correction factor', basal_units_per_hour:'Basal (U/h)', target_bg_mgdl:'Target', 'I:C':'I:C (g/U)'};
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : '';
const when = t => t ? new Date(t).toLocaleString('en',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}) : 'proposed';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tip = $('#tip');
function showTip(ev, html){ tip.innerHTML = html; tip.style.display='block'; const w=tip.offsetWidth, h=tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14; if (x + w > innerWidth - 8) x = ev.clientX - w - 14; if (y + h > innerHeight - 8) y = ev.clientY - h - 14; tip.style.left = x+'px'; tip.style.top = y+'px'; }
function hideTip(){ tip.style.display='none'; }
function el(tag, attrs={}, parent){ const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, attrs={}){ const t = el('text', {x, y, ...attrs}, parent); t.textContent = s; return t; }
const scale = (d0,d1,r0,r1) => v => r0 + (v-d0)/(d1-d0)*(r1-r0);

/* ---- a chart host that redraws when its width changes ---- */
function chart(host, height, draw){
  host.innerHTML = ''; const svg = el('svg', {class:'chart', role:'img'}); host.appendChild(svg);
  const run = () => { const w = Math.max(280, host.clientWidth); svg.setAttribute('viewBox', `0 0 ${w} ${height}`); svg.setAttribute('height', height); svg.innerHTML=''; draw(svg, w, height); };
  run(); new ResizeObserver(() => { if (host.clientWidth !== host._w){ host._w = host.clientWidth; run(); } }).observe(host);
  return svg;
}
function axes(svg, m, w, h, xs, ys, xticks, yticks, xfmt, yfmt, ylab){
  yticks.forEach(t => { el('line',{x1:m.l,x2:w-m.r,y1:ys(t),y2:ys(t),stroke:css('--grid')},svg); txt(svg,m.l-6,ys(t)+4,yfmt(t),{'text-anchor':'end'}); });
  xticks.forEach(t => txt(svg,xs(t),h-m.b+16,xfmt(t),{'text-anchor':'middle'}));
  el('line',{x1:m.l,x2:w-m.r,y1:h-m.b,y2:h-m.b,stroke:css('--line')},svg);
  if (ylab) txt(svg, 4, 12, ylab, {'text-anchor':'start'});
}

/* ---- tabs ---- */
const TABS = [['today','Today'],['glucose','Glucose'],['meals','Meals (I:C)'],['basal','Basal'],['settings','Changes'],['about','About']];
const nav = $('#nav');
TABS.forEach(([id,label]) => { const b = document.createElement('button'); b.textContent = label; b.setAttribute('role','tab'); b.dataset.tab = id; b.onclick = () => go(id); nav.appendChild(b); });
function go(id){ TABS.forEach(([t]) => $('#t-'+t).classList.toggle('on', t===id)); nav.querySelectorAll('button').forEach(b => b.setAttribute('aria-selected', b.dataset.tab===id)); (RENDER[id]||(()=>{}))(); window.scrollTo(0,0); try { history.replaceState(null,'','#'+id); } catch (e) { /* a page opened from a restricted origin cannot change its address */ } }
$('#meta').textContent = `Data through ${D.as_of} · run ${D.run.run_id || 'manual'}`;

/* ---- Today ---- */
const LEVEL = {act:['Act','b-act','▲'], watch:['Watch','b-watch','●'], ok:['OK','b-ok','✓']};
function tile(k,v,n){ return `<div class="card tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`; }
function renderToday(){
  const b14 = D.baseline.find(r => r.window==='last 14 days') || {}, b90 = D.baseline.find(r => r.window==='last 90 days') || {};
  let h = `<h2>Where you are</h2><p class="lead">Consensus targets: more than 70% of the day in 70-180 mg/dL, under 4% below 70 and under 1% below 54.</p><div class="grid g4">`
    + tile('Time in range, last 14 days', pct(b14.time_in_range_pct), `90 days: ${pct(b90.time_in_range_pct)} · target 70%`)
    + tile('Below 70, last 90 days', pct(b90.time_below_70_pct), `below 54: ${(b90.time_below_54_pct ?? 0).toFixed(2)}% · limits 4% and 1%`)
    + tile('Above 180, last 90 days', pct(b90.time_above_180_pct), `above 250: ${pct(b90.time_above_250_pct)}`)
    + tile('Estimated A1c (GMI), 90 days', f1(b90.gmi_pct)+'%', `mean ${f0(b90.mean_glucose_mgdl)} mg/dL · variability ${f0(b90.cv_pct)}%`) + `</div>`;
  h += `<h2>What needs a look</h2><p class="lead">Safest first. Each one opens the chart behind it.</p><div class="grid">`;
  D.problems.forEach(p => { const [lab,cls,ic] = LEVEL[p.level]; h += `<div class="card problem"><span class="badge ${cls}">${ic} ${lab}</span><div><h3>${esc(p.title)}</h3><p>${esc(p.summary)}</p><button class="btn" data-go="${p.tab}">See the chart</button></div></div>`; });
  h += `</div>`;
  const pr = D.proposal || {};
  h += `<h2>Setting change</h2><div class="card"><span class="badge ${pr.status==='proposed'?'b-act':'b-watch'}">${pr.status==='proposed'?'▲ Proposed':'● No change now'}</span><p style="margin:8px 0 0">${esc(cap(pr.reason||''))}</p>`
     + `<button class="btn" data-go="settings">Change log and how changes are judged</button></div>`;
  $('#t-today').innerHTML = h; $('#t-today').querySelectorAll('[data-go]').forEach(b => b.onclick = () => go(b.dataset.go));
}

/* ---- Glucose ---- */
function renderGlucose(){
  const t = $('#t-glucose');
  t.innerHTML = `<h2>A typical day</h2><p class="lead">Glucose by time of day over the last 90 days. The line is the median, the dark band the middle half of readings, the light band the middle 80%. Segment lines mark where your pump's settings change.</p>
  <div class="card"><div class="legend"><span><i style="background:var(--blue)"></i>median</span><span><i style="background:var(--band2)"></i>25th to 75th percentile</span><span><i style="background:var(--band)"></i>10th to 90th percentile</span></div><div id="agp"></div></div>
  <h2>Time in range by segment</h2><p class="lead">Each pump segment has its own settings, so each gets its own row (last 90 days).</p><div class="card"><div class="legend"><span><i style="background:var(--red)"></i>below 70</span><span><i style="background:var(--blue)"></i>in range 70-180</span><span><i style="background:var(--orange)"></i>above 180</span></div><div id="segbars"></div><div class="scroll"><table id="segtable"></table></div></div>
  <h2>Day by day</h2><p class="lead">Time in range for each day, with a 14-day average. The dashed line is the 70% target.</p><div class="card"><div id="daily"></div><details><summary>Show the last 28 days as a table</summary><div class="scroll"><table id="dailytable"></table></div></details></div>
  <h2>When the long highs and the lows start</h2><p class="lead">Highs: above 250 mg/dL for 90 minutes or more. Lows: below 70 for 15 minutes or more. Last ${D.episodes.days} days.</p><div class="card"><div class="legend"><span><i style="background:var(--orange)"></i>highs (${D.episodes.high_total})</span><span><i style="background:var(--red)"></i>lows (${D.episodes.low_total})</span></div><div id="episodes"></div></div>`;
  agpChart($('#agp')); segBars($('#segbars')); segTable($('#segtable')); dailyChart($('#daily')); dailyTable($('#dailytable')); episodeChart($('#episodes'));
}
function agpChart(host){
  const m={l:40,r:10,t:10,b:26}; const a = D.agp;
  chart(host, 300, (svg,w,h) => {
    const xs=scale(0,1440,m.l,w-m.r), ys=scale(40,350,h-m.b,m.t);
    el('rect',{x:m.l,y:ys(180),width:w-m.l-m.r,height:ys(70)-ys(180),fill:css('--ok'),opacity:.10},svg);
    axes(svg,m,w,h,xs,ys,[0,360,720,1080],[70,100,150,180,250,300],hm,v=>v);
    [70,180].forEach(v => el('line',{x1:m.l,x2:w-m.r,y1:ys(v),y2:ys(v),stroke:css('--ok'),'stroke-dasharray':'4 4',opacity:.8},svg));
    const path = (k) => a.map((r,i)=> (i?'L':'M')+xs(r.minute)+' '+ys(r[k])).join('');
    const band = (lo,hi,fill) => el('path',{d:path(lo)+a.slice().reverse().map(r=>'L'+xs(r.minute)+' '+ys(r[hi])).join('')+'Z',fill},svg);
    band('p10','p90',css('--band')); band('p25','p75',css('--band2'));
    el('path',{d:path('p50'),fill:'none',stroke:css('--blue'),'stroke-width':2.2,'stroke-linejoin':'round'},svg);
    D.segments.forEach(s => { if (s.start>0){ el('line',{x1:xs(s.start),x2:xs(s.start),y1:m.t,y2:h-m.b,stroke:css('--ink3'),'stroke-dasharray':'2 4',opacity:.6},svg); } });
    const cross = el('line',{y1:m.t,y2:h-m.b,stroke:css('--ink'),opacity:0},svg), dot = el('circle',{r:4,fill:css('--blue'),stroke:css('--card'),'stroke-width':2,opacity:0},svg);
    const hit = el('rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b,fill:'transparent'},svg);
    hit.onmousemove = ev => { const r=svg.getBoundingClientRect(); const x=(ev.clientX-r.left)*(w/r.width); const minute=Math.max(0,Math.min(1425,Math.round((x-m.l)/(w-m.l-m.r)*1440/15)*15)); const row=a.find(q=>q.minute===minute); if(!row) return;
      cross.setAttribute('x1',xs(minute)); cross.setAttribute('x2',xs(minute)); cross.setAttribute('opacity',.35); dot.setAttribute('cx',xs(minute)); dot.setAttribute('cy',ys(row.p50)); dot.setAttribute('opacity',1);
      showTip(ev,`<b>${hm(minute)}</b><br>median ${f0(row.p50)} mg/dL<br>middle half ${f0(row.p25)}-${f0(row.p75)}<br>middle 80% ${f0(row.p10)}-${f0(row.p90)}`); };
    hit.onmouseleave = () => { cross.setAttribute('opacity',0); dot.setAttribute('opacity',0); hideTip(); };
  });
}
function segBars(host){
  const rows = D.parts_90; const m={l:96,r:12,t:6,b:6}, rh=30;
  chart(host, rows.length*rh+m.t+m.b, (svg,w,h) => {
    const xs = scale(0,100,m.l,w-m.r);
    rows.forEach((r,i) => { const y=m.t+i*rh; txt(svg,m.l-8,y+rh/2+4,r.day_part,{'text-anchor':'end'});
      const lo=r.time_below_70_pct, hi=r.time_above_180_pct, mid=100-lo-hi; let x=0;
      [[lo,'--red','below 70'],[mid,'--blue','in range'],[hi,'--orange','above 180']].forEach(([v,c,name]) => { if (v<=0) return;
        const rect = el('rect',{x:xs(x)+1,y:y+4,width:Math.max(0,xs(x+v)-xs(x)-2),height:rh-10,rx:3,fill:css(c)},svg); x+=v;
        rect.onmousemove = ev => showTip(ev,`<b>${r.day_part}</b><br>${name}: ${v.toFixed(1)}%`); rect.onmouseleave = hideTip; });
      if (mid>8) txt(svg,xs(lo+mid/2),y+rh/2+4,mid.toFixed(0)+'%',{'text-anchor':'middle',fill:'#fff',style:'fill:#fff;font-weight:600'}); });
  });
}
function segTable(tb){
  tb.innerHTML = `<tr><th class="l">Segment</th><th>I:C</th><th>CF</th><th>Basal U/h</th><th>In range 90d (14d)</th><th>Below 70</th><th>Below 54</th><th>Above 180</th><th class="l">Reading</th></tr>` +
   D.segments.map(s => `<tr><td class="l">${hm(s.start)}-${hm(s.end)}</td><td>${f1(s.icr_now)}</td><td>${f0(s.cf_now)}</td><td>${s.basal_now==null?'n/a':s.basal_now.toFixed(2)}</td><td>${pct(s.tir_90d)} (${pct(s.tir_14d)})</td><td>${pct(s.below_70_90d)}</td><td>${(s.below_54_90d??0).toFixed(2)}%</td><td>${pct(s.above_180_90d)}</td><td class="l">${esc(s.reading)}</td></tr>`).join('');
}
function dailyChart(host){
  const d = D.daily; if (!d.length) { host.textContent='No counted days.'; return; }
  const m={l:40,r:10,t:10,b:26};
  chart(host,260,(svg,w,h) => {
    const t0=new Date(d[0].date).getTime(), t1=new Date(d[d.length-1].date).getTime(); const xs=scale(t0,t1,m.l,w-m.r), ys=scale(0,100,h-m.b,m.t);
    const ticks=[]; const step=Math.ceil(d.length/6); for(let i=0;i<d.length;i+=step) ticks.push(new Date(d[i].date).getTime());
    axes(svg,m,w,h,xs,ys,ticks,[0,25,50,75,100],tk=>new Date(tk).toISOString().slice(5,10),v=>v+'%');
    el('line',{x1:m.l,x2:w-m.r,y1:ys(70),y2:ys(70),stroke:css('--ink'),'stroke-dasharray':'5 4',opacity:.6},svg); txt(svg,w-m.r,ys(70)-5,'70% target',{'text-anchor':'end'});
    d.forEach(r => el('circle',{cx:xs(new Date(r.date).getTime()),cy:ys(r.tir),r:2.4,fill:css('--blue'),opacity:.45},svg));
    el('path',{d:d.filter(r=>r.tir_14d!=null).map((r,i)=>(i?'L':'M')+xs(new Date(r.date).getTime())+' '+ys(r.tir_14d)).join(''),fill:'none',stroke:css('--blue'),'stroke-width':2.4,'stroke-linejoin':'round'},svg);
    const hit=el('rect',{x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b,fill:'transparent'},svg);
    hit.onmousemove=ev=>{const r=svg.getBoundingClientRect();const x=(ev.clientX-r.left)*(w/r.width);const tt=t0+(x-m.l)/(w-m.l-m.r)*(t1-t0);let best=d[0];d.forEach(q=>{if(Math.abs(new Date(q.date)-tt)<Math.abs(new Date(best.date)-tt))best=q;});
      showTip(ev,`<b>${best.date}</b><br>in range ${pct(best.tir)}<br>14-day average ${pct(best.tir_14d)}<br>below 70 ${pct(best.below_70)} · above 180 ${pct(best.above_180)}`);};
    hit.onmouseleave=hideTip;
  });
}
function dailyTable(tb){ tb.innerHTML = `<tr><th class="l">Date</th><th>In range</th><th>Below 70</th><th>Above 180</th><th>Mean mg/dL</th></tr>` + D.daily.slice(-28).reverse().map(r=>`<tr><td class="l">${r.date}</td><td>${pct(r.tir)}</td><td>${pct(r.below_70)}</td><td>${pct(r.above_180)}</td><td>${f0(r.mean)}</td></tr>`).join(''); }
function episodeChart(host){
  const m={l:30,r:10,t:10,b:26}; const hi=D.episodes.highs, lo=D.episodes.lows, mx=Math.max(1,...hi,...lo);
  chart(host,230,(svg,w,h)=>{ const xs=scale(0,24,m.l,w-m.r), ys=scale(0,mx,h-m.b,m.t), bw=(w-m.l-m.r)/24;
    axes(svg,m,w,h,xs,ys,[0,3,6,9,12,15,18,21],[0,Math.ceil(mx/2),mx],tk=>String(tk).padStart(2,'0')+':00',v=>v);
    for(let k=0;k<24;k++){ [[hi[k],'--orange',0,'highs'],[lo[k],'--red',1,'lows']].forEach(([v,c,j,name])=>{ if(!v) return; const x=xs(k)+2+j*((bw-4)/2), bwid=(bw-4)/2-1;
      const r=el('rect',{x,y:ys(v),width:Math.max(2,bwid),height:h-m.b-ys(v),rx:2,fill:css(c)},svg); r.onmousemove=ev=>showTip(ev,`<b>${String(k).padStart(2,'0')}:00-${String(k+1).padStart(2,'0')}:00</b><br>${v} ${name} started`); r.onmouseleave=hideTip; }); } });
}

/* ---- Meals (I:C) ---- */
let showAll = false;
function renderMeals(){
  const t=$('#t-meals');
  t.innerHTML = `<h2>Do your meals behave like your pump setting?</h2><p class="lead">For each meal the glucose trace says how many grams of carbohydrate it behaved like, given the I:C your pump used. A dot is the I:C the meals behaved like, with its 90% range; the vertical mark is what the pump was set to. When the mark sits outside the range, the setting does not fit those meals. Meals are pooled by the pump segment they fall in (breakfast, lunch and dinner together).</p>
  <div class="seg"><button id="only" aria-pressed="${!showAll}">Current profile, latest period</button><button id="all" aria-pressed="${showAll}">All periods</button></div>
  <div class="card"><div class="legend"><span><i style="background:var(--blue)"></i>I:C the meals behaved like (90% range)</span><span><i style="background:var(--ink)"></i>pump setting for those meals</span><span><i style="background:var(--orange);border-radius:50%"></i>setting the pump holds now, when changed since</span></div><div id="forest"></div><div class="scroll"><table id="evtable"></table></div>
  <p class="note">Smaller g/U = more insulin per gram. "Behaves like 5.4 against a setting of 6.8" means those meals needed more insulin than the pump gave. The ratio found absorbs how you count carbs, so it is the ratio that works with how you eat. Segments with fewer than 5 meals at a setting are not shown.</p></div>`;
  $('#only').onclick=()=>{showAll=false;renderMeals();}; $('#all').onclick=()=>{showAll=true;renderMeals();};
  let rows = D.evidence.filter(r=>r.n>=5); if(!showAll){ const latest={}; rows.forEach(r=>{ if(!latest[r.segment]||r.end>latest[r.segment].end) latest[r.segment]=r; }); rows=Object.values(latest); }
  rows.sort((a,b)=>a.segment.localeCompare(b.segment)||a.start.localeCompare(b.start));
  const m={l:158,r:16,t:8,b:44}, rh=28; const mon = d => new Date(d).toLocaleString('en',{month:'short',year:'2-digit'});
  chart($('#forest'), Math.max(80,rows.length*rh+m.t+m.b), (svg,w,h)=>{
    const lo=Math.min(4,...rows.map(r=>r.lower))-.3, hi=Math.max(9,...rows.map(r=>r.upper))+.3, xs=scale(lo,hi,m.l,w-m.r);
    for(let v=Math.ceil(lo);v<=hi;v++){ el('line',{x1:xs(v),x2:xs(v),y1:m.t,y2:h-m.b,stroke:css('--grid')},svg); txt(svg,xs(v),h-m.b+16,v,{'text-anchor':'middle'}); }
    txt(svg,(m.l+w-m.r)/2,h-3,'I:C, grams of carbohydrate per unit of insulin',{'text-anchor':'middle'});
    rows.forEach((r,i)=>{ const y=m.t+i*rh+rh/2; txt(svg,m.l-8,y+4,`${r.segment} · ${mon(r.start)} to ${mon(r.end)}`,{'text-anchor':'end'});
      const ln=el('line',{x1:xs(r.lower),x2:xs(r.upper),y1:y,y2:y,stroke:css('--blue'),'stroke-width':3,'stroke-linecap':'round'},svg); el('circle',{cx:xs(r.effective),cy:y,r:5,fill:css('--blue'),stroke:css('--card'),'stroke-width':2},svg);
      el('rect',{x:xs(r.programmed)-1.5,y:y-9,width:3,height:18,fill:css('--ink'),rx:1},svg);
      if (r.now!=null && Math.abs(r.now-r.programmed)>=0.05 && r.latest) el('circle',{cx:xs(r.now),cy:y,r:8,fill:'none',stroke:css('--orange'),'stroke-width':2.5},svg);
      const hit=el('rect',{x:m.l,y:y-rh/2,width:w-m.l-m.r,height:rh,fill:'transparent'},svg); hit.onmousemove=ev=>showTip(ev,`<b>${r.segment} segment</b><br>${r.start} to ${r.end} · ${r.n} meals<br>behaved like ${f1(r.effective)} g/U (${f1(r.lower)}-${f1(r.upper)})<br>pump setting ${f1(r.programmed)}${r.current?'<br><i>current setting</i>':''}`); hit.onmouseleave=hideTip; });
  });
  $('#evtable').innerHTML = `<tr><th class="l">Segment</th><th class="l">Period</th><th>Meals</th><th>Pump I:C then</th><th>Pump now</th><th>Behaved like</th><th>90% range</th><th class="l">Reading</th></tr>` + rows.map(r=>{ const out = r.programmed<r.lower||r.programmed>r.upper;
    return `<tr><td class="l">${r.segment}</td><td class="l">${mon(r.start)} to ${mon(r.end)}${r.current?' <span class="chip">current</span>':''}</td><td>${r.n}</td><td>${f1(r.programmed)}</td><td>${f1(r.now)}</td><td>${f1(r.effective)}</td><td>${f1(r.lower)}-${f1(r.upper)}</td><td class="l">${out?(r.effective<r.programmed?'needed more insulin':'needed less insulin'):'consistent with the setting'}${(r.latest&&r.now!=null&&Math.abs(r.now-r.programmed)>=0.05)?' · setting since changed':''}</td></tr>`; }).join('');
}

/* ---- Basal ---- */
function renderBasal(){
  const t=$('#t-basal'), s=D.controller.signals;
  t.innerHTML = `<h2>What Control-IQ does against your programmed basal</h2><p class="lead">Your pump records the rate it delivered and the rate your profile calls for. If it adds a lot of insulin on most nights and glucose is still high, the programmed basal is probably too low there. Time after meals, boluses, exercise and unlogged food is left out.</p>
  <div class="card"><div class="scroll"><table id="sig"></table></div><p class="note">Rule: the controller runs 25% or more above the profile on at least half of the last 28 clean days, mean glucose is still above 120 mg/dL, and lows are within limits. Fewer than 14 clean days gives no signal. Daytime segments have few clean days because meals take most of the day.</p></div>
  <h2>Overnight, night by night</h2><p class="lead">Extra insulin delivered above the profile, as a share of the profile's insulin (${D.controller.nightly.length} clean nights). The dashed line is 25%.</p><div class="card"><div class="legend"><span><i style="background:var(--orange)"></i>25% or more above the profile</span><span><i style="background:var(--blue)"></i>under 25%</span></div><div id="nightly"></div></div>
  <h2>Overnight glucose, night by night</h2><p class="lead">Mean glucose over the same clean hours. The shaded band is the sleep-mode target, 112-120 mg/dL.</p><div class="card"><div id="nightglu"></div></div>`;
  $('#sig').innerHTML = `<tr><th class="l">Segment</th><th>Clean days</th><th>Median extra</th><th>Days at 25%+</th><th>Mean glucose</th><th>Below 70</th><th class="l">Signal</th></tr>` + s.map(r=>`<tr><td class="l">${r.part}</td><td>${r.days}</td><td>${r.median_excess_fraction==null?'n/a':(r.median_excess_fraction*100>=0?'+':'')+(r.median_excess_fraction*100).toFixed(0)+'%'}</td><td>${r.share_over_x==null?'n/a':(r.share_over_x*100).toFixed(0)+'%'}</td><td>${f0(r.mean_glucose)}</td><td>${r.below_70_pct==null?'n/a':r.below_70_pct.toFixed(2)+'%'}</td><td class="l">${esc(r.signal)}</td></tr>`).join('');
  nightChart($('#nightly'),'excess'); nightChart($('#nightglu'),'glucose');
}
function nightChart(host,kind){
  const n=D.controller.nightly; if(!n.length){host.textContent='No rate records in the archive.';return;}
  const m={l:46,r:10,t:10,b:26};
  chart(host,230,(svg,w,h)=>{
    const t0=new Date(n[0].date).getTime(), t1=new Date(n[n.length-1].date).getTime(), xs=scale(t0,t1,m.l,w-m.r);
    const vals=n.map(r=>kind==='excess'?r.excess*100:r.glucose).filter(v=>v!=null); const lo=kind==='excess'?Math.min(-50,...vals):80, hi=kind==='excess'?Math.max(100,...vals):Math.max(200,...vals), ys=scale(lo,hi,h-m.b,m.t);
    const ticks=[]; const step=Math.ceil(n.length/6); for(let i=0;i<n.length;i+=step) ticks.push(new Date(n[i].date).getTime());
    const yt=kind==='excess'?[-50,0,25,50,100].filter(v=>v>=lo&&v<=hi):[100,120,150,200].filter(v=>v>=lo&&v<=hi);
    axes(svg,m,w,h,xs,ys,ticks,yt,tk=>new Date(tk).toISOString().slice(5,10),v=>kind==='excess'?v+'%':v);
    if(kind==='excess'){ el('line',{x1:m.l,x2:w-m.r,y1:ys(25),y2:ys(25),stroke:css('--ink'),'stroke-dasharray':'5 4',opacity:.6},svg); el('line',{x1:m.l,x2:w-m.r,y1:ys(0),y2:ys(0),stroke:css('--ink3')},svg); }
    else el('rect',{x:m.l,y:ys(120),width:w-m.l-m.r,height:ys(112)-ys(120),fill:css('--ok'),opacity:.18},svg);
    const bw=Math.max(2,(w-m.l-m.r)/n.length-1.5);
    n.forEach(r=>{ const v=kind==='excess'?r.excess*100:r.glucose; if(v==null)return; const x=xs(new Date(r.date).getTime());
      const rect = kind==='excess' ? el('rect',{x:x-bw/2,y:Math.min(ys(v),ys(0)),width:bw,height:Math.abs(ys(v)-ys(0)),fill:v>=25?css('--orange'):css('--blue'),rx:1.5},svg) : el('circle',{cx:x,cy:ys(v),r:3,fill:css('--blue'),opacity:.8},svg);
      rect.onmousemove=ev=>showTip(ev,`<b>${r.date}</b><br>extra insulin ${(r.excess*100).toFixed(0)}% of profile<br>mean glucose ${f0(r.glucose)} mg/dL`); rect.onmouseleave=hideTip; });
  });
}

/* ---- Changes ---- */
function renderSettings(){
  const t=$('#t-settings'), pr=D.proposal||{};
  t.innerHTML = `<h2>The proposal</h2><div class="card"><span class="badge ${pr.status==='proposed'?'b-act':'b-watch'}">${pr.status==='proposed'?'▲ Proposed':'● No change now'}</span><p>${esc(cap(pr.reason||''))}</p>${pr.checks?`<details><summary>Checks behind it</summary><p class="note">${esc(pr.checks)}</p></details>`:''}${pr.safety?`<details><summary>Low-glucose safety check</summary><p class="note">${esc(pr.safety)}</p></details>`:''}</div>
  <h2>Every setting change</h2><p class="lead">Changes found in your pump's own profile history, whether the system proposed them or you made them. Each is judged 14 days after: the 14 days before against the 14 days after, over the segment's hours plus 4, once all 14 days are in. Several changes made together cannot be told apart, so the count of other changes within a day is shown.</p>
  <div class="card scroll"><table id="chg"></table></div><p class="note">${esc(pr.evaluation||'')}</p>`;
  const rows=D.changes; $('#chg').innerHTML = `<tr><th class="l">Made</th><th class="l">Setting</th><th>Segment</th><th>Change</th><th class="l">By</th><th>Others within 24 h</th><th class="l">Status</th><th class="l">Result</th></tr>` + (rows.length? rows.map(r=>`<tr><td class="l">${when(r.entered_on)}</td><td class="l">${esc(SETTING[r.setting]||r.setting)}</td><td>${r.segment}</td><td>${r.old} → ${r.new}</td><td class="l">${r.origin==='person'?'you':'the system'}</td><td>${r.concurrent??0}</td><td class="l"><span class="chip">${r.status}</span></td><td class="l">${esc(r.result||'')}</td></tr>`).join('') : '<tr><td class="l" colspan="8">No changes recorded yet.</td></tr>');
}

/* ---- About ---- */
function renderAbout(){
  $('#t-about').innerHTML = `<h2>How to read this page</h2><div class="card"><p>This page describes patterns in your own data. It is decision support, not medical advice: discuss any change with your endocrinologist.</p>
  <p><b>Glucose</b> uses the international consensus metrics for continuous glucose monitoring. A day counts when at least 70% of its readings are present.</p>
  <p><b>Meals (I:C)</b> compares each meal's glucose trace with what the pump's carb ratio predicts. It assumes the correction factor is right, and the ratio it finds includes any consistent miscounting of carbohydrate.</p>
  <p><b>Basal</b> reads the controller's own deliveries. Sleep mode aims for a higher glucose (112.5 to 120 mg/dL) than daytime, so some extra insulin at night is expected.</p>
  <p><b>Changes</b> are judged before versus after in one person: other things change in the same weeks, so a result is a pattern, not proof.</p></div>
  <h2>Where the detail lives</h2><div class="card"><p>Tables behind each chart are in the folders next to this page: <code>A_outcomes/</code>, <code>B_icr/</code>, <code>C_changes/</code>, <code>F_basal/</code> and <code>00_General/</code>. Each has a <code>supporting/</code> folder for diagnostics.</p>
  <p class="note">Run ${esc(D.run.run_id||'')} · commit ${esc((D.run.commit||'').slice(0,8))} · built ${esc(D.run.built||'')}</p></div>`;
}
const RENDER={today:renderToday,glucose:renderGlucose,meals:renderMeals,basal:renderBasal,settings:renderSettings,about:renderAbout};
const startTab = (location.hash || '').slice(1); go(startTab in RENDER ? startTab : 'today');
</script>
</body>
</html>
"""


def render_dashboard(data: dict) -> str:
    return PAGE.replace("__DATA__", to_script_json(data))
