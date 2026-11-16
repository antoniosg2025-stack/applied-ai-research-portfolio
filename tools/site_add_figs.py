"""Adds case 02-07 charts and the 'What if the agents never stopped?' section to the site template."""
import json, re
t = open("site/template.html").read()
css = """
.figs { display: grid; gap: 16px; margin-top: 26px; }
.figs .panel h3 { font-size: 18px; margin: 4px 6px 4px; line-height: 1.3; }
.whatif { background: linear-gradient(180deg, #0F1E33 0%, #13294A 100%); color: #E3EAF3; }
.whatif h2 { color: #fff; font-size: clamp(30px, 4.4vw, 50px); letter-spacing: -.02em; }
.whatif .lede { color: #C9D5E4; }
.whatif .eyebrow { margin-bottom: 12px; }
.wpanel { margin-top: 26px; background: #0B1729; border: 1px solid #27426A; border-radius: 18px; padding: 18px; }
.wpanel .control label { color: #fff; }
.wpanel .control output { color: var(--gold); }
.wpanel input[type=range] { accent-color: var(--gold); }
.wsub { font-size: 14px; color: #9DB0C7; margin: 0 6px 6px; }
.biz { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; margin-top: 16px; }
.b { background: #13243D; border: 1px solid #27426A; border-radius: 14px; padding: 14px 16px; display: flex; flex-direction: column; gap: 6px; cursor: pointer; text-align: left; color: inherit; font: inherit; }
.b.sel { border-color: var(--gold); box-shadow: 0 0 0 1px var(--gold) inset; }
.b .nm { font-family: var(--mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--gold); }
.b .u { font-size: 14px; color: #C9D5E4; }
.b .row { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.b .now { font-size: 14px; color: #9DB0C7; }
.b .now input { width: 64px; font: 600 15px var(--mono); background: #0B1729; color: #fff; border: 1px solid #3A5A86; border-radius: 8px; padding: 4px 6px; }
.b .agent { font-family: var(--display); font-weight: 800; font-size: 40px; line-height: 1; color: #fff; letter-spacing: -.02em; }
.b .x { font-family: var(--mono); font-weight: 600; color: var(--gold); font-size: 16px; }
.b .err { font-size: 13px; color: #F1A9A7; }
.b .src { font-size: 12px; color: #7F93AD; }
.b .src.as { color: #E0A526; }
.evid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin-top: 18px; }
.ev { border-left: 4px solid var(--gold); background: rgba(255,255,255,.04); border-radius: 0 12px 12px 0; padding: 12px 16px; font-size: 15px; color: #DCE5F0; }
.ev b { color: #fff; font-family: var(--mono); }
.wchart { width: 100%; height: 340px; }
.caveat { font-size: 13px; color: #9DB0C7; margin: 14px 4px 0; }
"""
t = t.replace("@media (max-width: 860px) {", css + "\n@media (max-width: 860px) {", 1)
t = t.replace('<a class="chip" href="#cases">All cases</a>', '<a class="chip" href="#cases">All cases</a>\n      <a class="chip" href="#what-if">What if</a>', 1)

whatif = '''
<section class="whatif" id="what-if">
  <div class="wrap">
    <p class="eyebrow">Why it matters · productivity</p>
    <h2>What if the agents never stopped to wait for me?</h2>
    <p class="lede read">Today every agent stops at my approval, and I can only look at one business at a time. When I spent a week on one project, the others stood still. An agent that decides like me only asks when it is unsure, so the same few minutes of my attention clear many more decisions.</p>
    <div class="wpanel">
      <div class="control">
        <label for="askq">How often the agent still asks me</label>
        <output id="askOut" for="askq">27%</output>
        <input id="askq" type="range" min="0" max="9" step="1" value="3">
      </div>
      <p class="wsub" id="askNote">27% is what the per-type profile measured in case 01, at 85% confidence. Each output below is multiplied by <b id="mult">3.7×</b>.</p>
      <div class="biz" id="biz"></div>
      <div id="wchart" class="wchart"></div>
      <p class="wsub" id="wchartNote"></p>
      <div class="evid">
        <div class="ev">SEO, real decision log: <b>9 of 24</b> changes ended the first week waiting for me. Over 12 weeks, days lost waiting fall from <b>1,260</b> to <b>356</b> with reversibility rules that travel across sites.</div>
        <div class="ev">Press: a weekly approval batch loses <b>35%</b> of the reach of pieces I would approve. When the agent publishes the <b>20%</b> it is sure about, the loss drops to <b>12%</b>.</div>
        <div class="ev">Agents: at 27% questions, the agent made <b>0.24</b> decisions per 100 that I would have made differently. Mistakes grow when it asks less: move the control to see the cost.</div>
      </div>
      <p class="caveat">How this is computed: output with the agent = output today × (100 ÷ % of decisions it still asks me), assuming my review time stays the same and the agent's own capacity is not the limit. The question and error rates come from the case 01 simulation. Numbers marked <span style="color: var(--gold)">assumption</span> are placeholders until I enter my real weekly figures.</p>
    </div>
  </div>
</section>
'''
t = t.replace('<section class="case-hero" id="case-01">', whatif + '\n<section class="case-hero" id="case-01">', 1)
t = t.replace('<span class="pill">Interactive charts for this case are being built</span>', '<div class="figs" data-case="__N__"></div>')
# give each figs container its case number
for n in ["02", "03", "04", "05", "06", "07"]:
    t = t.replace('<section class="stub" id="case-%s">' % n, '<section class="stub" id="case-%s">' % n, 1)
    i = t.index('<section class="stub" id="case-%s">' % n)
    j = t.index('data-case="__N__"', i)
    t = t[:j] + 'data-case="%s"' % n + t[j + len('data-case="__N__"'):]
# watched sections for the nav
t = t.replace('const watched=["cases","case-01"', 'const watched=["cases","what-if","case-01"')
t = t.replace("const D = __DATA__;", "const D = __DATA__;\nconst F = __FIGS__;")

js = r'''
// ---------- Case 02-07 charts, drawn from the notebooks' own figures ----------
function catLabel(ticks, x){ let best=null, bd=1e9; for(const [v,l] of ticks){ const d=Math.abs(v-x); if(d<bd){bd=d;best=l;} } return bd<0.51 ? best.replace(/\n/g," ") : String(x); }
function renderFig(el, f){
  const traces=[]; const ticks=f.xticks||[]; const numericX = ticks.every(([v,l])=>!isNaN(parseFloat(l)));
  const anyStack = f.bars.some(b=>b.b.some(v=>Math.abs(v)>1e-9));
  f.bars.forEach((b,i)=>{
    const x = b.x.map(v=> numericX ? v : catLabel(ticks, v));
    traces.push({ type:"bar", name:b.label||"", showlegend:!!b.label, x, y:b.h, marker:{ color:b.color.length>1 && !b.label ? b.color : b.color[0] },
      text: b.h.map(v=> Math.abs(v)>=1000 ? Math.round(v).toLocaleString() : (Math.abs(v)<10 ? (+v.toFixed(1)).toString() : Math.round(v).toString())),
      textposition: anyStack ? "inside" : "outside", cliponaxis:false, textfont:{ family:"IBM Plex Mono, monospace", size: small()?11:12 },
      hovertemplate:(b.label?b.label+": ":"")+"%{y:,.1f}<extra>%{x}</extra>" });
  });
  const ytick=f.yticks||[];
  f.hbars.forEach(b=>{
    const y=b.y.map(v=>catLabel(ytick, v));
    traces.push({ type:"bar", orientation:"h", name:b.label||"", showlegend:!!b.label, y, x:b.w, base:b.l, marker:{ color:b.color[0] },
      hovertemplate:(b.label?b.label+": ":"")+"%{x}<extra>%{y}</extra>" });
  });
  f.lines.forEach(l=>{
    traces.push({ type:"scatter", mode:"lines+markers", name:l.label, showlegend:!!l.label, x:l.x, y:l.y, line:{ color:l.color, width:3, dash:l.dash?"dash":"solid" }, marker:{ size:7, color:l.color },
      hovertemplate:(l.label||"")+": %{y:,.1f}<extra></extra>" });
  });
  if(f.y2){ f.y2.lines.forEach(l=> traces.push({ type:"scatter", mode:"lines+markers", name:l.label, x:l.x, y:l.y, yaxis:"y2", line:{ color:l.color, width:3 }, marker:{ size:6, color:l.color }, hovertemplate:(l.label||"")+": %{y:,.0f}<extra></extra>" })); }
  const lay = base({ hovermode:"closest", barmode: anyStack || f.hbars.length>1 ? "stack" : "group",
    xaxis:{ title:{ text:f.xlabel||"" }, type: f.xlog ? "log" : (numericX ? "linear" : "category"), gridcolor:C.line, fixedrange:true, automargin:true, tickangle: small() && !numericX ? -30 : 0 },
    yaxis:{ title:{ text: (f.ylabel||"").replace(/\n/g," ") }, type: f.ylog ? "log":"linear", gridcolor:C.line, fixedrange:true, automargin:true, rangemode:"tozero" },
    margin:{ l:12, r: f.y2?12:14, t:16, b:12 }, legend:{ orientation:"h", y:-0.3, x:0 } });
  if(f.xlog && ticks.length){ lay.xaxis.tickvals=ticks.map(t=>t[0]); lay.xaxis.ticktext=ticks.map(t=>t[1]); }
  if(f.hbars.length){ lay.yaxis = { type:"category", autorange:"reversed", fixedrange:true, automargin:true }; lay.xaxis.type="linear"; lay.xaxis.title={text:f.xlabel||""}; }
  if(f.y2){ lay.yaxis2 = { title:{ text:f.y2.ylabel||"" }, overlaying:"y", side:"right", showgrid:false, fixedrange:true, automargin:true, rangemode:"tozero" }; }
  Plotly.newPlot(el, traces, lay, cfg);
}
document.querySelectorAll(".figs").forEach(box=>{
  (F[box.dataset.case]||[]).forEach((f,i)=>{
    const p=document.createElement("div"); p.className="panel";
    const h=document.createElement("h3"); h.textContent=f.title; p.appendChild(h);
    const c=document.createElement("div"); c.className="chart"; c.id="fig-"+box.dataset.case+"-"+i; p.appendChild(c);
    box.appendChild(p); renderFig(c, f);
  });
});

// ---------- What if the agents never stopped ----------
const LEVELS=[{asks:100, errors:0, label:"today: every decision waits"}].concat(
  per.slice().reverse().map(s=>({asks:s.asks, errors:s.errors, tau:s.tau})));
const BIZ=[
  { key:"press", name:"Press", unit:"videos and posts a week", now:19, src:"My content plan: 18 GrabGo + 1 Equilibra a week", real:true },
  { key:"seo", name:"SEO", unit:"site changes a week", now:24, src:"Real log: 24 changes in the first week", real:true },
  { key:"equilibra", name:"Equilibra", unit:"market theses tested a week", now:3, src:"assumption", real:false },
  { key:"grabgo", name:"GrabGo", unit:"new locations contacted a week", now:10, src:"assumption", real:false },
  { key:"vending", name:"Vending", unit:"product-mix tests a week", now:1, src:"assumption", real:false },
  { key:"suppliers", name:"Suppliers", unit:"products evaluated a week", now:5, src:"assumption", real:false }
];
let selBiz="seo";
const askq=document.getElementById("askq"), bizBox=document.getElementById("biz");
BIZ.forEach(b=>{
  const el=document.createElement("div"); el.className="b"; el.dataset.key=b.key; el.tabIndex=0; el.setAttribute("role","button");
  el.innerHTML=`<span class="nm">${b.name}</span><span class="u">${b.unit}</span>
    <div class="row"><span class="agent num" data-v>0</span><span class="x" data-x></span></div>
    <span class="now"><label for="now-${b.key}">Today:</label> <input id="now-${b.key}" type="number" min="0" step="1" value="${b.now}"></span>
    <span class="err" data-e></span><span class="src ${b.real?"":"as"}">${b.src}</span>`;
  el.addEventListener("click",e=>{ if(e.target.tagName==="INPUT") return; selBiz=b.key; refresh(); });
  el.addEventListener("keydown",e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); selBiz=b.key; refresh(); } });
  el.querySelector("input").addEventListener("input",e=>{ b.now=Math.max(0, +e.target.value||0); refresh(); });
  bizBox.appendChild(el);
});
Plotly.newPlot("wchart", [], base({ font:{ family:"IBM Plex Sans, sans-serif", size: small()?12:13, color:"#C9D5E4" } }), cfg);
function fmt(v){ return v>=100 ? Math.round(v).toLocaleString() : (v>=10 ? Math.round(v).toString() : (+v.toFixed(1)).toString()); }
function refresh(){
  const L=LEVELS[+askq.value]; const m=100/L.asks;
  document.getElementById("askOut").textContent=Math.round(L.asks)+"%";
  document.getElementById("mult").textContent=(+m.toFixed(1))+"×";
  document.getElementById("askNote").innerHTML = L.asks===100 ? "Today: every decision waits for me. Output is what I can approve myself." :
    `At this setting the agent acts alone when it is at least ${Math.round(L.tau*100)}% sure. Measured in case 01: it asks ${Math.round(L.asks)} times and makes ${L.errors.toFixed(2)} decisions per 100 that I would have made differently. Each output is multiplied by <b>${(+m.toFixed(1))}×</b>.`;
  bizBox.querySelectorAll(".b").forEach(el=>{
    const b=BIZ.find(x=>x.key===el.dataset.key); const out=b.now*m;
    el.classList.toggle("sel", b.key===selBiz);
    el.querySelector("[data-v]").textContent=fmt(out);
    el.querySelector("[data-x]").textContent = L.asks===100 ? "" : "×"+(+m.toFixed(1));
    el.querySelector("[data-e]").textContent = L.asks===100 ? "No agent mistakes: nothing moves without me." : `≈ ${fmt(out*L.errors/100*12)} decisions in 12 weeks I would have made differently`;
  });
  const b=BIZ.find(x=>x.key===selBiz); const wk=[...Array(13).keys()];
  Plotly.react("wchart", [
    { type:"scatter", mode:"lines", name:"Every decision waits for me", x:wk, y:wk.map(w=>w*b.now), line:{ color:"#7F93AD", width:3, dash:"dot" }, hovertemplate:"%{y:,.0f}<extra>week %{x}</extra>" },
    { type:"scatter", mode:"lines", name:"Agent decides like me", x:wk, y:wk.map(w=>w*b.now*m), line:{ color:C.gold, width:4 }, fill:"tonexty", fillcolor:"rgba(224,165,38,.14)", hovertemplate:"%{y:,.0f}<extra>week %{x}</extra>" }
  ], base({ font:{ family:"IBM Plex Sans, sans-serif", size: small()?12:13, color:"#C9D5E4" }, hovermode:"x unified",
    margin:{ l:12, r:14, t:30, b:12 },
    title:{ text: b.name+": "+b.unit.replace(" a week","")+" over 12 weeks", font:{ size:15, color:"#fff", family:"Bricolage Grotesque, sans-serif" }, x:0.01, y:0.98 },
    xaxis:{ title:{ text:"Week" }, gridcolor:"#22395A", zeroline:false, fixedrange:true, automargin:true, dtick:2 },
    yaxis:{ title:{ text:"Cumulative" }, gridcolor:"#22395A", zeroline:false, fixedrange:true, automargin:true, rangemode:"tozero" },
    legend:{ orientation:"h", y:-0.28, x:0, font:{ color:"#C9D5E4" } } }), cfg);
  document.getElementById("wchartNote").textContent = "Tap any business above to see its 12 weeks. "+ (b.real ? "Today's figure is real." : "Today's figure is an assumption: type your real number in its box.");
}
askq.addEventListener("input", refresh); refresh();
'''
t = t.replace("// Navigation:", js + "\n// Navigation:", 1)
open("site/template.html", "w").write(t)
d = json.load(open("site/data/case01.json")); f = json.load(open("site/data/figs.json"))
open("site/index.html", "w").write(t.replace("__DATA__", json.dumps(d, separators=(",", ":"))).replace("__FIGS__", json.dumps(f, separators=(",", ":"))))
print("ok", t.count('data-case="0'))
