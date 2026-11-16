import re, json
t=open("site/template.html").read()
css = """
.topnav { position: sticky; top: 0; z-index: 20; background: rgba(15,30,51,.96); backdrop-filter: blur(8px); border-bottom: 1px solid #22395A; }
.topnav .wrap { display: flex; align-items: center; gap: 10px; padding-top: 10px; padding-bottom: 10px; }
.topnav .home { color: #fff; font-family: var(--display); font-weight: 700; font-size: 15px; text-decoration: none; white-space: nowrap; padding: 6px 12px 6px 0; border-right: 1px solid #2F4A6E; }
.chips { display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; }
.chips::-webkit-scrollbar { display: none; }
.chip { flex: none; font-family: var(--mono); font-size: 13px; color: #C9D5E4; text-decoration: none; border: 1px solid #2F4A6E; border-radius: 999px; padding: 6px 12px; white-space: nowrap; }
.chip.on { background: var(--gold); border-color: var(--gold); color: var(--navy); font-weight: 600; }
.stub { border-top: 1px solid var(--line); }
.stub .tag { font-family: var(--mono); font-size: 13px; letter-spacing: .08em; text-transform: uppercase; margin: 0 0 10px; }
.stub h2 { font-size: clamp(28px, 3.6vw, 40px); letter-spacing: -.015em; }
.stub .big { font-family: var(--display); font-weight: 800; font-size: clamp(44px, 7vw, 72px); letter-spacing: -.02em; line-height: 1; margin: 22px 0 6px; }
.pill { display: inline-block; margin-top: 18px; font-size: 13px; font-weight: 600; color: var(--muted); background: var(--card); border: 1px dashed #B8C2D0; border-radius: 999px; padding: 6px 12px; }
.pager { display: flex; justify-content: space-between; gap: 12px; margin-top: 34px; flex-wrap: wrap; }
.pager a { flex: 1 1 160px; text-decoration: none; border: 1px solid var(--line); background: var(--card); border-radius: 12px; padding: 12px 16px; color: var(--ink); font-weight: 600; font-size: 15px; }
.pager a span { display: block; font-family: var(--mono); font-size: 12px; color: var(--muted); font-weight: 500; letter-spacing: .06em; text-transform: uppercase; }
.pager a.next { text-align: right; }
.totop { position: fixed; right: 16px; bottom: 18px; z-index: 30; background: var(--teal); color: #fff; border: 0; border-radius: 999px; padding: 12px 16px; font: 600 14px var(--body); box-shadow: 0 8px 20px rgba(0,0,0,.2); cursor: pointer; }
a.case:hover { transform: translateY(-2px); }
[id] { scroll-margin-top: 64px; }
"""
t=t.replace("@media (max-width: 860px) {", css+"\n@media (max-width: 860px) {",1)
t=t.replace('<header class="cover">','<header class="cover" id="top">')
nav='''
<nav class="topnav" aria-label="Cases">
  <div class="wrap">
    <a class="home" href="#top">Portfolio</a>
    <div class="chips" id="chips">
      <a class="chip" href="#cases">All cases</a>
      <a class="chip" href="#case-01">01 Agents</a>
      <a class="chip" href="#case-02">02 Equilibra</a>
      <a class="chip" href="#case-03">03 Press</a>
      <a class="chip" href="#case-04">04 SEO</a>
      <a class="chip" href="#case-05">05 Vending</a>
      <a class="chip" href="#case-06">06 GrabGo</a>
      <a class="chip" href="#case-07">07 Suppliers</a>
    </div>
  </div>
</nav>
'''
t=t.replace("</header>\n","</header>\n"+nav,1)
t=t.replace('<section>\n  <div class="wrap">\n    <div class="sec-head">','<section id="cases">\n  <div class="wrap">\n    <div class="sec-head">',1)
for n in range(2,8):
    t,c=re.subn(r'<div class="case" style="(--accent: var\(--\w+\))">(\s*)<span class="tag">0%d'%n,
             lambda m: '<a class="case" href="#case-0%d" style="%s">%s<span class="tag">0%d'%(n,m.group(1),m.group(2),n), t, count=1)
    assert c==1, n
t=t.replace('<span class="state">Interactive version next</span>\n      </div>','<span class="state">Read the case →</span>\n      </a>')
t=t.replace('<span class="state">Open the interactive case ↓</span>','<span class="state">Open the interactive case →</span>')
t=t.replace('<p class="note" style="margin-top:18px">Code and full notebook:','<div class="pager"><a href="#cases"><span>Back</span>All cases</a><a class="next" href="#case-02"><span>Next</span>02 · Equilibra</a></div>\n    <p class="note" style="margin-top:18px">Code and full notebook:',1)
stubs=[
 ("02","Equilibra","gold","Copying the owner's money moves fails for people in other situations","$8,800","misplaced per user when the agent copies the owner's moves",
  "Equilibra helps users study their portfolio with their own AI. The owner's rule is simple: pay off every card first, keep a stable cushion, and put at risk only what a fixed deposit would pay in a year. Copying his actions breaks for users with different debts and income. Learning the rule over the situation, with 13 well-chosen questions to the owner, decides for 1,000 different users as he would.","02_equilibra_situation_rules.ipynb"),
 ("03","Press","coral","A correction with a reason teaches more than many approvals","4× faster","learning when the owner explains why, instead of only yes or no",
  "Press is the content agent for social networks. Waiting for weekly approval batches loses about 35% of the reach of pieces the owner would have approved. Letting the agent publish alone when it is confident cuts the loss to 12%, with 13 wrong posts out of 3,000. In our real posts, the Spanish version holds viewers about twice as long at 2 seconds in 8 of 8 pairs.","03_press_corrections_vs_approvals.ipynb"),
 ("04","SEO","sky","Autonomy for reversible changes, and rules that travel across sites","360 → 126","questions to the owner in 12 weeks",
  "Built on the SEO agent's real decision log: 7 of 10 documents were done alone, while 3 of 5 staging changes and 2 of 3 production changes waited for the owner. Letting only reversible changes run alone cuts the questions. A correction learned on one site travels to all of them, with one question about its scope, so repeated rules fall from 70 to 6 to 0. Treating every correction as global would wrongly block about 21 changes.","04_seo_reversibility_rule_transfer.ipynb"),
 ("05","Vending","teal","The owner's limits beat the textbook optimum","$83 → $155","weekly margin after travel",
  "Restocking vending machines looks like a forecasting problem. The textbook plan ignored the owner's time and driving distance. His own rules include them and nearly double the weekly margin after travel. Learning each correction once takes 12 corrections instead of 150. One rule alone, skipping the clinic on low-sales weeks, adds $47 a week and saves about $77 in driving.","05_vending_owner_rules_restock.ipynb"),
 ("06","GrabGo","gold","Cheaper checkout by changing the shelf, not the camera","−90%","money lost to recognition errors",
  "GrabGo turns any space into a store that charges from images: the shelf camera, or a photo the customer sends by WhatsApp. Giving look-alike products the same price cuts the money lost to recognition errors by about 90% with no new hardware. Photo-only checkout misses about $160 per 1,000 carts. The owner's privacy and cost weights choose a different setup per location: photo-only at the dealership and the hotel, camera plus photo at a 200-person factory.","06_grabgo_images_owner_tradeoffs.ipynb"),
 ("07","Suppliers","coral","Phone photos plus a scale, judged with the owner's tolerance","15% → 2.5%","of lots where the agent and the owner disagree",
  "Samples and orders from China are inspected by a person with a phone: photos, videos and the weight of each box. Photos alone miss swapped components, and about 13 lots a year would ship. A scale catches them on the first lot. One fixed tolerance rejects about 7 good lots a year; the owner's tolerance learned per product cuts disagreement to 2.5%, with about 6 questions to him a year.","07_suppliers_phone_inspection_weight.ipynb"),
]
names=["01 · Agents"]+[f"{s[0]} · {s[1]}" for s in stubs]
html=""
for i,(n,name,col,title,big,unit,text,nb) in enumerate(stubs):
    nxt = f'<a class="next" href="#case-0{int(n)+1}"><span>Next</span>{names[i+2]}</a>' if i+1<len(stubs) else '<a class="next" href="#top"><span>Back</span>To the top</a>'
    html+=f'''
<section class="stub" id="case-{n}">
  <div class="wrap">
    <p class="tag" style="color: var(--{col})">Case {n} · {name}</p>
    <h2>{title}</h2>
    <div class="big num" style="color: var(--{col})">{big}</div>
    <div style="color: var(--muted)">{unit}</div>
    <p class="lede read">{text}</p>
    <span class="pill">Interactive charts for this case are being built</span>
    <div class="pager"><a href="#case-0{int(n)-1}"><span>Previous</span>{names[i]}</a><a href="#cases"><span>Back</span>All cases</a>{nxt}</div>
    <p class="note" style="margin-top:18px">Code and full notebook: <a href="https://github.com/antoniosg2025-stack/applied-ai-research-portfolio/blob/main/notebooks/{nb}">{nb}</a></p>
  </div>
</section>
'''
t=t.replace("\n<footer>", html+'\n<button class="totop" id="totop" type="button">↑ All cases</button>\n<footer>',1)
js='''
// Navigation: jumps that work inside the viewer frame, plus the active chip
function go(id){ const el=document.getElementById(id); if(!el) return;
  el.scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth", block:"start"});
  try{ history.replaceState(null,"","#"+id); }catch(e){} }
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener("click",e=>{ e.preventDefault(); go(a.getAttribute("href").slice(1)); }));
document.getElementById("totop").addEventListener("click",()=>go("cases"));
const chipBox=document.getElementById("chips"), chips=[...chipBox.querySelectorAll(".chip")];
const watched=["cases","case-01","case-02","case-03","case-04","case-05","case-06","case-07"].map(id=>document.getElementById(id));
let lastCur=null;
function mark(){ let cur=null; for(const s of watched){ if(s.getBoundingClientRect().top<120) cur=s.id; }
  if(cur===lastCur) return; lastCur=cur;
  chips.forEach(c=>{ const on=c.getAttribute("href")==="#"+cur; c.classList.toggle("on",on);
    if(on) chipBox.scrollTo({left: c.offsetLeft - chipBox.offsetLeft - 40, behavior:"smooth"}); }); }
addEventListener("scroll",mark,{passive:true}); mark();
if(location.hash.length>1) setTimeout(()=>go(location.hash.slice(1)),300);
'''
t=t.replace("</script>\n</body>", js+"</script>\n</body>")
open("site/template.html","w").write(t)
d=json.load(open("site/data/case01.json"))
open("site/index.html","w").write(t.replace("__DATA__", json.dumps(d,separators=(",",":"))))
print("links", t.count('href="#case-0'), "sections", t.count('class="stub"'))
