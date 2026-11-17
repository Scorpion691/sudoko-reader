/* ---- shared static drawing code for the report and the slides (no animation) ---- */
const PDF=location.search.includes('pdf');if(PDF)document.documentElement.classList.add('pdf');
const SVGNS='http://www.w3.org/2000/svg';
const COL=['#3d8bff','#21d4fd','#a78bfa'];
const NM={UCS:'UCS',GREEDY:'Greedy',ASTAR:'A*'};
const REC='exact recovery rate';
const $=id=>document.getElementById(id);
const M=x=>Array.isArray(x)?x[0]:x,S=x=>Array.isArray(x)?x[1]:0;
const f1=x=>x.toFixed(1),f2=x=>x.toFixed(2),f3=x=>x.toFixed(3),f4=x=>x.toFixed(4),fi=x=>Math.round(x).toLocaleString('en-US');
const find=(t,o)=>t&&t.find(r=>Object.keys(o).every(k=>String(r[k])===String(o[k])));
function el(t,a,p){const e=document.createElementNS(SVGNS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function div(c,p){const d=document.createElement('div');d.className=c;if(p)p.appendChild(d);return d}

/* digits: one real MNIST image per digit, used as a CSS mask */
if(D.digits){let css='';for(const n in D.digits)css+=`.dg${n}{background-image:url(${D.digits[n]})}.dg${n}.right{background-image:url(${D.digitsG[n]})}.dg${n}.mis{background-image:url(${D.digitsR[n]})}`;
 const st=document.createElement('style');st.textContent=css;document.head.appendChild(st)}

/* a board in its final (corrected) state; mode 'read' shows the wrong digits in red instead */
function board(host,read,final,mode){
 const bad=new Set();for(let k=0;k<81;k++)if(read[k]&&final[k]!==read[k])bad.add(k);
 const text=!D.digits,dg=(n,c)=>text?`<div class="dg t ${c}">${n}</div>`:`<div class="dg dg${n} ${c}"></div>`;
 let lines='';for(let i=0;i<=9;i++){const w=i%3?.22:.7;lines+=`<line x1="${i*10}" y1="0" x2="${i*10}" y2="90" stroke-width="${w}"/><line x1="0" y1="${i*10}" x2="90" y2="${i*10}" stroke-width="${w}"/>`}
 let cells='';for(let k=0;k<81;k++){
  if(!read[k]){cells+='<div class="ic"></div>';continue}
  const b=bad.has(k);
  if(!b)cells+=`<div class="ic">${dg(read[k],'')}</div>`;
  else if(mode==='read')cells+=`<div class="ic">${dg(read[k],'mis')}</div>`;
  else cells+=`<div class="ic">${dg(final[k],'right')}<i class="was">${read[k]}</i></div>`}
 host.classList.add('i-board');host.innerHTML=`<svg viewBox="0 0 90 90">${lines}</svg><div class="i-cells">${cells}</div>`}

/* ---- charts ---- */
function frame(host,h){host.innerHTML='';const W=Math.max(280,host.clientWidth||520),m={t:14,r:14,b:34,l:46};
 const s=el('svg',{viewBox:`0 0 ${W} ${h}`});host.appendChild(s);const defs=el('defs',{},s);
 return{W,H:h,m,iw:W-m.l-m.r,ih:h-m.t-m.b,s,defs,g:el('g',{transform:`translate(${m.l},${m.t})`},s),id:host.id||('c'+Math.random().toString(36).slice(2,7))}}
function yscale(f,log,lo,hi){
 if(log){const A=Math.floor(Math.log10(lo*.7)),B=Math.ceil(Math.log10(hi*1.2));return{log:1,lo:10**A,hi:10**B,fn:v=>f.ih-(Math.log10(Math.max(v,10**A))-A)/(B-A)*f.ih}}
 return{log:0,lo:0,hi,fn:v=>f.ih-v/hi*f.ih}}
function ticks(y){if(y.log){const r=[];for(let p=Math.round(Math.log10(y.lo));p<=Math.round(Math.log10(y.hi));p++)r.push(10**p);return r}
 const st=y.hi<=1.5?.25:Math.ceil(y.hi/5),r=[];for(let v=0;v<=y.hi+1e-9;v+=st)r.push(v);return r}
const tlab=v=>v>=1e6?v/1e6+'M':v>=1000?v/1000+'k':String(+v.toFixed(2));
function axes(f,y,xt,xl){
 ticks(y).forEach(v=>{const yy=y.fn(v);el('line',{x1:0,x2:f.iw,y1:yy,y2:yy,class:'grd'},f.g);
  const t=el('text',{x:-8,y:yy+4,'text-anchor':'end',class:'axt'},f.g);t.textContent=tlab(v)});
 xt.forEach(([x,l])=>{const t=el('text',{x,y:f.ih+18,'text-anchor':'middle',class:'axt'},f.g);t.textContent=l});
 if(xl){const t=el('text',{x:f.iw/2,y:f.ih+31,'text-anchor':'middle',class:'axt'},f.g);t.textContent=xl}}
function mono(P){const n=P.length;if(n<3)return P.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join('');
 const dx=[],dy=[],m=[],t=[];for(let i=0;i<n-1;i++){dx[i]=P[i+1][0]-P[i][0];dy[i]=P[i+1][1]-P[i][1];m[i]=dy[i]/dx[i]}
 t[0]=m[0];t[n-1]=m[n-2];for(let i=1;i<n-1;i++)t[i]=m[i-1]*m[i]<=0?0:(m[i-1]+m[i])/2;
 for(let i=0;i<n-1;i++){if(m[i]===0){t[i]=t[i+1]=0}else{const a=t[i]/m[i],b=t[i+1]/m[i],s=a*a+b*b;if(s>9){const k=3/Math.sqrt(s);t[i]=k*a*m[i];t[i+1]=k*b*m[i]}}}
 let d='M'+P[0][0]+' '+P[0][1];for(let i=0;i<n-1;i++){const h=dx[i]/3;d+=`C${P[i][0]+h} ${P[i][1]+t[i]*h} ${P[i+1][0]-h} ${P[i+1][1]-t[i+1]*h} ${P[i+1][0]} ${P[i+1][1]}`}return d}
function legendTo(id,items){const h=$(id);if(!h)return;h.innerHTML='';items.forEach(([n,c])=>{const s=document.createElement('span');s.innerHTML=`<i style="background:${c}"></i>${n}`;h.appendChild(s)})}
function lineChart(host,series,o){
 const all=series.flatMap(s=>s.p),f=frame(host,o.h||250),xs=[...new Set(all.map(p=>p.x))].sort((a,b)=>a-b);
 const x=v=>(v-xs[0])/(xs[xs.length-1]-xs[0])*f.iw,y=yscale(f,o.log,Math.min(...all.map(p=>p.y)),o.log?Math.max(...all.map(p=>p.y+p.sd)):1.05);
 axes(f,y,xs.map(v=>[x(v),String(v)]),o.xl);
 series.forEach((s,si)=>{const gid=f.id+'g'+si,g=el('linearGradient',{id:gid,x1:0,y1:0,x2:0,y2:1},f.defs);
  el('stop',{offset:0,'stop-color':s.c,'stop-opacity':.34},g);el('stop',{offset:1,'stop-color':s.c,'stop-opacity':0},g);
  const P=s.p.map(p=>[x(p.x),y.fn(p.y)]),d=mono(P);
  if(!s.dash)el('path',{d:d+`L${P[P.length-1][0]} ${f.ih}L${P[0][0]} ${f.ih}Z`,fill:PDF?s.c:`url(#${gid})`,'fill-opacity':PDF?.14:1},f.g);
  el('path',{d,fill:'none',stroke:s.c,'stroke-width':s.dash?1.8:2.6,'stroke-linecap':'round','stroke-dasharray':s.dash?'6 5':''},f.g);
  s.p.forEach(p=>{el('line',{x1:x(p.x),x2:x(p.x),y1:y.fn(Math.max(y.lo,p.y-p.sd)),y2:y.fn(Math.min(y.hi,p.y+p.sd)),stroke:s.c,'stroke-opacity':.45,'stroke-width':1.4},f.g);
   el('circle',{cx:x(p.x),cy:y.fn(p.y),r:4,fill:'#0a1240',stroke:s.c,'stroke-width':2.2},f.g)})})}
function barChart(host,groups,names,o){
 const f=frame(host,o.h||250),all=groups.flatMap(g=>g.b);
 const y=yscale(f,o.log,Math.min(...all.map(b=>b.y)),o.log?Math.max(...all.map(b=>b.y+b.sd)):1.1),gw=f.iw/groups.length,bw=gw*.78/names.length;
 axes(f,y,groups.map((g,i)=>[gw*(i+.5),g.l]),o.xl);
 names.forEach((n,i)=>{const g=el('linearGradient',{id:f.id+'b'+i,x1:0,y1:0,x2:0,y2:1},f.defs);
  el('stop',{offset:0,'stop-color':COL[i]},g);el('stop',{offset:1,'stop-color':COL[i],'stop-opacity':.2},g)});
 groups.forEach((g,gi)=>g.b.forEach((b,i)=>{const bx=gw*gi+gw*.11+i*bw,top=y.fn(Math.min(b.y,y.hi));
  el('rect',{x:bx+1.5,y:top,width:Math.max(4,bw-3),height:Math.max(2,f.ih-top),rx:Math.min(6,(bw-3)/2),fill:PDF?COL[i]:`url(#${f.id}b${i})`,'fill-opacity':PDF?.85:1},f.g);
  const cx=bx+bw/2;el('line',{x1:cx,x2:cx,y1:y.fn(Math.max(y.lo,b.y-b.sd)),y2:y.fn(Math.min(y.hi,b.y+b.sd)),stroke:'#fff','stroke-opacity':.55,'stroke-width':1.1},f.g);
  if(o.lab&&b.y>0&&(i===0||b.y<.99)){const t=el('text',{x:cx,y:y.fn(Math.min(y.hi,b.y+b.sd))-4,class:'vl'},f.g);t.textContent=o.f(b.y)}}))}

/* ready-made charts used by both documents */
const SER=(key)=>['UCS','GREEDY','ASTAR'].map((a,i)=>({n:NM[a],c:COL[i],dash:a=='ASTAR',p:D.exp1.filter(r=>r.algorithm==a&&(key!='nodes expanded'||+r.misreads>0)).map(r=>({x:+r.misreads,y:M(r[key]),sd:S(r[key])}))}));
function chartNodes(id,h){lineChart($(id),SER('nodes expanded'),{log:1,xl:'injected misreads',h})}
function chartRecovery(id,h){lineChart($(id),SER(REC),{log:0,xl:'injected misreads',h})}
function chartCsp(id,h){const nm=['BT','BT+FC','BT+FC+AC-3'];
 barChart($(id),[25,30,35].map(v=>({l:v+' givens',b:nm.map(n=>{const r=find(D.exp2,{givens:v,method:n}).assignments;return{y:M(r),sd:S(r)}})})),nm,{log:1,f:fi,xl:'puzzle difficulty',h})}
function chartNN(id,h){const nm=['NN-guided (UCS)','no correction','random correction'];
 barChart($(id),[0,1,2,3].map(v=>({l:v+(v==1?' misread':' misreads'),b:nm.map(n=>{const r=find(D.exp3,{misreads:v,method:n})[REC];return{y:M(r),sd:S(r)}})})),nm,{log:0,lab:1,f:f2,xl:'injected misreads',h})}
function confusion(host,Mx){host.innerHTML='';const n=9,c=26,W=n*c+30,s=el('svg',{viewBox:`0 0 ${W} ${n*c+30}`},host),mx=Math.max(...Mx.flat().map((v,i)=>(Math.floor(i/n)==i%n?0:v)),1);
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const v=Mx[i][j],on=i===j,a=on?.28:Math.min(1,.18+v/mx*.82);
  el('rect',{x:24+j*c,y:i*c,width:c-2,height:c-2,rx:4,fill:on?'#01b574':'#f0605d','fill-opacity':v?a:.05},s);
  const t=el('text',{x:24+j*c+c/2-1,y:i*c+c/2+3,'text-anchor':'middle',class:'axt'},s);t.style.fill='#fff';t.style.fontSize='9px';t.textContent=v}
 for(let i=0;i<n;i++){let t=el('text',{x:24+i*c+c/2-1,y:n*c+12,'text-anchor':'middle',class:'axt'},s);t.textContent=i+1;
  t=el('text',{x:16,y:i*c+c/2+3,'text-anchor':'end',class:'axt'},s);t.textContent=i+1}
 let t=el('text',{x:24+n*c/2,y:n*c+26,'text-anchor':'middle',class:'axt'},s);t.textContent='predicted';
 t=el('text',{x:6,y:n*c/2,'text-anchor':'middle',class:'axt',transform:`rotate(-90 6 ${n*c/2})`},s);t.textContent='true'}
