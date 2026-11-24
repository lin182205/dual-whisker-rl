'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value || 0).toLocaleString('zh-CN');
const steps = value => `${Number((Number(value || 0) / 10000).toFixed(2)).toLocaleString('zh-CN')} 万`;
const percent = value => `${(Number(value || 0) * 100).toFixed(1)}%`;
const bytes = value => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(1)} GB` : `${(value / 1024 ** 2).toFixed(1)} MB`;
const date = value => value ? new Date(value * 1000).toLocaleString('zh-CN', {hour12:false}) : '尚无更新';
const states = {queued:'排队中',starting:'启动中',running:'训练中',pausing:'暂停中',paused:'已暂停',stopping:'停止中',stopped:'已停止',completed:'已完成',failed:'失败',unknown:'待核对',unresponsive:'待响应'};
const actions = {imported:'接入监控',created:'创建任务',started:'开始运行',set_target:'调整目标',pause:'请求暂停',resume:'加入续训队列',stop:'请求停止',takeover:'接管控制',failed:'任务失败'};
const active = new Set(['running','starting','pausing','stopping','unresponsive']);
let token = '', tasks = [], selected = null, detail = null, filter = 'all', tab = 'curves', busy = false, lastCharts = 0, charts = {}, confirmAction = null, sourceCheckpoints = [], overviewBusy = false, goalTask = null;
Chart.register({id:'segments',afterDatasetsDraw(chart,args,options){const {ctx,chartArea,scales}=chart;ctx.save();ctx.strokeStyle='#657a995e';ctx.setLineDash([3,4]);for(const point of options.points||[]){const x=scales.x.getPixelForValue(point);if(x>=chartArea.left&&x<=chartArea.right){ctx.beginPath();ctx.moveTo(x,chartArea.top);ctx.lineTo(x,chartArea.bottom);ctx.stroke();}}ctx.restore();}});

async function api(path, body, retried=false) {
  const response = await fetch(path, {method:body ? 'POST':'GET', credentials:'same-origin', headers:body ? {'Content-Type':'application/json','X-CSRF-Token':token}:{}, body:body ? JSON.stringify(body):undefined});
  const result = await response.json();
  if (response.status===403 && body && !retried && result.error?.includes('会话已失效')) {
    token=(await api('/api/bootstrap')).csrf_token;
    return api(path,body,true);
  }
  if (!response.ok) throw new Error(result.error || (typeof result.detail === 'string' ? result.detail : '输入无效，请检查目标步数和任务状态'));
  return result;
}
function toast(text, error=false) { $('toast').textContent=text; $('toast').className=`toast${error?' error':''}`; clearTimeout(toast.timer); toast.timer=setTimeout(()=>$('toast').classList.add('hidden'),6000); }
function duration(seconds) { if (seconds == null) return '估时采集中'; const hours=seconds/3600; return hours >= 24 ? `约 ${(hours/24).toFixed(1)} 天` : hours >= 1 ? `约 ${hours.toFixed(1)} 小时` : `约 ${Math.max(1,Math.ceil(seconds/60))} 分钟`; }
function badge(task) { return `<span class="badge ${esc(task.state)}">${esc(states[task.state] || task.state)}</span>`; }
function setMeter(id, value) { $(id).style.width=`${Math.max(0,Math.min(100,Number(value)||0))}%`; }
function renderResources(r) {
  $('gpu-load').textContent=r.gpu ? `${r.gpu.percent}%`:'暂不可用'; $('gpu-name').textContent=r.gpu?.name || '未读取到 GPU';
  $('gpu-memory').textContent=r.gpu ? `${(r.gpu.used_mb/1024).toFixed(1)} / ${(r.gpu.total_mb/1024).toFixed(0)} GB`:'—';
  $('cpu-load').textContent=r.cpu_percent==null?'采集中':`${r.cpu_percent}%`; $('cpu-cores').textContent=`实例配额 ${r.cpu_cores} 核`;
  $('memory').textContent=`${bytes(r.memory_used)} / ${bytes(r.memory_total)}`; $('disk').textContent=`数据盘 ${bytes(r.disk_used)} / ${bytes(r.disk_total)}`;
  setMeter('gpu-bar',r.gpu?.percent); setMeter('vram-bar',r.gpu ? r.gpu.used_mb/r.gpu.total_mb*100:0); setMeter('cpu-bar',r.cpu_percent); setMeter('ram-bar',r.memory_used/r.memory_total*100);
}
function renderTasks() {
  $('task-count').textContent=tasks.length; $('running-count').textContent=`${tasks.filter(t=>active.has(t.state)).length} / 2 运行中`;
  const visible=tasks.filter(t=>filter==='all' || (filter==='active' && active.has(t.state)) || (filter==='queued' && ['queued','paused'].includes(t.state)) || (filter==='ended' && ['completed','stopped','failed','unknown'].includes(t.state)));
  $('tasks').innerHTML=visible.length ? visible.map(t=>`<button class="task-card ${t.id===selected?'chosen':''}" data-task="${esc(t.id)}"><div class="task-top"><strong>${esc(t.name)}</strong>${badge(t)}</div><div class="task-meta"><span>${t.variant==='original'?'15 cm 原版':'22.5 cm 改进版'}</span><span>· ${t.mode==='legacy'?'仅监控':'可控制'}</span></div><div class="meter"><i data-width="${t.session_progress*100}"></i></div><div class="task-progress"><span>本次完成</span><span>${percent(t.session_progress)}</span></div><div class="task-footer"><span>${steps(t.current_steps)} / ${steps(t.target_steps)}</span><span>${active.has(t.state)?duration(t.eta_seconds):esc(states[t.state])}</span></div></button>`).join('') : '<div class="empty">这里还没有任务<br>创建一组训练，开始新的实验。</div>';
  $('tasks').querySelectorAll('[data-width]').forEach(el=>el.style.width=`${el.dataset.width}%`);
}
function renderDetail() {
  const t=tasks.find(t=>t.id===selected);
  if (!t) { $('detail').innerHTML='<div class="empty">选择或新建一个训练任务</div>'; return; }
  const managed=t.mode==='managed', ended=['completed','stopped','failed'].includes(t.state);
  $('detail').classList.remove('empty');
  $('detail').innerHTML=`<div class="detail-top"><div><h2>${esc(t.name)}</h2><p class="subtle">${t.variant==='original'?'原版 · 15 厘米触须':'改进版 · 22.5 厘米触须＋绕圈扣分'}　${badge(t)}</p><p class="subtle">${managed?'已接入训练控制':'仅监控 · 接管后可调整目标及暂停'}${t.stale?'　· 日志暂未更新':''}</p></div><div class="detail-percent">${percent(t.session_progress)}<small>本次计划完成比例</small></div></div><div class="meter"><i id="detail-bar"></i></div><div class="detail-stats"><div><span>起始累计步数</span><strong>${steps(t.start_steps)}</strong></div><div><span>当前累计步数</span><strong>${steps(t.current_steps)}</strong></div><div><span>累计目标</span><strong>${steps(t.target_steps)}</strong></div></div><p class="hint">本次计划新增 ${steps(t.target_steps-t.start_steps)} 步 · 累计进度 ${percent(t.total_progress)}${t.steps_per_second ? ` · ${Math.round(t.steps_per_second)} 步/秒`:''}</p><div class="detail-bottom"><div class="hint">${active.has(t.state)?`预计剩余 ${duration(t.eta_seconds)}<br>`:''}最近更新 ${date(t.last_update)}</div><div class="controls">${!managed?'<button id="takeover" class="primary">接管控制</button>':''}<button id="change-target" class="secondary" ${!managed||ended?'disabled':''} title="${!managed?'请先接管控制':'调整累计目标'}">调整目标</button>${t.state==='paused'?'<button id="resume-task" class="primary">继续训练</button>':`<button id="pause-task" class="secondary" ${!managed||!['running','starting','queued'].includes(t.state)?'disabled':''}>暂停</button>`}<button id="stop-task" class="danger" ${!managed||ended?'disabled':''}>停止</button></div></div>${t.error?`<p class="notice error">${esc(t.error)}</p>`:''}`;
  setMeter('detail-bar',t.session_progress*100);
}
async function refresh(force=false) {
  if (overviewBusy) return;
  overviewBusy=true;
  try {
    const data=await api('/api/overview'); tasks=data.tasks; renderResources(data.resources);
    if (!selected && tasks.length) selected=tasks[0].id;
    renderTasks(); renderDetail(); $('refresh-time').textContent=`更新于 ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`;
    $('connection').classList.toggle('hidden',!data.last_error); $('connection').textContent=data.last_error ? `管理服务提示：${data.last_error}`:'';
    if (selected && (force || Date.now()-lastCharts>15000)) await loadTab();
  } catch (error) { $('connection').textContent=`连接中断，显示上次数据。请检查 SSH 转发。${error.message}`; $('connection').classList.remove('hidden'); }
  finally { overviewBusy=false; }
}
async function chooseTask(id) { selected=id; detail=null; lastCharts=0; renderTasks(); renderDetail(); await loadTab(); }
async function getDetail() { detail=await api(`/api/tasks/${selected}`); return detail; }
async function loadTab() {
  if (!selected) return;
  const id=selected;
  try {
    if (tab==='curves') await renderCharts();
    else if (tab==='logs') {
      const data=await getDetail(); if (id!==selected) return;
      const previous=$('log-attempt').value;
      $('log-attempt').innerHTML=data.attempts.map((a,i)=>`<option value="${esc(a.id)}">第 ${i+1} 次运行${a.legacy?' · 接管前':''}</option>`).join('');
      $('log-attempt').value=data.attempts.some(a=>a.id===previous)?previous:data.attempts.at(-1)?.id || '';
      await loadLogs();
    } else if (tab==='checkpoints') {
      const rows=await api(`/api/tasks/${selected}/checkpoints`); if (id!==selected) return;
      $('checkpoint-rows').innerHTML=rows.length ? rows.map(cp=>`<tr><td>${steps(cp.steps)} 步</td><td>${date(cp.at)}</td><td>${bytes(cp.size)}</td><td><button class="secondary" data-continue="${esc(cp.id)}">据此续训</button></td></tr>`).join('') : '<tr><td colspan="4">尚无完整检查点，训练到保存间隔后会出现在这里。</td></tr>';
    } else {
      const data=await getDetail(); if (id!==selected) return;
      $('event-list').innerHTML=data.events.length ? data.events.map(e=>`<div class="event"><time>${date(e.at)}</time><span>${esc(actions[e.action]||e.action)}${e.detail?.target_steps?` · 目标 ${steps(e.detail.target_steps)} 步`:''}</span></div>`).join('') : '<div class="empty">暂无操作记录</div>';
    }
  } catch(error) { toast(error.message,true); }
}
async function loadLogs() { const data=await api(`/api/tasks/${selected}/logs?attempt_id=${encodeURIComponent($('log-attempt').value)}`); $('log-text').textContent=data.text; }
async function renderCharts() {
  const id=selected;
  const chosen=$('compare').checked ? tasks.filter(t=>active.has(t.state)||t.id===selected).slice(0,2) : tasks.filter(t=>t.id===selected);
  const data=await Promise.all(chosen.map(async t=>({task:t,metrics:await api(`/api/tasks/${t.id}/metrics`)})));
  if (id!==selected) return;
  const specs=[['success','rollout/success_rate_last_500',true],['reward','rollout/ep_rew_mean',false],['loop','rollout/blank_loop_ratio',true],['collision','rollout/collision_rate',true]];
  const missing=new Set();
  for (const [name,tag,ratio] of specs) {
    const datasets=data.map((item,index)=>{
      let points=item.metrics.series[tag] || [];
      if ($('x-axis').value==='session') points=points.filter(p=>p[0]>=item.task.start_steps).map(p=>[p[0]-item.task.start_steps,p[1]]);
      if (!points.length) missing.add(item.task.name);
      return {label:item.task.name,data:points.map(p=>({x:p[0]/10000,y:ratio?p[1]*100:p[1]})),borderColor:['#7daeff','#72dbc1'][index],backgroundColor:'transparent',borderWidth:2,pointRadius:0,pointHitRadius:10,tension:0};
    });
    charts[name]?.destroy();
    const hasData=datasets.some(d=>d.data.length);
    const upper=ratio?Math.min(100,Math.max(5,Math.ceil(Math.max(0,...datasets.flatMap(d=>d.data.map(p=>p.y)))*1.15/5)*5)):undefined;
    $(`${name}-empty`).classList.toggle('hidden',hasData); $(`${name}-empty`).textContent='未记录 / 尚未生成此指标';
    charts[name]=new Chart($(`${name}-chart`),{type:'line',data:{datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,parsing:false,normalized:data.length===1,interaction:{mode:'nearest',axis:'x',intersect:false},plugins:{segments:{points:data.flatMap(item=>item.metrics.boundaries.map(b=>(b.step-($('x-axis').value==='session'?item.task.start_steps:0))/10000))},legend:{display:data.length>1,labels:{color:'#91a0b8',font:{size:11},boxWidth:12}},tooltip:{callbacks:{title:items=>items.length?`${number(items[0].parsed.x)} 万步`:''}}},scales:{x:{type:'linear',grid:{color:'#233044'},ticks:{color:'#7c90ad',maxTicksLimit:5,font:{size:11}},title:{display:true,text:$('x-axis').value==='session'?'本次新增（万步）':'累计（万步）',color:'#7487a2',font:{size:11}}},y:{min:ratio?0:undefined,max:upper,grid:{color:'#233044'},ticks:{color:'#7c90ad',maxTicksLimit:4,font:{size:11},callback:v=>ratio?`${v}%`:number(v)}}}}});
  }
  $('curve-notes').textContent=missing.size?'部分任务尚无对应指标，曲线留空；缺失数据不会按零绘制。接管及恢复的运行段在任务记录中保留。':'';
  lastCharts=Date.now();
}
async function act(action,extra={}) { if(busy) return; busy=true; try { await api(`/api/tasks/${selected}/actions`,{action,request_id:crypto.randomUUID(),...extra}); toast({pause:'暂停请求已提交，等待本轮更新完成后保存并释放资源。',resume:'已加入续训队列。',stop:'停止请求已提交，等待保存完成。',set_target:'目标已保存，将在训练边界生效。',takeover:'接管成功，已从检查点加入运行队列。'}[action]); await refresh(true); } catch(e) { toast(e.message,true); throw e; } finally {busy=false;} }
function showConfirm(title,message,info,action,label='确认') { $('confirm-title').textContent=title; $('confirm-message').textContent=message; $('confirm-info').textContent=info; $('confirm-action').textContent=label; confirmAction=action; $('confirm-dialog').showModal(); }
async function newTask(sourceId='',checkpointId='') {
  const form=$('new-form'); form.reset(); sourceCheckpoints=[];
  form.elements.source.innerHTML='<option value="">从零训练</option>'+tasks.map(t=>`<option value="${esc(t.id)}">${esc(t.name)} · 检查点续训</option>`).join('');
  form.elements.source.value=sourceId; $('new-dialog').showModal(); await updateSource(checkpointId);
}
async function updateSource(checkpointId='') {
  const f=$('new-form'), source=f.elements.source.value; $('source-checkpoint-label').classList.toggle('hidden',!source); f.elements.variant.disabled=!!source;
  sourceCheckpoints=[];
  if (source) {
    f.elements.variant.value=tasks.find(t=>t.id===source).variant; sourceCheckpoints=await api(`/api/tasks/${source}/checkpoints`);
    f.elements.checkpoint.innerHTML=sourceCheckpoints.map(cp=>`<option value="${esc(cp.id)}">${steps(cp.steps)} 步 · ${date(cp.at)}</option>`).join('');
    if(checkpointId) f.elements.checkpoint.value=checkpointId;
    const cp=sourceCheckpoints.find(c=>c.id===f.elements.checkpoint.value); f.elements.target.value=(cp?.steps||0)/10000+2000;
  }
  updateNewPreview();
}
function updateNewPreview() { const f=$('new-form'), source=f.elements.source.value, cp=sourceCheckpoints.find(c=>c.id===f.elements.checkpoint.value), start=source?(cp?.steps||0):0, target=Number(f.elements.target.value)*10000; $('new-preview').textContent=source&&!cp?'源任务尚无完整检查点，暂不能创建续训。':`起始 ${steps(start)} 步 → 累计目标 ${steps(target)} 步，计划新增 ${steps(target-start)} 步。`; }
function updateTargetPreview() { const t=goalTask,f=$('target-form'),value=Number(f.elements.amount.value)*10000,target=f.elements.mode.value==='add'?t.target_steps+value:value; $('target-preview').textContent=`当前进度 ${steps(t.current_steps)} 步；累计目标 ${steps(t.target_steps)} → ${steps(target)} 步。`; }

document.addEventListener('click',async event=>{
  const b=event.target.closest('button'); if(!b) return;
  try {
    if(b.dataset.close) $(b.dataset.close).close();
    else if(b.dataset.task) await chooseTask(b.dataset.task);
    else if(b.dataset.filter) { filter=b.dataset.filter; document.querySelectorAll('.filter').forEach(x=>x.classList.toggle('active',x===b)); renderTasks(); }
    else if(b.dataset.tab) { tab=b.dataset.tab; document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b)); document.querySelectorAll('.tab-content').forEach(x=>x.classList.toggle('hidden',x.id!==tab)); await loadTab(); }
    else if(b.dataset.continue) await newTask(selected,b.dataset.continue);
    else if(b.id==='new-task') await newTask();
    else if(b.id==='refresh') { lastCharts=0; await refresh(true); }
    else if(b.id==='log-refresh') await loadLogs();
    else if(b.id==='pause-task') await act('pause');
    else if(b.id==='resume-task') await act('resume');
    else if(b.id==='change-target') { goalTask={...tasks.find(t=>t.id===selected)}; $('target-form').reset(); $('target-form').elements.amount.value=goalTask.target_steps/10000; updateTargetPreview(); $('target-dialog').showModal(); }
    else if(b.id==='stop-task') showConfirm('保存并停止训练','任务将在本轮更新完成后保存模型并停止。之后可以选择检查点创建新的续训任务。','未保存的训练不会因正常停止而被直接丢弃。',()=>act('stop'),'保存并停止');
    else if(b.id==='takeover') { const p=await api(`/api/tasks/${selected}/takeover-preview`); showConfirm('从检查点接管控制','接管需要停止当前进程，并从完整检查点启动受管理的新运行段。原日志与模型会保留。',`当前记录 ${steps(p.current_steps)} 步；检查点 ${steps(p.checkpoint.steps)} 步；预计重新采样 ${steps(p.rollback_steps)} 步。执行前会先验证模型兼容性。`,()=>act('takeover',{checkpoint_id:p.checkpoint.id}),'验证并接管'); }
    else if(b.id==='confirm-action' && confirmAction) { b.disabled=true; try { await confirmAction(); $('confirm-dialog').close(); } finally { b.disabled=false; } }
  } catch(error) { toast(error.message,true); }
});
$('new-form').addEventListener('submit',async event=>{ event.preventDefault(); if(busy) return; busy=true; const f=event.target, button=f.querySelector('[type=submit]'); button.disabled=true; try { const body={request_id:crypto.randomUUID(),name:f.elements.name.value,variant:f.elements.variant.value,target_steps:Math.round(Number(f.elements.target.value)*10000)}; if(f.elements.seed.value!=='') body.seed=Number(f.elements.seed.value); if(f.elements.source.value) {if(!f.elements.checkpoint.value) throw Error('源任务尚无有效检查点'); body.source_task_id=f.elements.source.value; body.checkpoint_id=f.elements.checkpoint.value;} const result=await api('/api/tasks',body); selected=result.task_id; $('new-dialog').close(); toast(`任务已入队，累计目标 ${steps(result.target_steps)} 步。`); await refresh(true); } catch(e) {toast(e.message,true);} finally {busy=false;button.disabled=false;} });
$('target-form').addEventListener('submit',async event=>{event.preventDefault(); const t=goalTask,f=event.target,value=Math.round(Number(f.elements.amount.value)*10000); try {await act('set_target',{target_steps:f.elements.mode.value==='add'?t.target_steps+value:value,expected_revision:t.revision}); $('target-dialog').close();} catch(e){} });
$('new-form').elements.source.addEventListener('change',()=>updateSource().catch(e=>toast(e.message,true)));
$('new-form').addEventListener('input',updateNewPreview); $('new-form').elements.checkpoint.addEventListener('change',updateNewPreview);
$('target-form').addEventListener('input',updateTargetPreview); $('target-form').elements.mode.addEventListener('change',()=>{const f=$('target-form'); f.elements.amount.value=f.elements.mode.value==='add'?500:goalTask.target_steps/10000;updateTargetPreview();});
$('compare').addEventListener('change',()=>renderCharts().catch(e=>toast(e.message,true))); $('x-axis').addEventListener('change',()=>renderCharts().catch(e=>toast(e.message,true))); $('log-attempt').addEventListener('change',()=>loadLogs().catch(e=>toast(e.message,true)));
(async()=>{try {token=(await api('/api/bootstrap')).csrf_token; await refresh(true); setInterval(()=>refresh(),5000);} catch(e) {toast(e.message,true);}})();
