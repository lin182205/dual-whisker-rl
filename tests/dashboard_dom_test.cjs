// DOM交互验收。所有POST均在内存中拦截，不会控制服务器上的正式训练。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {JSDOM, VirtualConsole} = require('../results/dashboard_ui_qa/node_modules/jsdom');
const base = process.env.DASHBOARD_URL || 'http://127.0.0.1:8866';

(async () => {
  const overview = await (await fetch(base + '/api/overview')).json();
  assert.equal(overview.tasks.length, 2, '验收需接入当前两组只读任务');
  const snapshots = new Map([['/api/overview', overview]]);
  for (const task of overview.tasks) {
    for (const suffix of ['', '/metrics', '/checkpoints']) {
      const key = `/api/tasks/${task.id}${suffix}`;
      snapshots.set(key, await (await fetch(base + key)).json());
    }
  }
  const errors = [], posts = [], charts = [];
  const console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error.message));
  const html = fs.readFileSync(path.join(__dirname, '../dual_whisker_rl/dashboard/static/index.html'), 'utf8');
  const dom = new JSDOM(html, {url: base, runScripts:'outside-only', pretendToBeVisual:true, virtualConsole:console});
  const w = dom.window;
  w.addEventListener('error', event => errors.push(event.message));
  w.setInterval = () => 0;
  w.Chart = class {static register() {} constructor(canvas, options) {charts.push(options);} destroy() {}};
  Object.defineProperty(w.crypto, 'randomUUID', {value:crypto.randomUUID});
  if (!w.HTMLDialogElement.prototype.showModal) w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
  if (!w.HTMLDialogElement.prototype.close) w.HTMLDialogElement.prototype.close=function(){this.open=false;};
  let managed = false, paused = false, sessionExpired = false, bootstrapCount = 0;
  w.fetch = async (url, options={}) => {
    if (options.method === 'POST') {
      if (sessionExpired) {sessionExpired=false;return {ok:false,status:403,json:async()=>({error:'页面会话已失效，请刷新页面'})};}
      const body = JSON.parse(options.body);
      assert.ok(options.headers['X-CSRF-Token']);
      posts.push({url,body});
      return {ok:true,json:async()=>({task_id:overview.tasks[0].id,target_steps:body.target_steps,state:'queued'})};
    }
    if (url === '/api/bootstrap') {bootstrapCount++;return {ok:true,json:async()=>({csrf_token:'isolated-dom-test'})};}
    let value = structuredClone(snapshots.get(url));
    if (url === '/api/overview' && managed) value.tasks[0] = {...value.tasks[0],mode:'managed',state:paused?'paused':'running'};
    if (value === undefined) throw Error('Unexpected GET ' + url);
    return {ok:true,json:async()=>value};
  };
  const wait = async predicate => {
    for (let i=0;i<150;i++) {if(predicate()) return;await new Promise(resolve=>setTimeout(resolve,20));}
    throw Error('DOM验收等待超时；错误：' + errors.join('; '));
  };
  const click = selector => {const el=w.document.querySelector(selector);assert.ok(el,selector);el.click();};
  const input = (element,value,event='input') => {element.value=value;element.dispatchEvent(new w.Event(event,{bubbles:true}));};
  try {
    w.eval(fs.readFileSync(path.join(__dirname, '../dual_whisker_rl/dashboard/static/app.js'), 'utf8'));
    await wait(()=>charts.length===4);
    assert.equal(w.document.querySelectorAll('.task-card').length,2);
    assert.ok(w.document.querySelector('#change-target').disabled, '旧任务必须先接管');
    assert.ok(!w.document.querySelector('#loop-empty').classList.contains('hidden'), '旧版缺失指标不能填零');
    assert.ok(w.document.querySelector('#success-empty').classList.contains('hidden'));
    assert.ok(w.document.querySelector('#detail').textContent.replaceAll(',','').includes('3500'));

    const improved = overview.tasks.find(task=>task.variant==='improved');
    click(`[data-task="${improved.id}"]`);
    await wait(()=>charts.length===8);
    assert.ok(w.document.querySelector('#loop-empty').classList.contains('hidden'));
    click('#compare');
    await wait(()=>charts.length===12);
    assert.equal(charts.at(-1).data.datasets.length,2);

    click('#new-task');
    await wait(()=>w.document.querySelector('#new-dialog').open);
    const form=w.document.querySelector('#new-form');
    input(form.elements.name,'DOM隔离验收，不发送到服务器');
    input(form.elements.target,'2000');
    assert.ok(w.document.querySelector('#new-preview').textContent.replaceAll(',','').includes('2000'));
    form.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
    await wait(()=>posts.length===1 && !w.document.querySelector('#new-dialog').open);
    assert.equal(posts[0].body.target_steps,20000000);
    assert.equal(posts[0].body.variant,'improved');

    // 客户端内存中的管理模式样本，仅用于测试编辑对话框和恢复按钮。
    managed=true;
    click('#refresh');
    await wait(()=>w.document.querySelector('#change-target') && !w.document.querySelector('#change-target').disabled);
    click('#change-target');
    const target=w.document.querySelector('#target-form');
    input(target.elements.mode,'add','change');
    input(target.elements.amount,'500');
    assert.ok(w.document.querySelector('#target-preview').textContent.replaceAll(',','').includes('4000'));
    sessionExpired=true;
    target.dispatchEvent(new w.Event('submit',{bubbles:true,cancelable:true}));
    await wait(()=>posts.length===2 && !w.document.querySelector('#target-dialog').open);
    assert.equal(posts[1].body.action,'set_target');
    assert.equal(posts[1].body.target_steps,40000000);
    assert.equal(bootstrapCount,2,'服务重启后的会话应自动更新，且请求不重复执行');

    paused=true;
    click('#refresh');
    await wait(()=>w.document.querySelector('#resume-task'));
    click('#resume-task');
    await wait(()=>posts.length===3);
    assert.equal(posts[2].body.action,'resume');
    assert.deepEqual(errors,[]);
    process.stdout.write('DOM验收通过：真实数据渲染、缺失曲线、双组对比、新建表单、追加目标、恢复操作。所有POST仅在内存中模拟。\n');
  } finally {
    // 等待点击处理器的Promise链完成，再销毁DOM，避免在清理阶段访问已关闭页面。
    await new Promise(setImmediate);
    dom.window.close();
  }
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
