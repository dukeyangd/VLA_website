let state={hosts:[],tasks:[],collections:[],conversions:[],replay_jobs:[],replay_issues:{},replay_config:{},skills:[],skills_config:{}},sharedState={connected:false,tasks:[],sessions:[]};
let review=null,currentEp=null,filter='all',fieldOptions=[],manifest=null,uploadingCollection=null,lockState=null,lockTimer=null,remotePreview=null,stateLoading=false;
let uploadSkillMode='existing',selectedUploadSkillId='';
let replayDataset=null,replayIssues=null,replaySelectedEps=new Set(),replayPickMode='range',activeReplayJob=null,replayConfig={presets:[]},viserUrl='http://127.0.0.1:8081',replayExperiments=[],replayExecution='local',viserFrameLoaded=false,replayFocusEp=null,viserFailStreak=0;
/** 01→02：仅 valid 进入筛选网格；invalid 只写进终端提示 */
let replayLabelFilter=null,replayLabelTip='';
let selectedReplaySkillId='',replaySkillDatasets=[];
const clientId=localStorage.getItem('humanoidStudioClientId')||crypto.randomUUID();localStorage.setItem('humanoidStudioClientId',clientId);
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,options={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...options});const text=await r.text();let x={};if(text){try{x=JSON.parse(text)}catch{x={message:text}}}if(!r.ok||x.ok===false)throw new Error(x.message||text||'请求失败');return x}
function toast(msg,bad=false){const el=$('toast');el.textContent=msg;el.className=bad?'show error':'show';clearTimeout(el._t);el._t=setTimeout(()=>el.className='',2600)}
function fmtBytes(n=0){if(n<1024)return n+' B';if(n<1048576)return(n/1024).toFixed(1)+' KB';if(n<1073741824)return(n/1048576).toFixed(1)+' MB';return(n/1073741824).toFixed(2)+' GB'}
function post(url,data={}){return api(url,{method:'POST',body:JSON.stringify(data)})}
function del(url,data={}){return api(url,{method:'DELETE',body:JSON.stringify(data)})}

let uploadProgressById={};
let uploadProgressTimer=null;

function uploadProgressHtml(c){
  if(!c||c.status!=='uploading')return '';
  const p=uploadProgressById[c.id]||{};
  const pct=Number(p.pct??c.upload_progress);
  const done=Number(p.done??c.upload_bytes_done??0);
  const total=Number(p.total??c.bytes_total??0);
  const known=Number.isFinite(pct)&&pct>=0;
  const width=known?Math.max(0,Math.min(100,pct)):0;
  const label=known
    ?`${Math.round(width)}% · ${fmtBytes(done)}${total?` / ${fmtBytes(total)}`:''}${p.speed?` · ${esc(p.speed)}`:''}`
    :`上传中…${total?` · 共 ${fmtBytes(total)}`:''}`;
  return `<div class="upload-progress${known?'':' is-indeterminate'}" data-upload-bar="${esc(c.id)}">
    <div class="upload-progress-track"><i style="width:${width}%"></i></div>
    <small>${label}</small>
  </div>`;
}
async function refreshUploadProgressOverlay(){
  const uploading=(state.collections||[]).filter(c=>c.status==='uploading');
  if(!uploading.length){
    uploadProgressById={};
    if(uploadProgressTimer){clearInterval(uploadProgressTimer);uploadProgressTimer=null}
    return;
  }
  for(const c of uploading){
    if(c.upload_progress!=null){
      uploadProgressById[c.id]={
        pct:Number(c.upload_progress)||0,
        done:Number(c.upload_bytes_done)||0,
        total:Number(c.bytes_total)||0,
        speed:c.upload_speed||'',
      };
    }
  }
  try{
    const r=await fetch(`/static/upload-progress.json?t=${Date.now()}`,{cache:'no-store'});
    if(r.ok){
      const data=await r.json();
      const items=data.items||data;
      if(items&&typeof items==='object'){
        for(const [id,p] of Object.entries(items)){
          if(!p||typeof p!=='object')continue;
          uploadProgressById[id]={
            pct:Number(p.pct??p.upload_progress)||0,
            done:Number(p.done??p.upload_bytes_done)||0,
            total:Number(p.total??p.bytes_total)||0,
            speed:p.speed||p.upload_speed||'',
          };
        }
      }
    }
  }catch(_){}
  // Update bars in-place when possible
  document.querySelectorAll('[data-upload-bar]').forEach(el=>{
    const id=el.dataset.uploadBar;
    const p=uploadProgressById[id];
    if(!p)return;
    const pct=Math.max(0,Math.min(100,Number(p.pct)||0));
    el.classList.remove('is-indeterminate');
    const bar=el.querySelector('i');
    if(bar)bar.style.width=`${pct}%`;
    const small=el.querySelector('small');
    if(small)small.textContent=`${Math.round(pct)}% · ${fmtBytes(p.done||0)}${p.total?` / ${fmtBytes(p.total)}`:''}${p.speed?` · ${p.speed}`:''}`;
  });
  if(!uploadProgressTimer)uploadProgressTimer=setInterval(()=>{void refreshUploadProgressOverlay()},2500);
}
function ensureUploadProgressPolling(){
  const uploading=(state.collections||[]).some(c=>c.status==='uploading');
  if(uploading)void refreshUploadProgressOverlay();
  else if(uploadProgressTimer){clearInterval(uploadProgressTimer);uploadProgressTimer=null}
}

let selectedManagedSkillId='';
let selectedCollectSkillId=localStorage.getItem('studioCollectSkillId')||'';

document.querySelectorAll('.nav').forEach(b=>b.onclick=()=>{
  try{
    const page=b.dataset.page;
    const target=$('page-'+page);
    if(!page||!target)return;
    const leavingCollect=$('page-collect')?.classList.contains('active')&&page!=='collect';
    const leavingInfer=$('page-infer')?.classList.contains('active')&&page!=='infer';
    if(leavingCollect){void releaseCurrentLock();for(const p of[$('player0'),$('player1')])p?.pause?.();stopCollectCameras()}
    if(leavingInfer)stopInferPreviewMedia();
    document.querySelectorAll('.nav,.page').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    target.classList.add('active');
    try{localStorage.setItem('studioActivePage', page)}catch(_){}
    const names={
      skills:['技能卡 · 管理','技能卡管理'],
      collect:['01 · 采集打标','采集'],
      replay:['02 · 筛选','筛选'],
      train:['03 · 训练','训练'],
      infer:['04 · 推理','推理'],
    };
    if(names[page]){
      if($('eyebrow'))$('eyebrow').textContent=names[page][0];
      if($('pageTitle'))$('pageTitle').textContent=names[page][1];
    }
    if(page==='skills')initSkillsManagePage();
    if(page==='collect'){renderCollectSkillGrid();updateCollectFlowUI();fillCollectConfigForm();startCollectStackPolling();if($('reviewCollection')?.value)loadReview()}
    if(page==='replay')initReplayPage();
    if(page==='train')initTrainPage();
    if(page==='infer')initInferPage();
    window.scrollTo({top:0,behavior:'smooth'});
  }catch(e){console.error('nav switch failed',e);toast(e.message||'切换模块失败',true)}
});
function restoreActivePage(){
  let page='collect';
  try{page=localStorage.getItem('studioActivePage')||'collect'}catch(_){}
  if(page==='collect')return;
  const btn=document.querySelector(`.nav[data-page="${page}"]`);
  if(btn)btn.click();
}

/* Dock magnification disabled: scaled buttons overlap neighbors and steal clicks. */

async function loadState(){if(stateLoading)return;stateLoading=true;const selected=$('reviewCollection')?.value,currentId=currentEp?.id;try{state=await api('/api/state');if(review&&!review.shared){const fresh=state.collections.find(x=>x.id===review.id);if(fresh){review=fresh;currentEp=currentId?fresh.episodes.find(x=>x.id===currentId)||null:null}}renderAll();if(selected)$('reviewCollection').value=selected;sharedState=await api('/api/shared/state').catch(e=>({ok:true,connected:false,error:e.message,tasks:sharedState.tasks||[],sessions:sharedState.sessions||[],updated_at:sharedState.updated_at}));if(review?.shared){const fresh=sharedState.sessions.find(x=>x.id===review.id);if(fresh){const keepEps=(review.episodes&&review.episodes.length)?review.episodes:(fresh.episodes||[]);review={...fresh,shared:true,source_type:'shared',episodes:keepEps};currentEp=currentId?keepEps.find(x=>x.id===currentId)||null:null}}renderAll();if(selected){$('reviewCollection').value=selected;renderEpisodes()}}catch(e){toast(e.message,true)}finally{stateLoading=false}}
function options(items,value='id',label='name'){return items.map(x=>`<option value="${esc(x[value])}">${esc(x[label])}</option>`).join('')}
function resolveReplayHostId(){
  const raw=String($('replayHost')?.value||'').trim();
  if(raw&&state?.hosts?.some(h=>h.id===raw))return raw;
  const cfg=state?.replay_config?.host_id||'cluster_0';
  if(cfg&&state?.hosts?.some(h=>h.id===cfg))return cfg;
  return state?.hosts?.[0]?.id||'cluster_0';
}
function renderAll(){
  // Only fill real <select> host pickers. replayHost is a hidden <input> — setting
  // innerHTML on it can wipe value and break prune / probe ("远端模式需要有效 Host").
  ['taskHost','importHost','uploadHost','sharedHost'].forEach(id=>{const el=$(id);if(!el||el.tagName!=='SELECT')return;const old=el.value;el.innerHTML=options(state.hosts);if(state.hosts.some(x=>x.id===old))el.value=old});
  if($('replayHost'))$('replayHost').value=resolveReplayHostId();
  const skills=state.skills||[];
  if(selectedCollectSkillId&&!skills.some(s=>s.id===selectedCollectSkillId))selectedCollectSkillId='';
  ensureCollectSkillSelected();
  if($('collectionTask'))$('collectionTask').value=selectedCollectSkillId||'';
  const oldImportTask=$('importTask').value;$('importTask').innerHTML='<option value="">请选择任务</option>'+options(state.tasks);if(state.tasks.some(x=>x.id===oldImportTask))$('importTask').value=oldImportTask;
  const remoteTask=$('remotePublishTask');if(remoteTask){const old=remoteTask.value,items=sharedState.tasks||[];remoteTask.innerHTML='<option value="">请选择共享任务</option>'+items.map(x=>`<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');if(items.some(x=>x.name===old))remoteTask.value=old}
  if(state.shared){$('sharedHost').value=state.shared.host_id;$('sharedRoot').value=state.shared.root;$('sharedEnabled').checked=state.shared.enabled!==false}
  if(state.skills_config?.skills_root&&$('taskRemote')&&!$('taskRemote').dataset.touched)$('taskRemote').value=state.skills_config.skills_root;
  const allTasks=new Set([...(skills.map(x=>x.id)),...state.tasks.map(x=>x.name),...(sharedState.tasks||[]).map(x=>x.name)]);$('mTasks').textContent=allTasks.size;$('mRunning').textContent=state.collections.filter(x=>['collecting_offline','stopping_offline','uploading','publishing','running','syncing','stopping','importing'].includes(x.status)).length;$('mBytes').textContent=fmtBytes(state.collections.reduce((n,x)=>n+(x.bytes_total||0),0));$('mReview').textContent=state.collections.reduce((n,x)=>n+(x.episodes||[]).filter(e=>e.status==='unreviewed').length,0)+(sharedState.sessions||[]).reduce((n,x)=>n+(x.counts?.unreviewed||0),0);
  const badge=$('sharedConnection');if(badge){badge.textContent='在线';badge.className='connection-badge online quiet'}renderCollections();renderSharedSessions();renderReviewSelect({autoBind:true});renderHosts();
  if(replayDataset?.path&&state.replay_issues?.[replayDataset.path]){
    replayIssues=state.replay_issues[replayDataset.path];
  }
  renderReplayJobs();renderReplayIssues();updateReplayMetrics();renderCollectSkillGrid();updateCollectFlowUI();fillCollectConfigForm();
}
function renderCollectSkillGrid(){
  const el=$('collectSkillGrid');if(!el)return;
  const skills=state.skills||[];
  if(selectedCollectSkillId&&!skills.some(s=>s.id===selectedCollectSkillId)){
    selectedCollectSkillId='';
    localStorage.removeItem('studioCollectSkillId');
  }
  if($('collectionTask'))$('collectionTask').value=selectedCollectSkillId||'';
  syncCollectRootUI();
  if(!skills.length){
    el.className='skill-pick-grid empty';
    el.textContent='暂无技能卡 — 点右上角「新建技能卡」创建';
    return;
  }
  el.className='skill-pick-grid';
  el.innerHTML=skills.map(s=>{
    const sel=s.id===selectedCollectSkillId?' selected':'';
    return `<button type="button" class="skill-pick-card${sel}" onclick="selectCollectSkill('${esc(s.id)}')"><span class="badge">${esc(s.badge||s.id)}</span><h3>${esc(s.title||s.id)}</h3><p>${esc(s.description||s.prompt||'')}</p><div class="meta"><span class="pill">${(s.sessions||[]).length} sessions</span><span class="pill">${(s.datasets||[]).length} 数据集</span></div></button>`;
  }).join('');
}
function selectCollectSkill(id){
  selectedCollectSkillId=id;
  localStorage.setItem('studioCollectSkillId',id);
  if($('collectionTask'))$('collectionTask').value=id;
  renderCollectSkillGrid();
  syncCollectRootUI();
  renderCollections();
  renderReviewSelect({autoBind:true});
  updateCollectFlowUI();
}
function toggleCollectSkillCreate(force){
  const pane=$('collectSkillCreate');if(!pane)return;
  pane.hidden=typeof force==='boolean'?!force:!pane.hidden;
}
async function createCollectSkill(){
  const title=($('collectNewTitle')?.value||'').trim();
  if(!title)return toast('请填写技能名称',true);
  try{
    const x=await post('/api/skills',{
      title,
      id:($('collectNewId')?.value||'').trim(),
      badge:($('collectNewBadge')?.value||'').trim(),
      prompt:($('collectNewPrompt')?.value||'').trim(),
      description:($('collectNewPrompt')?.value||'').trim(),
      collect_root:($('collectNewRoot')?.value||'').trim(),
      host_id:'cluster_0',
      source:'collect_page',
    });
    selectedCollectSkillId=x.skill?.id||'';
    if(selectedCollectSkillId)localStorage.setItem('studioCollectSkillId',selectedCollectSkillId);
    ['collectNewTitle','collectNewId','collectNewBadge','collectNewPrompt','collectNewRoot'].forEach(id=>{if($(id))$(id).value=''});
    toggleCollectSkillCreate(false);
    toast(`已创建 ${x.skill?.title||x.skill?.id} · collect_root 已就绪`);
    await refreshCollectSkills();
    syncCollectRootUI();
  }catch(e){toast(e.message,true)}
}
function currentCollectSkill(){
  return (state.skills||[]).find(s=>s.id===selectedCollectSkillId)||null;
}
function syncCollectRootUI(){
  const skill=currentCollectSkill();
  const root=skill?.collect_root||'';
  if($('collectRootInput'))$('collectRootInput').value=root;
  if($('collectPathHint')){
    $('collectPathHint').textContent=root
      ?`采集写入：${root} / <时间戳> · 后续数采脚本可读 STUDIO_SESSION_DIR`
      :'选技能后自动在 collect_root 下建时间戳目录';
  }
  const cols=skillCollections(selectedCollectSkillId);
  const pending=pendingUploadCollections(selectedCollectSkillId).length;
  if($('collectSkillHint')){
    if(!skill)$('collectSkillHint').textContent='尚未选择技能卡 — 点上方卡片后，下方会列出该技能全部数据集状态';
    else $('collectSkillHint').textContent=`已选 ${skill.title||skill.id} · ${cols.length} 个本机会话 · ${pending} 个待上传 · 见下方「数据集状态」`;
  }
}
async function saveCollectRoot(){
  if(!selectedCollectSkillId)return toast('请先选择技能卡',true);
  const root=($('collectRootInput')?.value||'').trim();
  if(!root)return toast('请填写本机采集根目录',true);
  try{
    const x=await api(`/api/skills/${encodeURIComponent(selectedCollectSkillId)}`,{method:'PATCH',body:JSON.stringify({collect_root:root})});
    const idx=(state.skills||[]).findIndex(s=>s.id===selectedCollectSkillId);
    if(idx>=0)state.skills[idx]=x.skill; else state.skills=[...(state.skills||[]),x.skill];
    toast('采集根目录已保存');
    syncCollectRootUI();
  }catch(e){toast(e.message,true)}
}
function skillCollections(skillId){
  if(!skillId)return [];
  const skill=(state.skills||[]).find(s=>s.id===skillId);
  const roots=[skill?.collect_root,skill?.remote_folder,skill?.ref_root].filter(Boolean).map(String);
  const taskName=String(skill?.task_name||'');
  return (state.collections||[]).filter(x=>{
    if(x.source_type==='imported'||x.status==='published')return false;
    const sid=String(x.skill_id||'');
    const tname=String(x.task_name||'');
    if(sid===skillId||tname===skillId)return true;
    if(taskName&&(tname===taskName||sid===taskName))return true;
    const local=String(x.local_dir||'');
    if(roots.some(r=>local===r||local.startsWith(String(r).replace(/\/+$/,'')+'/')))return true;
    return false;
  });
}
function pendingUploadCollections(skillId){
  return skillCollections(skillId).filter(x=>['pending_upload','upload_error'].includes(x.status));
}
function pickActiveSkillCollection(skillId){
  const items=skillCollections(skillId);
  const rank={collecting_offline:0,stopping_offline:1,pending_upload:2,upload_error:3,interrupted:8};
  return [...items].sort((a,b)=>(rank[a.status]??20)-(rank[b.status]??20)||String(b.created_at||'').localeCompare(String(a.created_at||'')))[0]||null;
}
function pipelineStatusLabel(status){
  return({
    collecting_offline:'采集中',
    stopping_offline:'结束中 / 生成 labels',
    pending_upload:'采集结束 · 待上传',
    uploading:'上传中',
    upload_error:'上传失败',
    uploaded:'已上传',
    published:'已发布共享',
    interrupted:'已中断',
    error:'错误',
    mounted:'已挂载（历史/远端）',
  })[status]||statusLabel(status);
}
function ensureCollectSkillSelected(){
  const skills=state.skills||[];
  if(!skills.length)return;
  if(selectedCollectSkillId&&skills.some(s=>s.id===selectedCollectSkillId))return;
  // Prefer a skill that still has pending uploads / live collect.
  const ranked=skills.map(s=>{
    const cols=skillCollections(s.id);
    const score=cols.some(c=>['collecting_offline','stopping_offline'].includes(c.status))?0
      :cols.some(c=>['pending_upload','upload_error'].includes(c.status))?1
      :cols.length?2:3;
    return {id:s.id,score,n:cols.length};
  }).sort((a,b)=>a.score-b.score||b.n-a.n);
  selectedCollectSkillId=ranked[0]?.id||skills[0].id;
  localStorage.setItem('studioCollectSkillId',selectedCollectSkillId);
  if($('collectionTask'))$('collectionTask').value=selectedCollectSkillId;
}
function updateCollectFlowUI(){
  const active=pickActiveSkillCollection(selectedCollectSkillId);
  const collecting=active&&['collecting_offline','stopping_offline'].includes(active.status)&&!!active.worker_alive;
  const staleStopping=active&&active.status==='stopping_offline'&&!active.worker_alive;
  const pending=pendingUploadCollections(selectedCollectSkillId);
  const readyUpload=pending.length>0;
  if($('collectStopBtn'))$('collectStopBtn').disabled=!collecting;
  if($('reviewFinishBtn'))$('reviewFinishBtn').hidden=!collecting;
  ['reviewUploadBtn','reviewUploadBtnTop'].forEach(id=>{
    const btn=$(id);if(!btn)return;
    btn.hidden=!readyUpload;
    btn.textContent=pending.length>1?`上传到 cluster_0（${pending.length}）`:'上传到 cluster_0';
  });
  const banner=$('reviewFlowBanner');
  if(banner){
    if(!selectedCollectSkillId){banner.className='scan-preview empty';banner.textContent='先选择技能并开始采集，Episode 会实时出现在下方'}
    else if(staleStopping){
      banner.className='scan-preview';
      banner.innerHTML=`<b>${esc(active.session_name||active.id)}</b><span>结束流程中断（Studio 重启）· 请在上方数据集状态栏点「强制结束并生成 labels」</span><span>${esc(active.local_dir||'')}</span>`;
    }else if(active){
      banner.className='scan-preview';
      const uploadHint=readyUpload?` · ${pending.length} 个待上传`:'';
      banner.innerHTML=`<b>${esc(active.session_name||active.local_dir?.split('/').pop()||active.id)}</b><span>${statusLabel(active.status)}${uploadHint} · ${active.file_count||0} 文件 · ${(active.episodes||[]).length} episodes</span><span>${esc(active.local_dir||'')}</span><span>${esc(active.message||'')}</span>`;
    }else{
      banner.className='scan-preview empty';
      banner.textContent=pending.length?'上方可直接上传历史会话 · 或开始新的离线数采':'当前技能暂无进行中会话 — 可点「开始离线数采」，或在上方查看已挂载数据集';
    }
  }
  const envEl=$('collectLaunchEnv');
  if(envEl){
    const env=active?.launch_env;
    if(env?.STUDIO_SESSION_DIR && collecting){
      envEl.hidden=false;
      envEl.textContent=Object.entries(env).map(([k,v])=>`${k}=${v}`).join('\n');
    }else if(!collecting){
      envEl.hidden=true;
    }
  }
  renderCollectStackStatus(active?.stack);
  if(collecting){bindCollectCameras();setCollectControlsEnabled(true);startCollectStackPolling()}
  else{setCollectControlsEnabled(false)}
}
function renderCollections(){
  const el=$('collectionList');if(!el)return;
  ensureCollectSkillSelected();
  if(!selectedCollectSkillId){el.className='empty';el.textContent='请先选择技能卡';return}
  const cols=skillCollections(selectedCollectSkillId);
  const skill=(state.skills||[]).find(s=>s.id===selectedCollectSkillId);
  const datasets=Array.isArray(skill?.datasets)?skill.datasets:[];
  const byLocal=new Map(cols.map(c=>[String(c.local_dir||'').replace(/\/+$/,''),c]));
  const byId=new Map(cols.map(c=>[String(c.id),c]));
  const rows=[];
  for(const c of cols){
    const eps=(c.episodes||[]);
    const valid=eps.filter(e=>e.status==='valid').length;
    const invalid=eps.filter(e=>e.status==='invalid').length;
    const unrev=eps.filter(e=>e.status==='unreviewed').length;
    rows.push({
      key:c.id,
      kind:'collection',
      title:c.session_name||c.local_dir?.replace(/\/+$/,'').split('/').pop()||c.id,
      path:c.local_dir||'',
      status:c.status,
      meta:`${c.file_count||0} 文件 · ${eps.length||0} ep · ${valid}✓ ${invalid}✗ ${unrev}?`,
      msg:shortMsg(c.message),
      collection:c,
    });
  }
  for(const d of datasets){
    const path=String(d.local_dir||d.path||d.remote_path||'').replace(/\/+$/,'');
    const cid=String(d.collection_id||'');
    if((path&&byLocal.has(path))||(cid&&byId.has(cid)))continue;
    rows.push({
      key:d.id||path,
      kind:'dataset',
      title:d.label||d.id||path.split('/').pop()||'dataset',
      path:path||d.remote_path||'',
      status:d.status||'mounted',
      meta:`${d.total_episodes!=null?d.total_episodes+' eps':''}${d.source?` · ${d.source}`:''}${d.ready?' · 就绪':''}`.replace(/^ · /,''),
      msg:d.remote_path&&d.remote_path!==path?d.remote_path:'',
      dataset:d,
    });
  }
  if(!rows.length){
    el.className='empty';
    el.textContent='当前技能还没有本机会话或挂载数据集 — 开始采集后会出现在这里';
    return;
  }
  const rank={upload_error:0,pending_upload:1,uploading:2,stopping_offline:3,collecting_offline:4,interrupted:5,mounted:9};
  rows.sort((a,b)=>(rank[a.status]??20)-(rank[b.status]??20)||String(b.title).localeCompare(String(a.title)));
  el.className='collect-dataset-board';
  el.innerHTML=rows.map(r=>{
    const st=pipelineStatusLabel(r.status);
    const error=['error','upload_error','interrupted'].includes(r.status);
    const pending=['pending_upload','upload_error'].includes(r.status);
    const uploading=r.status==='uploading';
    const pathAttr=encodeURIComponent(r.path||'');
    const dsIdAttr=encodeURIComponent(r.dataset?.id||'');
    const mountDel=`<button type="button" class="ghost danger-ghost" onclick="removeBoardDataset(decodeURIComponent('${pathAttr}'),decodeURIComponent('${dsIdAttr}'))" title="从技能卡移除挂载">删除</button>`;
    const actions=r.collection?collectionActions(r.collection):(pending?`<button class="primary green" disabled title="无本机 collection 记录">仅挂载</button>${mountDel}`:`${mountDel}`);
    const progress=r.collection?uploadProgressHtml(r.collection):'';
    return `<div class="session-row collect-ds-row ${error?'is-error':''}${pending?' is-pending':''}${uploading?' is-uploading':''}">
      <div><div class="session-title">${esc(r.title)} <span class="pill">${r.kind==='collection'?'本机会话':'技能挂载'}</span></div>
      <small title="${esc(r.path)}">${esc(r.path||'—')}</small></div>
      <div><b>${esc(r.meta||'—')}</b><small title="${esc(r.msg)}">${esc(r.msg||'')}</small></div>
      <div class="session-status-col"><span class="session-state ${error?'error':''}${pending?' ready':''}${uploading?' ready':''}">${esc(st)}</span>${progress}</div>
      <div class="session-actions">${actions}</div>
    </div>`;
  }).join('');
  ensureUploadProgressPolling();
}
async function refreshCollectSkills(){
  try{const x=await api('/api/skills');state.skills=x.skills||[];state.skills_config=x.config||state.skills_config;renderAll();toast(`已加载 ${state.skills.length} 个技能`)}catch(e){toast(e.message,true)}
}
async function initSkillsManagePage(){await refreshManagedSkills()}
async function refreshManagedSkills(){
  try{
    const x=await api('/api/skills');
    state.skills=x.skills||[];
    state.skills_config=x.config||state.skills_config;
    if(selectedManagedSkillId&&!(state.skills||[]).some(s=>s.id===selectedManagedSkillId))selectedManagedSkillId='';
    renderManagedSkillGrid();
    await renderManagedDatasets();
    renderAll();
  }catch(e){toast(e.message,true);renderManagedSkillGrid()}
}
function renderManagedSkillGrid(){
  const el=$('mgmtSkillGrid');if(!el)return;
  const skills=state.skills||[];
  if($('mgmtSkillCount'))$('mgmtSkillCount').textContent=`${skills.length} 张`;
  const delBtn=$('mgmtDeleteSkillBtn'),addBtn=$('mgmtAddDatasetBtn');
  if(delBtn)delBtn.disabled=!selectedManagedSkillId;
  if(addBtn)addBtn.disabled=!selectedManagedSkillId;
  if(!skills.length){el.className='skill-pick-grid empty';el.textContent='还没有技能卡，请在左侧创建';return}
  el.className='skill-pick-grid';
  el.innerHTML=skills.map(s=>{
    const sel=s.id===selectedManagedSkillId?' selected':'';
    const n=(s.datasets||[]).length;
    return `<button type="button" class="skill-pick-card${sel}" onclick="selectManagedSkill('${esc(s.id)}')"><span class="badge">${esc(s.badge||s.id)}</span><h3>${esc(s.title||s.id)}</h3><p>${esc(s.description||s.prompt||'')}</p><div class="meta"><span class="pill">${n} 数据集</span><span class="pill">${esc(s.id)}</span></div></button>`;
  }).join('');
}
async function selectManagedSkill(id){
  selectedManagedSkillId=id;
  renderManagedSkillGrid();
  await renderManagedDatasets();
}
async function renderManagedDatasets(){
  const el=$('mgmtDatasetList');if(!el)return;
  const hint=$('mgmtDatasetHint');
  if(!selectedManagedSkillId){
    el.className='mgmt-dataset-list empty-state';el.textContent='尚未选择技能';
    if(hint)hint.textContent='点选上方技能卡后，在此增删挂载路径（不删远端文件）';
    return;
  }
  try{
    const x=await api(`/api/skills/${encodeURIComponent(selectedManagedSkillId)}/datasets`);
    const skill=x.skill||(state.skills||[]).find(s=>s.id===selectedManagedSkillId);
    const datasets=x.datasets||[];
    if(hint)hint.textContent=`当前：${skill?.title||selectedManagedSkillId} · ${datasets.length} 条挂载`;
    if(!datasets.length){el.className='mgmt-dataset-list empty-state';el.textContent='该技能尚未挂载数据集';return}
    el.className='mgmt-dataset-list';
    el.innerHTML=datasets.map(d=>{
      const path=d.path||'';
      const ready=d.ready?'ready':'not-ready';
      const pathAttr=encodeURIComponent(path);
      return `<div class="mgmt-dataset-row"><div><h4>${esc(d.label||d.id||'dataset')}</h4><code>${esc(path)}</code><div class="meta"><span class="pill ${ready}">${d.ready?'就绪':'未就绪'}</span>${d.total_episodes!=null?`<span class="pill">${d.total_episodes} eps</span>`:''}${d.source?`<span class="pill">${esc(d.source)}</span>`:''}</div></div><button type="button" class="ghost" onclick="removeManagedDataset(decodeURIComponent('${pathAttr}'))">移除</button></div>`;
    }).join('');
  }catch(e){el.className='mgmt-dataset-list empty-state';el.textContent=e.message}
}
async function createManagedSkill(){
  const title=($('mgmtSkillTitle')?.value||'').trim();
  if(!title)return toast('请填写技能名称',true);
  try{
    const x=await post('/api/skills',{
      title,
      id:($('mgmtSkillId')?.value||'').trim(),
      badge:($('mgmtSkillBadge')?.value||'').trim(),
      prompt:($('mgmtSkillPrompt')?.value||'').trim(),
      description:($('mgmtSkillPrompt')?.value||'').trim(),
      host_id:'cluster_0',
    });
    selectedManagedSkillId=x.skill?.id||'';
    ['mgmtSkillTitle','mgmtSkillId','mgmtSkillBadge','mgmtSkillPrompt'].forEach(id=>{if($(id))$(id).value=''});
    toast(`已创建 ${x.skill?.title||x.skill?.id}`);
    await refreshManagedSkills();
  }catch(e){toast(e.message,true)}
}
async function deleteManagedSkill(){
  if(!selectedManagedSkillId)return toast('请先选择技能卡',true);
  const skill=(state.skills||[]).find(s=>s.id===selectedManagedSkillId);
  if(!confirm(`确认删除技能卡「${skill?.title||selectedManagedSkillId}」？\n\n只会删除本机技能目录与挂载记录，不会删除 cluster 上的数据集文件。`))return;
  try{
    await del(`/api/skills/${encodeURIComponent(selectedManagedSkillId)}`);
    toast('技能卡已删除');
    selectedManagedSkillId='';
    await refreshManagedSkills();
  }catch(e){toast(e.message,true)}
}
async function addManagedDataset(){
  if(!selectedManagedSkillId)return toast('请先选择技能卡',true);
  const path=($('mgmtDatasetPath')?.value||'').trim();
  const label=($('mgmtDatasetLabel')?.value||'').trim();
  if(!path)return toast('请填写数据集绝对路径',true);
  try{
    await post(`/api/skills/${encodeURIComponent(selectedManagedSkillId)}/datasets`,{
      path,label,remote_path:path,host_id:'cluster_0',probe:true,
    });
    if($('mgmtDatasetPath'))$('mgmtDatasetPath').value='';
    if($('mgmtDatasetLabel'))$('mgmtDatasetLabel').value='';
    toast('已添加到技能卡');
    await refreshManagedSkills();
  }catch(e){toast(e.message,true)}
}
async function removeManagedDataset(path){
  if(!selectedManagedSkillId)return;
  if(!confirm(`从技能卡移除该数据集挂载？\n\n${path}\n\n不会删除磁盘上的数据。`))return;
  try{
    await del(`/api/skills/${encodeURIComponent(selectedManagedSkillId)}/datasets`,{path});
    toast('已移除挂载');
    await refreshManagedSkills();
  }catch(e){toast(e.message,true)}
}
function statusLabel(s){return({collecting_offline:'离线采集中',stopping_offline:'正在结束',pending_upload:'待上传',uploading:'补传中',publishing:'发布中',published:'已发布共享',publish_error:'待重新发布',uploaded:'已上传',upload_error:'上传失败',running:'启动中',syncing:'实时同步中',importing:'上传外部数据中',stopping:'正在结束',completed:'已完成',interrupted:'已中断',error:'错误',cancelled:'已取消'})[s]||s}
function shortMsg(s,n=140){const one=String(s||'').split(/\r?\n/).find(x=>x.trim())||'';return one.length>n?one.slice(0,n)+'…':one}
function collectionDeleteBtn(x){return`<button type="button" class="ghost danger-ghost" onclick="deleteCollection('${esc(x.id)}')" title="从列表删除（可选删本机目录）">删除</button>`}
function collectionActions(x){const reviewBtn=`<button onclick="goReview('local:${x.id}')">去标注</button>`;const delBtn=collectionDeleteBtn(x);if(x.status==='publish_error')return`<button class="primary green" onclick="retryPublish('${x.id}')">重新发布</button>`+delBtn;if(x.status==='published')return'<span class="session-state">请在共享会话中标注</span>'+delBtn;if(x.source_type!=='offline')return(x.worker_alive?`<button onclick="stopCollection('${x.id}')">取消上传</button>`:'')+delBtn;if(x.status==='collecting_offline')return reviewBtn+`<button onclick="stopCollection('${x.id}')">结束数采</button>`+delBtn;if(x.status==='stopping_offline')return reviewBtn+(x.worker_alive?'<span class="session-state">正在结束…</span>':`<button class="primary green" onclick="stopCollection('${x.id}')">强制结束并生成 labels</button>`)+delBtn;if(x.status==='interrupted')return reviewBtn+`<button onclick="stopCollection('${x.id}')">结束并生成 JSON</button>`+delBtn;if(x.status==='pending_upload')return reviewBtn+`<button class="primary green" onclick="openUploadModal('${x.id}')">上传</button>`+delBtn;if(x.status==='upload_error')return reviewBtn+`<button class="primary green" onclick="retryUpload('${x.id}')">重新上传</button>`+(x.task_id&&x.remote_dir?`<button onclick="openUploadModal('${x.id}')">修改目标</button>`:'')+delBtn;return reviewBtn+delBtn}
async function deleteCollection(id){
  const c=(state.collections||[]).find(x=>x.id===id);
  const title=c?.session_name||c?.local_dir||id;
  const path=c?.local_dir||'';
  if(!confirm(`从 Studio 删除该数采记录？\n\n${title}\n${path}\n\n默认只移除列表与技能挂载，不删磁盘文件。`))return;
  let deleteFiles=false;
  if(path&&!String(path).startsWith('/mnt/')){
    deleteFiles=confirm(`是否同时删除本机目录？\n\n${path}\n\n确定 = 删除磁盘数据（不可恢复）\n取消 = 仅从列表移除`);
  }
  try{
    const x=await del(`/api/collections/${encodeURIComponent(id)}`,{delete_files:deleteFiles});
    if(x.warning)toast(x.warning,true);
    else if(x.wiped)toast(`已删除记录并清空本机目录`);
    else toast(x.detached?'已删除记录并解除技能挂载':'已删除记录');
    if(review&&review.id===id){review=null;if($('reviewPanel'))$('reviewPanel').hidden=true}
    await loadState();
  }catch(e){toast(e.message,true)}
}
async function removeBoardDataset(path,datasetId){
  const skillId=selectedCollectSkillId;
  if(!skillId)return toast('请先选择技能',true);
  const label=path||datasetId||'';
  if(!confirm(`从当前技能卡移除该挂载？\n\n${label}\n\n不会删除磁盘 / 远端文件。`))return;
  try{
    await del(`/api/skills/${encodeURIComponent(skillId)}/datasets`,{path:path||'',id:datasetId||''});
    toast('已移除挂载');
    await loadState();
  }catch(e){toast(e.message,true)}
}
function renderSharedSessions(){const el=$('sharedSessionList'),sessions=sharedState.sessions||[];$('sharedSummary').textContent=sharedState.connected?`${(sharedState.tasks||[]).length} 任务 · ${sessions.length} 会话`:(sharedState.error?`离线`:'离线');if(!sessions.length){el.className='empty';el.textContent=sharedState.connected?'还没有已发布会话':'共享工作区不可达';return}el.className='shared-session-grid';el.innerHTML=sessions.map(x=>{const c=x.counts||{};const total=x.episode_count||((c.valid||0)+(c.invalid||0)+(c.unreviewed||0));return `<div class="shared-session-card"><div><span class="pill">${esc(x.task_name)}</span><h3>${esc(x.name)}</h3><small>${total} eps · ${esc(x.uploaded_at||'')}</small></div><div class="shared-counts"><b>${c.valid||0}</b> valid <b>${c.invalid||0}</b> invalid <b>${c.unreviewed||0}</b> 待标注</div><button class="ghost" onclick="goReview('shared:${x.id}')">查看与标注</button></div>`}).join('')}
async function fetchSharedSession(id){
  const x=await api(`/api/shared/sessions/${encodeURIComponent(id)}`);
  const session=x.session||x;
  const idx=(sharedState.sessions||[]).findIndex(s=>s.id===id);
  if(idx>=0)sharedState.sessions[idx]={...sharedState.sessions[idx],...session};
  return session;
}
function renderConversions(){const el=$('conversionList');if(!el)return;if(!state.conversions.length){el.className='empty';el.textContent='还没有转换任务';return}el.className='';el.innerHTML=state.conversions.map(x=>`<div class="session-row"><div><div class="session-title">${esc(x.id)}</div><small>${esc(x.created_at)}</small></div><div><b>${x.mappings?.length||0} 个字段映射</b><small>${esc(x.output_dir)}</small></div><div><span class="session-state ${x.status==='error'?'error':''}">${statusLabel(x.status)}</span><small>${esc(x.message)}</small></div><div class="session-actions">${x.status==='running'?`<button onclick="cancelConversion('${x.id}')">取消</button>`:`<button onclick="showConversion('${x.id}')">日志</button>`}</div></div>`).join('')}
function renderReviewSelect(opts={}){
  const el=$('reviewCollection');if(!el)return;
  const old=el.value;
  const locals=skillCollections(selectedCollectSkillId);
  const shared=(sharedState.sessions||[]).filter(x=>{
    if(!selectedCollectSkillId)return true;
    return x.task_name===selectedCollectSkillId||x.skill_id===selectedCollectSkillId;
  });
  el.innerHTML='<option value="">请选择待标注数据</option><optgroup label="本机会话（当前技能）">'+locals.map(x=>`<option value="local:${x.id}">[${x.source_type==='offline'?'离线':'历史'}] ${esc(x.session_name||x.task_name||x.local_dir?.replace(/\/+$/,'').split('/').pop())} (${statusLabel(x.status)})</option>`).join('')+'</optgroup><optgroup label="共享会话">'+shared.map(x=>`<option value="shared:${x.id}">[共享] ${esc(x.task_name)} / ${esc(x.name)} · ${x.counts?.unreviewed||0} 待标注</option>`).join('')+'</optgroup>';
  const ids=[...locals.map(x=>'local:'+x.id),...shared.map(x=>'shared:'+x.id)];
  if(ids.includes(old))el.value=old;
  else if(opts.autoBind){
    const active=pickActiveSkillCollection(selectedCollectSkillId);
    if(active){el.value='local:'+active.id;if(!review||review.id!==active.id)void loadReview()}
  }
  updateCollectFlowUI();
}
function renderHosts(){
  $('hostList').innerHTML=(state.hosts||[]).map(x=>{
    const ready=x.train_ready?'<span class="pill ready">train就绪</span>':'<span class="pill not-ready">train未就绪</span>';
    const paths=`zoo=${esc(x.model_zoo||'')} · data=${esc(x.train_data||'')}`;
    return `<div class="host-item"><b>${esc(x.name)}</b><code>${esc(x.target)}</code>${ready}
      <small class="quiet">${paths}</small>
      <button class="ghost" onclick="testHost('${x.id}')">测试连接</button>
      <button class="ghost" onclick="preflightTrainHost('${x.id}')">训练完整性</button>
    </div>`;
  }).join('');
}
async function preflightTrainHost(id){
  try{
    const x=await post(`/api/hosts/${encodeURIComponent(id)}/train-preflight`,{});
    toast(x.message||`${id} 校验通过`);
    await loadState();renderHosts();fillTrainHostSelect();
  }catch(e){toast(e.message,true);await loadState();renderHosts();fillTrainHostSelect()}
}
function fillTrainHostSelect(){
  const el=$('trainExecMode');if(!el)return;
  const old=el.value;
  const hosts=state.hosts||[];
  const opts=[];
  hosts.forEach(h=>{
    const ready=!!h.train_ready || h.id==='cluster_0';
    const label=ready?h.name:`${h.name}（未就绪）`;
    opts.push(`<option value="${esc(h.id)}" ${ready?'':'disabled'}>${esc(label)}</option>`);
  });
  opts.push('<option value="local">本机 local</option>');
  el.innerHTML=opts.join('');
  const prefer=state.train_config?.host_id||old||'cluster_0';
  if([...el.options].some(o=>o.value===prefer&&!o.disabled))el.value=prefer;
  else el.value='cluster_0';
  updateTrainHostHint();
}
function updateTrainHostHint(){
  const el=$('trainHostHint');if(!el)return;
  const v=$('trainExecMode')?.value||'cluster_0';
  if(v==='local'){el.textContent='本机执行（不经 SSH）';return}
  const h=(state.hosts||[]).find(x=>x.id===v);
  if(!h){el.textContent='';return}
  el.textContent=`${h.train_ready?'就绪':'未就绪'} · zoo ${h.model_zoo||''}`;
}

async function createTask(){toast('请到顶部「技能卡」管理页新建',true)}
function collectConfigFromForm(){
  const cfg=state.collect_config||{};
  return {
    ...cfg,
    gmr_root:($('cfgGmrRoot')?.value||cfg.gmr_root||'').trim(),
    groot_root:($('cfgGrootRoot')?.value||cfg.groot_root||'').trim(),
    deploy_root:($('cfgDeployRoot')?.value||cfg.deploy_root||'').trim(),
    camera_repo:($('cfgCameraRepo')?.value||cfg.camera_repo||'').trim(),
    camera_host:($('cfgCameraHost')?.value||cfg.camera_host||'').trim(),
    camera_port:Number($('cfgCameraPort')?.value||cfg.camera_port||5555),
    hand_zmq_host:($('cfgHandHost')?.value||cfg.hand_zmq_host||'').trim(),
    hand_zmq_port:Number($('cfgHandPort')?.value||cfg.hand_zmq_port||5558),
    g1_network_interface:($('cfgG1Iface')?.value||cfg.g1_network_interface||'').trim(),
    task_prompt:($('cfgTaskPrompt')?.value||cfg.task_prompt||'demo').trim(),
    xrt_pybind:($('cfgXrt')?.value||cfg.xrt_pybind||'').trim(),
  };
}
function fillCollectConfigForm(){
  const cfg=state.collect_config||{};
  const set=(id,v)=>{if($(id)!=null)$(id).value=v??''};
  set('cfgGmrRoot',cfg.gmr_root);set('cfgGrootRoot',cfg.groot_root);
  set('cfgDeployRoot',cfg.deploy_root);set('cfgCameraRepo',cfg.camera_repo);
  set('cfgCameraHost',cfg.camera_host);set('cfgCameraPort',cfg.camera_port);
  set('cfgHandHost',cfg.hand_zmq_host);set('cfgHandPort',cfg.hand_zmq_port);
  set('cfgG1Iface',cfg.g1_network_interface);set('cfgTaskPrompt',cfg.task_prompt);
  set('cfgXrt',cfg.xrt_pybind);
}
async function saveCollectConfig(silent){
  try{
    const x=await post('/api/collect/config',collectConfigFromForm());
    state.collect_config=x.config;fillCollectConfigForm();
    if(!silent)toast('采集配置已保存');
  }catch(e){if(!silent)toast(e.message,true);throw e}
}
function renderCollectStackStatus(stack){
  const el=$('collectStackStatus');if(!el)return;
  const lights=(stack&&stack.lights)||{};
  el.querySelectorAll('[data-k]').forEach(node=>{
    const key=node.dataset.k;
    let state=lights[key];
    if(!state){
      if(stack&&stack[key]===true)state='ok';
      else if(stack&&stack[key]===false)state='idle';
      else state=(key==='camera'?'skip':'idle');
    }
    node.classList.remove('ok','error','idle','skip','on','off');
    node.classList.add('lamp', state);
  });
}
function setCollectControlsEnabled(on){
  ['collectRecStart','collectRecSave','collectRecDiscard','collectDeployY']
    .forEach(id=>{if($(id))$(id).disabled=!on});
  if(!on){
    ['collectDeployStand','collectDeployEnter'].forEach(id=>{
      const b=$(id);if(!b)return;b.disabled=true;b.classList.remove('btn-next','primary','green');
    });
  }
}
function setCollectDeployPhase(phase){
  const order=['waiting_deploy','deploy_starting','init_done','standing','streaming'];
  const alias={starting:'waiting_deploy',waiting:'waiting_deploy',deploy:'deploy_starting',deploy_running:'deploy_starting',stopped:'waiting_deploy'};
  phase=alias[phase]||phase||'waiting_deploy';
  document.querySelectorAll('#collectPhaseBar span').forEach(s=>{
    s.classList.toggle('on',s.dataset.p===phase);
    const i=order.indexOf(phase),j=order.indexOf(s.dataset.p);
    s.classList.toggle('done',j>=0&&i>=0&&j<i);
  });
  const labels={
    waiting_deploy:'等待 y Deploy',
    deploy_starting:'Deploy 启动中…',
    init_done:'Init Done — 请点 ]',
    standing:'已站立 — 可 Enter',
    streaming:'流式运行中',
  };
  if($('collectDeployPhase')){
    $('collectDeployPhase').textContent=labels[phase]||phase;
    $('collectDeployPhase').className='status-pill'+(phase==='init_done'?' ready':'');
  }
  if($('collectDeployHint')){
    $('collectDeployHint').textContent=phase==='init_done'
      ?'已检测到 Init Done，请点 ] 站立'
      :'Deploy：网页按 y → 等 Init Done → ] → Enter';
  }
  const collecting=!!$('collectDeployY')&&!$('collectDeployY').disabled;
  ['collectDeployStand','collectDeployEnter'].forEach(id=>{
    const b=$(id);if(!b)return;b.disabled=true;b.classList.remove('btn-next','primary','green');
  });
  if(!collecting&&phase==='waiting_deploy')return;
  if($('collectDeployY'))$('collectDeployY').disabled=false;
  if(phase==='init_done'&&$('collectDeployStand')){
    $('collectDeployStand').disabled=false;
    $('collectDeployStand').classList.add('btn-next','primary','green');
  }
  if((phase==='standing'||phase==='streaming')&&$('collectDeployEnter')){
    $('collectDeployEnter').disabled=false;
    if(phase==='standing')$('collectDeployEnter').classList.add('btn-next','primary','green');
  }
  if(phase==='standing'||phase==='streaming'){
    if($('collectDeployStand'))$('collectDeployStand').disabled=false;
  }
}
async function refreshCollectLogs(){
  const active=pickActiveSkillCollection(selectedCollectSkillId);
  if(!active||!['collecting_offline','stopping_offline'].includes(active.status)){
    setCollectControlsEnabled(false);return;
  }
  activeCollectId=active.id;
  setCollectControlsEnabled(true);
  renderCollectStackStatus(active.stack);
  try{
    const x=await api(`/api/collections/${active.id}/collect-logs`);
    renderCollectStackStatus(x.stack);
    const pre=$('collectDeployLog');
    if(pre){
      const text=x.logs||'';
      pre.textContent=text.slice(-16000)||'（暂无日志）';
      pre.scrollTop=pre.scrollHeight;
    }
    setCollectDeployPhase(x.phase||active.deploy_phase||'waiting_deploy');
  }catch(_){}
}
let collectLogTimer=null,collectCamBound=false,collectCamTimer=null,activeCollectId='';
function bindCollectCameras(force){
  if(collectCamBound&&!force)return;
  collectCamBound=true;
  [['collectCam0','cam0','collectCamEmpty0'],['collectCam1','cam1','collectCamEmpty1']].forEach(([imgId,slot,emptyId])=>{
    const img=$(imgId),empty=$(emptyId);if(!img)return;
    img.onload=()=>{if(empty)empty.hidden=true;img.classList.add('is-live')};
    img.onerror=()=>{if(empty)empty.hidden=false;img.classList.remove('is-live')};
  });
  const tick=()=>{
    if(!collectCamBound||document.hidden||!$('page-collect')?.classList.contains('active'))return;
    const t=Date.now();
    [['collectCam0','cam0'],['collectCam1','cam1']].forEach(([imgId,slot])=>{
      const img=$(imgId);if(!img)return;
      // Snapshot polling — never open long-lived MJPEG (Edge/Chromium tab crashes).
      img.src=`/api/collect/camera/snapshot/${slot}?t=${t}`;
    });
  };
  if(collectCamTimer)clearInterval(collectCamTimer);
  collectCamTimer=setInterval(tick,700);
  tick();
  api('/api/collect/camera/meta').then(x=>{
    const keys=x.keys||[];
    if(keys[0]&&$('collectCamLabel0'))$('collectCamLabel0').textContent=keys[0];
    if(keys[1]&&$('collectCamLabel1'))$('collectCamLabel1').textContent=keys[1];
  }).catch(()=>{});
}
function stopCollectCameras(){
  collectCamBound=false;
  if(collectCamTimer){clearInterval(collectCamTimer);collectCamTimer=null}
  ['collectCam0','collectCam1'].forEach(id=>{const img=$(id);if(img){img.removeAttribute('src');img.classList.remove('is-live')}});
  ['collectCamEmpty0','collectCamEmpty1'].forEach(id=>{if($(id))$(id).hidden=false});
}
function startCollectStackPolling(){
  if(collectLogTimer)return;
  collectLogTimer=setInterval(()=>{
    if(!$('page-collect')?.classList.contains('active'))return;
    void refreshCollectLogs();
  },1500);
  void refreshCollectLogs();
}
async function collectRecordCmd(cmd){
  const active=pickActiveSkillCollection(selectedCollectSkillId);
  if(!active)return toast('没有进行中的采集',true);
  try{
    const x=await post(`/api/collections/${active.id}/record-cmd`,{cmd});
    toast(x.message||`已发送 ${cmd}`);
  }catch(e){toast(e.message,true)}
}
async function collectDeployCmd(cmd){
  const active=pickActiveSkillCollection(selectedCollectSkillId);
  if(!active)return toast('没有进行中的采集',true);
  try{
    await post(`/api/collections/${active.id}/deploy-cmd`,{cmd});
    toast(`Deploy 指令: ${cmd}`);
    void refreshCollectLogs();
  }catch(e){toast(e.message,true)}
}
function submitCollectTty(){
  const raw=($('collectTtyInput')?.value||'').trim().toLowerCase();
  if(!raw)return;
  let cmd=raw;
  if(raw==='y')cmd='y';
  else if(raw===']'||raw==='stand')cmd='stand';
  else if(raw==='enter'||raw==='')cmd='enter';
  else if(raw.length===1)cmd='key:'+raw;
  if($('collectTtyInput'))$('collectTtyInput').value='';
  return collectDeployCmd(cmd);
}
async function collectKillDeploy(){
  try{
    const x=await post('/api/collect/kill-deploy',{});
    toast(x.message||'已停 Deploy');
  }catch(e){toast(e.message,true)}
}
async function startCollection(){
  const skillId=selectedCollectSkillId||$('collectionTask')?.value||'';
  if(!skillId)return toast('请先在顶部选择或新建技能卡',true);
  try{
    await saveCollectConfig(true).catch(()=>{});
    const body={skill_id:skillId,collect_config:collectConfigFromForm(),auto_stack:true};
    const manual=($('localDir')?.value||'').trim();
    if(manual)body.local_dir=manual;
    const x=await post('/api/collections/start',body);
    toast(`已开始本地采集 · ${x.session?.local_dir||x.collection.local_dir}`);
    if($('localDir'))$('localDir').value='';
    if(x.launch_env&&$('collectLaunchEnv')){
      $('collectLaunchEnv').hidden=false;
      $('collectLaunchEnv').textContent=Object.entries(x.launch_env).map(([k,v])=>`${k}=${v}`).join('\n');
    }
    await loadState();
    bindCollectCameras(true);
    startCollectStackPolling();
    setCollectControlsEnabled(true);
    goReview('local:'+x.collection.id);
    startCollectReviewPolling();
  }catch(e){toast(e.message,true)}
}
async function stopCollection(id){try{await post(`/api/collections/${id}/stop`);toast('正在最终扫描并生成本地 labels.json');stopCollectCameras();setCollectControlsEnabled(false);setTimeout(loadState,1200)}catch(e){toast(e.message,true)}}
async function stopActiveCollection(){
  const active=pickActiveSkillCollection(selectedCollectSkillId);
  if(!active||!['collecting_offline','interrupted'].includes(active.status))return toast('没有进行中的采集',true);
  return stopCollection(active.id);
}
async function finishActiveCollection(){
  await stopActiveCollection();
  toast('结束后可在打标区确认，再点「上传到 cluster_0」');
}
async function uploadActiveReview(){
  const value=$('reviewCollection')?.value||'';
  let id=value.startsWith('local:')?value.slice(6):'';
  const item=id?state.collections.find(x=>x.id===id):null;
  if(item&&['pending_upload','upload_error'].includes(item.status))return openUploadModal(id);
  const pending=pendingUploadCollections(selectedCollectSkillId);
  if(pending.length===1)return openUploadModal(pending[0].id);
  if(pending.length>1){
    // Prefer the dropdown selection if it's pending; else newest pending.
    const newest=pending.sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')))[0];
    return openUploadModal(newest.id);
  }
  if(!id)id=pickActiveSkillCollection(selectedCollectSkillId)?.id||'';
  if(!id)return toast('没有可上传的本地会话（请先选技能，并确认有「待上传」会话）',true);
  return openUploadModal(id);
}
function goReview(id){
  const nav=document.querySelector('.nav[data-page="collect"]');
  if(nav&&!$('page-collect').classList.contains('active'))nav.click();
  setTimeout(()=>{
    if($('reviewCollection'))$('reviewCollection').value=id;
    loadReview();
    updateCollectFlowUI();
    $('collectReviewSection')?.scrollIntoView({behavior:'smooth',block:'start'});
  },0);
}
let collectReviewPollTimer=null;
function startCollectReviewPolling(){
  if(collectReviewPollTimer)return;
  collectReviewPollTimer=setInterval(async()=>{
    if(!$('page-collect')?.classList.contains('active'))return;
    const active=pickActiveSkillCollection(selectedCollectSkillId);
    if(!active||!['collecting_offline','stopping_offline','pending_upload'].includes(active.status))return;
    try{
      await loadState();
      const value=$('reviewCollection')?.value;
      if(value?.startsWith('local:')&&value.slice(6)===active.id){
        const x=await api(`/api/collections/${active.id}`);
        review=x.collection;
        renderEpisodes();
      }
      updateCollectFlowUI();
    }catch(_){}
  },3500);
}

function setUploadSkillMode(mode){
  uploadSkillMode=mode;
  $('uploadPickExistingTab')?.classList.toggle('active',mode==='existing');
  $('uploadPickNewTab')?.classList.toggle('active',mode==='new');
  if($('uploadSkillPickPane'))$('uploadSkillPickPane').hidden=mode!=='existing';
  if($('uploadSkillNewPane'))$('uploadSkillNewPane').hidden=mode!=='new';
}
function renderUploadSkillGrid(selectedId){
  const el=$('uploadSkillGrid');if(!el)return;
  const skills=state.skills||[];
  if(!skills.length){el.className='skill-pick-grid empty';el.textContent='还没有技能卡，请切换到「新建技能」';return}
  el.className='skill-pick-grid';
  el.innerHTML=skills.map(s=>{
    const sel=s.id===selectedId?' selected':'';
    const n=(s.sessions||[]).length;
    return `<button type="button" class="skill-pick-card${sel}" onclick="selectUploadSkill('${esc(s.id)}')"><span class="badge">${esc(s.badge||s.id)}</span><h3>${esc(s.title||s.id)}</h3><p>${esc(s.description||s.prompt||'无说明')}</p><div class="meta"><span class="pill">${n} sessions</span></div></button>`;
  }).join('');
}
function selectUploadSkill(id){
  selectedUploadSkillId=id;
  if($('uploadSkillId'))$('uploadSkillId').value=id;
  const skill=(state.skills||[]).find(s=>s.id===id);
  const task=(state.tasks||[]).find(t=>t.skill_id===id||t.name===id);
  if(task&&$('uploadTask'))$('uploadTask').value=task.id;
  renderUploadSkillGrid(id);
  fillUploadDefaultsFromSkill(skill,task);
}
function fillUploadDefaultsFromSkill(skill,task){
  if(!uploadingCollection)return;
  const hostId=uploadingCollection.host_id||skill?.host_id||task?.host_id||$('uploadHost')?.value;
  if(hostId&&$('uploadHost'))$('uploadHost').value=hostId;
  const session=uploadingCollection.local_dir.replace(/\/+$/,'').split('/').pop();
  const root=(skill?.remote_dir||task?.remote_dir||skill?.folder||'').replace(/\/+$/,'');
  if(uploadingCollection.remote_dir)$('uploadRemote').value=uploadingCollection.remote_dir;
  else if(root)$('uploadRemote').value=root+'/'+session;
}
async function openUploadModal(id){
  const item=state.collections.find(x=>x.id===id);if(!item)return toast('数采会话不存在',true);
  try{const x=await api('/api/skills');state.skills=x.skills||[];state.skills_config=x.config||state.skills_config}catch(_){}
  uploadingCollection=item;
  const retrying=item.status==='upload_error';
  $('uploadModalTitle').textContent=retrying?'再次上传 · 确认技能归属':'上传 · 指定技能归属';
  $('uploadConfirmBtn').textContent=retrying?'确认并再次上传':'确认归属并开始上传';
  $('uploadSession').textContent=`本地 Session：${item.local_dir} · 当前绑定：${item.skill_id||item.task_name||'未指定'}。请选择已有技能卡或新建技能。`;
  selectedUploadSkillId=item.skill_id||item.task_name||'';
  setUploadSkillMode((state.skills||[]).length?'existing':'new');
  renderUploadSkillGrid(selectedUploadSkillId);
  if(selectedUploadSkillId)selectUploadSkill(selectedUploadSkillId);
  if($('newSkillTitle')){$('newSkillTitle').value='';$('newSkillId').value='';$('newSkillBadge').value='';$('newSkillPrompt').value=''}
  if($('uploadHost'))$('uploadHost').disabled=false;
  fillUploadDefaults();
  $('uploadModal').showModal();
}
function fillUploadDefaults(){
  if(!uploadingCollection)return;
  if(selectedUploadSkillId){
    const skill=(state.skills||[]).find(s=>s.id===selectedUploadSkillId);
    const task=(state.tasks||[]).find(t=>t.skill_id===selectedUploadSkillId||t.name===selectedUploadSkillId);
    fillUploadDefaultsFromSkill(skill,task);
    return;
  }
  const task=state.tasks.find(x=>x.id===$('uploadTask').value);
  if(!task)return;
  $('uploadHost').value=uploadingCollection.host_id||task.host_id;
  if(uploadingCollection.remote_dir)$('uploadRemote').value=uploadingCollection.remote_dir;
  else{const session=uploadingCollection.local_dir.replace(/\/+$/,'').split('/').pop();$('uploadRemote').value=task.remote_dir.replace(/\/+$/,'')+'/'+session}
}
async function retryUpload(id){const item=state.collections.find(x=>x.id===id);if(!item)return toast('数采会话不存在',true);return openUploadModal(id)}
async function uploadCollection(){
  if(!uploadingCollection)return toast('请先选择待上传会话',true);
  try{
    let skillId=$('uploadSkillId')?.value||selectedUploadSkillId;
    if(uploadSkillMode==='new'){
      const title=$('newSkillTitle').value.trim();
      if(!title)return toast('请填写新技能名称',true);
      const created=await post('/api/skills',{
        title,
        id:$('newSkillId').value.trim(),
        badge:$('newSkillBadge').value.trim(),
        prompt:$('newSkillPrompt').value.trim(),
        description:$('newSkillPrompt').value.trim(),
        host_id:$('uploadHost').value,
        remote_base:(state.skills_config||{}).remote_base||$('taskRemote')?.value||''
      });
      skillId=created.skill.id;
      state.skills=state.skills||[];
      state.skills.unshift(created.skill);
      if(created.task){state.tasks=state.tasks||[];state.tasks.push(created.task);$('uploadTask').value=created.task.id}
      selectedUploadSkillId=skillId;if($('uploadSkillId'))$('uploadSkillId').value=skillId;
      fillUploadDefaultsFromSkill(created.skill,created.task);
    }
    if(!skillId)return toast('请选择或新建一个技能卡',true);
    const body={skill_id:skillId,host_id:$('uploadHost').value,remote_dir:$('uploadRemote').value};
    if($('uploadTask').value)body.task_id=$('uploadTask').value;
    await post(`/api/collections/${uploadingCollection.id}/upload`,body);
    $('uploadModal').close();
    toast('已写入技能目录 JSON，并开始上传（远端脚本后续可接）');
    uploadingCollection=null;
    await loadState();
    await refreshCollectSkills();
  }catch(e){toast(e.message,true)}
}
function openImportModal(){if(!state.tasks.length)return toast('请先在数据采集页注册任务',true);$('importModal').showModal();if(!$('importTask').value){$('importTask').value=state.tasks[0].id;fillImportDefaults()}}
function fillImportDefaults(){const t=state.tasks.find(x=>x.id===$('importTask').value);if(!t)return;$('importHost').value=t.host_id;suggestImportRemote()}
function suggestImportRemote(){const t=state.tasks.find(x=>x.id===$('importTask').value);if(!t)return;const parts=$('importLocal').value.replace(/\/+$/,'').split('/'),session=parts[parts.length-1]||new Date().toLocaleString('sv-SE').replace(' ','-').replaceAll(':','-');$('importRemote').value=t.remote_dir+'/'+session}
async function importDataset(){try{const x=await post('/api/collections/import',{task_id:$('importTask').value,host_id:$('importHost').value,local_dir:$('importLocal').value,remote_dir:$('importRemote').value});$('importModal').close();toast(`已发现 ${x.collection.episodes.length} 个 Episode；上传发布完成后才可标注`);await loadState()}catch(e){toast(e.message,true)}}

async function retryPublish(id){try{await post(`/api/shared/collections/${id}/publish`);toast('共享发布成功');await loadState()}catch(e){toast(e.message,true)}}

function clearRemotePreview(){remotePreview=null;const preview=$('remotePublishPreview');if(preview){preview.className='scan-preview empty';preview.textContent='尚未扫描'}if($('confirmRemotePublish'))$('confirmRemotePublish').disabled=true}
function suggestRemoteSessionPath(){clearRemotePreview();const task=(sharedState.tasks||[]).find(x=>x.name===$('remotePublishTask').value);if(task)$('remotePublishPath').value=task.remote_dir.replace(/\/+$/,'')+'/'}
function openRemotePublishModal(){if(!sharedState.connected)return toast('共享工作区当前离线，无法扫描远端 Session',true);if(!(sharedState.tasks||[]).length)return toast('请先注册一个共享任务',true);const select=$('remotePublishTask');select.innerHTML='<option value="">请选择共享任务</option>'+(sharedState.tasks||[]).map(x=>`<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');select.value=(sharedState.tasks||[])[0].name;suggestRemoteSessionPath();$('remotePublishModal').showModal()}
async function scanRemoteSession(){const task_name=$('remotePublishTask').value,remote_path=$('remotePublishPath').value;clearRemotePreview();try{const x=await post('/api/shared/remote-sessions/scan',{task_name,remote_path});remotePreview={task_name,remote_path,x};const counts=x.counts||{};$('remotePublishPreview').className='scan-preview';$('remotePublishPreview').innerHTML=`<b>${esc(x.name)}</b><span>${x.episodes.length} 个 Episode · ${x.file_count} 个文件 · ${fmtBytes(x.bytes_total)}</span><span><strong>${counts.valid||0}</strong> valid · <strong>${counts.invalid||0}</strong> invalid · <strong>${counts.unreviewed||0}</strong> 待标注</span>`;$('confirmRemotePublish').disabled=!x.episodes.length;if(!x.episodes.length)toast('没有发现可发布的 Episode',true)}catch(e){toast(e.message,true);$('remotePublishPreview').textContent='扫描失败：'+e.message}}
async function publishRemoteSession(){const task_name=$('remotePublishTask').value,remote_path=$('remotePublishPath').value;if(!remotePreview||remotePreview.task_name!==task_name||remotePreview.remote_path!==remote_path)return toast('路径已变化，请重新扫描确认',true);const button=$('confirmRemotePublish');button.disabled=true;try{const x=await post('/api/shared/remote-sessions/publish',{task_name,remote_path});$('remotePublishModal').close();clearRemotePreview();toast(`已发布 ${x.name}，共 ${x.episode_count} 个 Episode`);await loadState()}catch(e){button.disabled=false;toast(e.message,true)}}

function setMarkEnabled(enabled,message){$('markInvalid').disabled=!enabled;$('markValid').disabled=!enabled;$('lockNotice').textContent=message;$('lockNotice').className='lock-notice '+(enabled?'owned':'blocked')}
async function releaseCurrentLock(){if(lockTimer){clearInterval(lockTimer);lockTimer=null}if(!lockState)return;const old=lockState;lockState=null;try{await post('/api/shared/locks/release',{session_id:old.session_id,episode_id:old.episode_id,client_id:clientId})}catch{} }
async function loadReview(){const value=$('reviewCollection').value;await releaseCurrentLock();currentEp=null;if(!value){review=null;renderEpisodes();setMarkEnabled(false,'请选择待标注数据');updateCollectFlowUI();return}try{if(value.startsWith('shared:')){const id=value.slice(7);let session=(sharedState.sessions||[]).find(x=>x.id===id);if(!session)throw new Error('共享会话缓存中不存在，请刷新');if(!(session.episodes&&session.episodes.length))session=await fetchSharedSession(id);review={...session,shared:true,source_type:'shared'};setMarkEnabled(false,sharedState.connected?'选择 Episode 后申请协作锁':'共享工作区离线，只能查看缓存')}else{const id=value.replace(/^local:/,'');const x=await api(`/api/collections/${id}`);review=x.collection;setMarkEnabled(true,'本机会话：标注立即写入本地 labels.json')}renderEpisodes();selectFirst();updateCollectFlowUI()}catch(e){toast(e.message,true);updateCollectFlowUI()}}
async function refreshReview(){const value=$('reviewCollection').value;if(!value)return toast('请先选择数采会话',true);if(value.startsWith('shared:')){await loadState();$('reviewCollection').value=value;await loadReview();toast(sharedState.connected?'共享数据已刷新':'共享工作区离线',!sharedState.connected);return}try{const id=value.replace(/^local:/,'');await post(`/api/collections/${id}/refresh`);await loadState();$('reviewCollection').value=value;await loadReview();toast('本地数据已重新扫描')}catch(e){toast(e.message,true)}}
function renderEpisodes(){const eps=review?.episodes||[];$('validCount').textContent=eps.filter(x=>x.status==='valid').length;$('invalidCount').textContent=eps.filter(x=>x.status==='invalid').length;const shown=eps.filter(x=>filter==='all'||x.status===filter),el=$('episodeList');if(!shown.length){el.className='episode-list empty';el.textContent=review?'当前筛选下没有 Episode':'请选择数采会话';return}el.className='episode-list';el.innerHTML=shown.map((x,i)=>`<div class="episode-item ${currentEp?.id===x.id?'active':''}" onclick="selectEpisode('${x.id}')"><span class="num">${String(i+1).padStart(3,'0')}</span><div><b>${esc(x.name)} ${x.lock?'🔒':''}</b><small>${esc(x.lock?`${x.lock.hostname} / ${x.lock.ip}`:x.relative_path)}</small></div><i class="mark ${x.status}"></i></div>`).join('')}
function selectFirst(){const eps=(review?.episodes||[]).filter(x=>filter==='all'||x.status===filter);if(eps.length)selectEpisode(eps[0].id)}
async function selectEpisode(id){if(lockState&&(lockState.episode_id!==id||lockState.session_id!==review.id))await releaseCurrentLock();currentEp=review.episodes.find(x=>x.id===id);if(!currentEp)return;renderEpisodes();const i=review.episodes.indexOf(currentEp);$('epIndex').textContent=`EPISODE ${String(i+1).padStart(3,'0')} / ${review.episodes.length}`;$('epName').textContent=currentEp.name;$('epStatus').textContent=({unreviewed:'待标注',valid:'合格 / valid',invalid:'不合格 / invalid'})[currentEp.status];let views=currentEp.videos||[];if(!views.length&&currentEp.media)views=[{label:'默认视角',relative_path:currentEp.relative_path}];$('epPath').textContent=views.length?views.slice(0,2).map(x=>x.label).join(' + '):currentEp.relative_path;for(let n=0;n<2;n++){const p=$(`player${n}`),empty=$(`videoEmpty${n}`),label=$(`viewLabel${n}`),view=views[n];p.pause();p.removeAttribute('src');p.load();if(view){label.textContent=view.label||`视角 ${n+1}`;p.src=review.shared?`/api/shared/media/${review.id}/${currentEp.id}/${n}`:`/api/media/${review.id}/${currentEp.id}/${n}`;p.style.display='block';empty.style.display='none'}else{label.textContent=`视角 ${n+1}`;p.style.display='none';empty.style.display='grid';empty.textContent=n===0?'该 Episode 没有视频':'该 Episode 没有第二视角视频'}}const players=[$('player0'),$('player1')].filter(p=>p.style.display!=='none');Promise.all(players.map(p=>p.play().catch(()=>null)));if(!review.shared){setMarkEnabled(true,'本机会话：标注立即写入本地 labels.json');return}if(lockState&&lockState.session_id===review.id&&lockState.episode_id===currentEp.id){currentEp.version=lockState.version;setMarkEnabled(true,'已持有该 Episode 的协作锁；页面每 30 秒自动续租');return}setMarkEnabled(false,'正在申请共享 Episode 锁…');if(!sharedState.connected){setMarkEnabled(false,'共享工作区离线，视频和标注暂不可用');return}try{const result=await post('/api/shared/locks/acquire',{session_id:review.id,episode_id:currentEp.id,client_id:clientId});if(!result.acquired){const h=result.holder||{};setMarkEnabled(false,`由 ${h.hostname||'其他电脑'} / ${h.ip||'-'} 标注中，锁超时后自动开放`);return}currentEp.version=result.version;lockState={session_id:review.id,episode_id:currentEp.id,version:result.version};setMarkEnabled(true,'已取得 90 秒协作锁；页面每 30 秒自动续租');if(lockTimer)clearInterval(lockTimer);lockTimer=setInterval(async()=>{if(!lockState)return;try{const x=await post('/api/shared/locks/renew',{session_id:lockState.session_id,episode_id:lockState.episode_id,client_id:clientId});if(!x.renewed){lockState=null;if(lockTimer)clearInterval(lockTimer);lockTimer=null;setMarkEnabled(false,'协作锁已失效，请重新选择 Episode')}else setMarkEnabled(true,'协作锁已续租；页面每 30 秒自动续租')}catch{setMarkEnabled(false,'共享连接中断，暂不能提交标注')}},30000)}catch(e){setMarkEnabled(false,'申请协作锁失败：'+e.message)}}
async function markCurrent(status){if(!review||!currentEp)return toast('请先选择 Episode',true);try{const old=currentEp.id;if(review.shared){if(!lockState)return toast('未持有该 Episode 的协作锁',true);await post('/api/shared/mark',{session_id:review.id,episode_id:old,client_id:clientId,status,version:lockState.version});lockState=null;if(lockTimer){clearInterval(lockTimer);lockTimer=null}toast(status==='valid'?'共享标注已写入 valid':'共享标注已写入 invalid');await loadState();const detail=await fetchSharedSession(review.id);review={...detail,shared:true,source_type:'shared'};renderEpisodes()}else{await post(`/api/collections/${review.id}/mark`,{episode_id:old,status});const x=await api(`/api/collections/${review.id}`);review=x.collection;renderEpisodes();toast(status==='valid'?'已写入 valid，本地 labels.json 已更新':'已写入 invalid，本地 labels.json 已更新')}const pending=review.episodes.find(e=>e.status==='unreviewed')||review.episodes[Math.min(review.episodes.findIndex(e=>e.id===old)+1,review.episodes.length-1)];if(pending)selectEpisode(pending.id);else{currentEp=null;renderEpisodes()}updateCollectFlowUI()}catch(e){toast(e.message,true)}}
document.querySelectorAll('.filter-row .chip').forEach(b=>b.onclick=()=>{document.querySelectorAll('.filter-row .chip').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.filter;renderEpisodes();selectFirst()});
async function showManifest(){if(!review)return toast('请先选择数采会话',true);try{if(review.shared){const valid=review.episodes.filter(x=>x.status==='valid').map(x=>Number((x.name.match(/(\d+)$/)||[])[1])).filter(Number.isFinite),invalid=review.episodes.filter(x=>x.status==='invalid').map(x=>Number((x.name.match(/(\d+)$/)||[])[1])).filter(Number.isFinite);manifest={filename:'远端任务 JSON（当前 session）',path:review.remote_path,manifest:{[review.name]:{valid_count:valid.length,valid,invalid}}}}else manifest=await api(`/api/collections/${review.id}/manifest`);$('jsonTitle').textContent=manifest.filename;$('jsonPath').textContent=manifest.path;$('jsonPreview').textContent=JSON.stringify(manifest.manifest,null,2);$('jsonModal').showModal()}catch(e){toast(e.message,true)}}

async function probeFields(){try{const x=await post('/api/fields/probe',{host_id:$('convertHost').value,input_dir:$('convertInput').value});fieldOptions=x.fields||[];$('probeSummary').textContent=`发现 ${fieldOptions.length} 个字段 · 检查 ${x.files?.length||0} 个数据文件`;if(!fieldOptions.length)toast('未发现可识别字段，可手动填写',true);$('mappingRows').innerHTML='';addMapping();updateDims()}catch(e){toast(e.message,true)}}
function addMapping(source='',shape=[],start='',end=''){
  const host=$('mappingRows');
  if(!host)return;
  const row=document.createElement('div');row.className='mapping-row';const opts=fieldOptions.map(x=>`<option value="${esc(x.name)}" data-shape="${esc(JSON.stringify(x.shape||[]))}" ${x.name===source?'selected':''}>${esc(x.name)}</option>`).join('');row.innerHTML=`${fieldOptions.length?`<select class="source" onchange="syncShape(this)"><option value="">选择字段</option>${opts}</select>`:`<input class="source" placeholder="observation.state">`}<span class="shape">${esc(JSON.stringify(shape||[]))}</span><input class="start" type="number" min="0" max="511" value="${start}" oninput="updateDims()"><input class="end" type="number" min="1" max="512" value="${end}" oninput="updateDims()"><span class="dims">—</span><button class="remove" onclick="this.parentElement.remove();updateDims()">×</button>`;host.appendChild(row);updateDims()
}
function syncShape(sel){const opt=sel.selectedOptions[0];sel.parentElement.querySelector('.shape').textContent=opt?.dataset.shape||'[]'}
function mappings(){return[...document.querySelectorAll('.mapping-row')].map(r=>({source:r.querySelector('.source').value,start:Number(r.querySelector('.start').value),end:Number(r.querySelector('.end').value)}))}
function updateDims(){
  const occupied=new Set();
  document.querySelectorAll('.mapping-row').forEach(r=>{
    const s=Number(r.querySelector('.start')?.value),e=Number(r.querySelector('.end')?.value),n=Math.max(0,e-s);
    const dims=r.querySelector('.dims');if(dims)dims.textContent=n?`${n}D`:'—';
    if(s>=0&&e<=512)for(let i=s;i<e;i++)occupied.add(i);
  });
  if($('usedDims'))$('usedDims').textContent=occupied.size;
  if($('dimBar'))$('dimBar').style.width=(occupied.size/512*100)+'%';
}
async function startConversion(){try{const x=await post('/api/conversions',{host_id:$('convertHost').value,input_dir:$('convertInput').value,output_dir:$('convertOutput').value,script:$('convertScript').value,manifest:$('convertManifest').value,extra_args:$('extraArgs').value,mappings:mappings()});toast(`转换任务 ${x.conversion.id} 已启动`);await loadState()}catch(e){toast(e.message,true)}}
async function cancelConversion(id){try{await post(`/api/conversions/${id}/cancel`);await loadState();toast('已取消转换')}catch(e){toast(e.message,true)}}
async function showConversion(id){try{const x=await api(`/api/conversions/${id}`);manifest={logs:x.conversion.logs};$('jsonTitle').textContent=`${id} 日志`;$('jsonPath').textContent=x.conversion.command||'';$('jsonPreview').textContent=(x.conversion.logs||[]).join('\n')||'暂无日志';$('jsonModal').showModal()}catch(e){toast(e.message,true)}}

function openHostModal(){$('hostModal').showModal()}async function addHost(){try{await post('/api/hosts',{name:$('hostName').value,target:$('hostTarget').value});$('hostName').value=$('hostTarget').value='';await loadState();toast('Host 已添加')}catch(e){toast(e.message,true)}}async function testHost(id){try{const x=await post('/api/hosts/test',{host_id:id});toast(x.message)}catch(e){toast(e.message,true)}}
function openSharedModal(){if(state.shared){$('sharedHost').value=state.shared.host_id;$('sharedRoot').value=state.shared.root;$('sharedEnabled').checked=state.shared.enabled!==false}$('sharedModal').showModal()}
async function saveSharedConfig(){try{await post('/api/shared/config',{host_id:$('sharedHost').value,root:$('sharedRoot').value,enabled:$('sharedEnabled').checked});$('sharedModal').close();toast('共享数据库已初始化');await loadState()}catch(e){toast(e.message,true)}}

function linkDualPlayers(){const a=$('player0'),b=$('player1');let syncing=false;const mate=p=>p===a?b:a;for(const p of[a,b]){p.addEventListener('play',()=>{const q=mate(p);if(syncing||q.style.display==='none')return;syncing=true;if(Math.abs(q.currentTime-p.currentTime)>.15)q.currentTime=p.currentTime;q.play().catch(()=>{}).finally(()=>syncing=false)});p.addEventListener('pause',()=>{const q=mate(p);if(syncing||q.style.display==='none'||q.paused)return;syncing=true;q.pause();syncing=false});p.addEventListener('seeking',()=>{const q=mate(p);if(syncing||q.style.display==='none')return;syncing=true;q.currentTime=p.currentTime;syncing=false});p.addEventListener('error',()=>{if(!p.getAttribute('src'))return;const n=p===a?0:1,empty=$(`videoEmpty${n}`);p.style.display='none';empty.style.display='grid';empty.textContent='视频加载或解码失败；请确认远端可访问，且视频为浏览器支持的 H.264 MP4 / WebM'})}}
window.addEventListener('beforeunload',()=>{if(lockState)fetch('/api/shared/locks/release',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:lockState.session_id,episode_id:lockState.episode_id,client_id:clientId}),keepalive:true})});
linkDualPlayers();addMapping();restoreActivePage();loadState();
setInterval(()=>{if(!document.hidden)loadState()},8000);
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){
    stopCollectCameras();
    stopInferPreviewMedia();
  }else if($('page-collect')?.classList.contains('active')){
    const collecting=(state.collections||[]).some(c=>['collecting_offline','stopping_offline'].includes(c.status));
    if(collecting)bindCollectCameras(true);
  }else if($('page-infer')?.classList.contains('active')&&activeInferJob){
    void refreshInferPreview();
  }
});
// Only auto-heal Viser while a replay job is actually running (idle page stays light).
setInterval(()=>{
  if(document.hidden||!$('page-replay')?.classList.contains('active'))return;
  const running=(state.replay_jobs||[]).some(j=>j.kind==='replay'&&j.status==='running');
  if(running||viserWatchTimer)void checkViser(true);
},8000);
let replaySummaryPoll=0;setInterval(async()=>{if(document.hidden||!$('page-replay')?.classList.contains('active'))return;const job=state.replay_jobs?.find(j=>j.kind==='replay'&&j.status==='running');if(!job)return;replaySummaryPoll=(replaySummaryPoll+1)%3;if(replaySummaryPoll)return;try{const x=await post(`/api/replay/sessions/${job.id}/refresh-summary`);replayIssues=x.issues;state.replay_issues=state.replay_issues||{};state.replay_issues[job.dataset_path]=replayIssues;renderReplayIssues();renderReplayEpGrid();updateReplayMetrics()}catch(_){}},5000);

async function initReplayPage(){
  try{
    const x=await api('/api/replay/config');replayConfig=x;const cfg=x.config||{};
    if($('replayHost'))$('replayHost').value=cfg.host_id||'cluster_0';
    if(cfg.qa_root)$('replayQaRoot').value=cfg.qa_root;
    if(cfg.execution_mode&&$('replayExecMode'))$('replayExecMode').value=cfg.execution_mode;
    $('replayPreset').innerHTML='<option value="">选择或手动输入</option>'+(x.presets||[]).map(p=>`<option value="${esc(p.path)}">${esc(p.label)}</option>`).join('');
    viserUrl=`http://127.0.0.1:${cfg.viser_port||8081}/`;
    await refreshReplaySkills({quiet:true});
    const inputPath=$('replayDatasetPath').value.trim();
    const preferred=cfg.last_dataset_path||cfg.default_dataset_path||(x.presets||[])[0]?.path;
    if(!inputPath&&!selectedReplaySkillId&&preferred){
      $('replayDatasetPath').value=preferred;
      syncReplayPresetSelect(preferred);
    }else if(inputPath){
      syncReplayPresetSelect(inputPath);
    }
    const running=state.replay_jobs?.find(j=>j.kind==='replay'&&j.status==='running');
    if(running)activeReplayJob=running.id;
    renderReplayJobs();updateReplayMetrics();checkViser(true);
    const path=currentDatasetPath();
    if(path&&(!replayDataset||replayDataset.path!==path))await probeReplayDataset();
  }catch(e){toast(e.message,true)}
}
async function refreshReplaySkills(opts={}){
  try{
    const x=await api('/api/skills');
    state.skills=x.skills||[];
    state.skills_config=x.config||state.skills_config;
    renderReplaySkillGrid();
    if(!opts.quiet)toast(`已加载 ${state.skills.length} 个技能`);
  }catch(e){if(!opts.quiet)toast(e.message,true);renderReplaySkillGrid()}
}
function syncReplayAttachSkillSelect(){
  const sel=$('replayAttachSkill');if(!sel)return;
  const skills=state.skills||[];
  const cur=sel.value||selectedReplaySkillId||'';
  sel.innerHTML='<option value="">选择目标技能卡</option>'+skills.map(s=>`<option value="${esc(s.id)}">${esc(s.title||s.id)}</option>`).join('');
  if(cur&&[...sel.options].some(o=>o.value===cur))sel.value=cur;
}
function renderReplaySkillGrid(){
  const el=$('replaySkillGrid');if(!el)return;
  const skills=state.skills||[];
  if(!skills.length){el.className='skill-pick-grid empty';el.textContent='暂无技能卡 — 请先在 01 上传时创建，或在 01 新建后再回到本页挂载数据集';syncReplayAttachSkillSelect();return}
  el.className='skill-pick-grid';
  el.innerHTML=skills.map(s=>{
    const sel=s.id===selectedReplaySkillId?' selected':'';
    const n=(s.datasets||s.sessions||[]).length;
    return `<button type="button" class="skill-pick-card${sel}" onclick="selectReplaySkill('${esc(s.id)}')"><span class="badge">${esc(s.badge||s.id)}</span><h3>${esc(s.title||s.id)}</h3><p>${esc(s.description||s.prompt||'')}</p><div class="meta"><span class="pill">${n} 数据集/会话</span></div></button>`;
  }).join('');
  syncReplayAttachSkillSelect();
}
async function selectReplaySkill(id){
  selectedReplaySkillId=id;
  renderReplaySkillGrid();
  const attach=$('replayAttachSkill');if(attach)attach.value=id;
  try{
    const x=await api(`/api/skills/${encodeURIComponent(id)}/datasets`);
    replaySkillDatasets=x.datasets||[];
    const sel=$('replaySkillDataset');
    if(sel){
      sel.innerHTML=replaySkillDatasets.length
        ? replaySkillDatasets.map(d=>{
            const usePath=(d.remote_path&&String(d.remote_path).startsWith('/mnt'))?d.remote_path:d.path;
            return `<option value="${esc(usePath)}" data-local="${esc(d.path||'')}" data-remote="${esc(d.remote_path||'')}">${esc(d.label||d.id)}${d.ready?' ✓':' （未就绪）'}${d.total_episodes!=null?` · ${d.total_episodes} eps`:''}</option>`;
          }).join('')
        : '<option value="">该技能下暂无数据集</option>';
    }
    if(x.valid_json&&$('replayLabelsPath')&&!$('replayLabelsPath').value.trim())$('replayLabelsPath').value=x.valid_json;
    // 02 Isaac 正式回放优先 unified；否则 V2.1 采集会话
    const ready=x.preferred_replay
      ||replaySkillDatasets.find(d=>d.ready&&String(d.path||d.remote_path||'').toLowerCase().includes('unified'))
      ||replaySkillDatasets.find(d=>d.ready)
      ||replaySkillDatasets[0];
    if(ready?.path||ready?.remote_path){
      const usePath=(ready.remote_path&&String(ready.remote_path).startsWith('/mnt'))?ready.remote_path:ready.path;
      $('replayDatasetPath').value=usePath;
      if(sel)sel.value=ready.path;
      if(ready.labels_path&&$('replayLabelsPath'))$('replayLabelsPath').value=ready.labels_path;
      if($('replayRemote830Path')){
        if(ready.remote_path)$('replayRemote830Path').value=ready.remote_path;
        else if(usePath.includes('/datasets/830/'))$('replayRemote830Path').value=usePath;
        else{
          const base='/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830';
          $('replayRemote830Path').value=`${base}/${usePath.replace(/\/+$/,'').split('/').pop()}`;
        }
      }
      syncReplayPresetSelect(usePath);
      await probeReplayDataset();
      toast(`已选技能，扫描 ${ready.label||ready.id}`);
    }else{
      toast('技能下暂无可扫描数据集，可在下方「添加已有数据集」挂载 cluster 路径',true);
    }
  }catch(e){toast(e.message,true)}
}
function fillAttachPathFromCurrent(){
  const path=currentDatasetPath();
  if(!path)return toast('请先填写或扫描一个数据集路径',true);
  if($('replayAttachPath'))$('replayAttachPath').value=path;
  if($('replayAttachDetails'))$('replayAttachDetails').open=true;
  toast('已填入当前路径');
}
async function attachDatasetToReplaySkill(){
  const skillId=($('replayAttachSkill')?.value||selectedReplaySkillId||'').trim();
  const path=($('replayAttachPath')?.value||'').trim();
  const label=($('replayAttachLabel')?.value||'').trim();
  if(!skillId)return toast('请选择目标技能卡',true);
  if(!path)return toast('请填写 cluster_0 上的数据集绝对路径',true);
  try{
    const x=await post(`/api/skills/${encodeURIComponent(skillId)}/datasets`,{
      path,
      label,
      remote_path:path,
      host_id:resolveReplayHostId(),
      probe:true,
    });
    state.skills=(state.skills||[]).map(s=>s.id===skillId?(x.skill||s):s);
    if(!(state.skills||[]).some(s=>s.id===skillId)&&x.skill)state.skills=[x.skill,...(state.skills||[])];
    selectedReplaySkillId=skillId;
    $('replayDatasetPath').value=path;
    toast(`已挂到技能「${x.skill?.title||skillId}」`);
    await selectReplaySkill(skillId);
    await refreshReplaySkills({quiet:true});
  }catch(e){toast(e.message,true)}
}
function onReplaySkillDatasetChange(){
  const path=$('replaySkillDataset')?.value||'';
  if(!path)return;
  $('replayDatasetPath').value=path;
  const ds=(replaySkillDatasets||[]).find(d=>d.path===path||d.remote_path===path);
  if(ds?.labels_path&&$('replayLabelsPath'))$('replayLabelsPath').value=ds.labels_path;
  if($('replayRemote830Path')){
    const rem=ds?.remote_path||'';
    if(rem)$('replayRemote830Path').value=rem;
    else if(path.includes('/datasets/830/'))$('replayRemote830Path').value=path;
    else{
      const base='/mnt/data2/wpy/workspace/Phi0_Dataset/Phi0-MixCorpus/datasets/830';
      const name=path.replace(/\/+$/,'').split('/').pop();
      $('replayRemote830Path').value=`${base}/${name}`;
    }
  }
  syncReplayPresetSelect(path);
  void probeReplayDataset();
}
function currentRemote830Path(){return ($('replayRemote830Path')?.value||'').trim()}
function currentDatasetPath(){return ($('replayDatasetPath')?.value||'').trim()}
function syncReplayPresetSelect(path){
  const sel=$('replayPreset');if(!sel)return;
  const hit=[...sel.options].some(o=>o.value===path);
  sel.value=hit?path:'';
}
function markReplayDatasetStale(){
  const path=currentDatasetPath();
  if(replayDataset&&replayDataset.path!==path){
    replayDataset=null;replayIssues=null;replayExperiments=[];replaySelectedEps.clear();
    renderReplayEpGrid();renderReplayIssues();renderReplayExperiments();updateReplayMetrics();
    const box=$('replayDatasetSummary');
    if(box){box.className='scan-preview empty';box.textContent='路径已变更，请点击「扫描数据集」加载新数据'}
  }
}
async function ensureReplayDataset(){
  const path=currentDatasetPath();
  if(!path)throw new Error('请填写数据集路径');
  if(replayDataset?.path===path)return replayDataset;
  $('replayDatasetPath').value=path;
  await probeReplayDataset();
  if(!replayDataset||replayDataset.path!==path)throw new Error('数据集扫描失败，请检查路径');
  return replayDataset;
}
async function saveReplayExecMode(){
  try{
    const body={execution_mode:$('replayExecMode').value,host_id:resolveReplayHostId(),qa_root:$('replayQaRoot').value};
    if(currentDatasetPath())body.last_dataset_path=currentDatasetPath();
    await post('/api/replay/config',body);
  }catch(e){toast(e.message,true)}
}
document.getElementById('replayExecMode')?.addEventListener('change',saveReplayExecMode);
async function applyReplayPreset(){
  const v=$('replayPreset').value;
  if(!v)return;
  $('replayDatasetPath').value=v;
  await probeReplayDataset();
}
function parseReplayRange(text,total){
  text=String(text||'').trim();if(!text)return[];const eps=new Set();
  for(const part of text.split(/[\s,]+/)){if(!part)continue;if(part.includes('-')){const [a,b]=part.split('-');for(let i=+a;i<=+b;i++)eps.add(i)}else eps.add(+part)}
  return [...eps].filter(n=>Number.isFinite(n)&&n>=0&&n<total).sort((a,b)=>a-b)
}
/** 可筛选 Episode 池：有 01 标签时仅 valid，否则全量 */
function getReplayInventoryEps(){
  const total=replayDataset?.total_episodes|0;
  if(replayLabelFilter){
    const allow=new Set((replayLabelFilter.valid||[]).map(Number));
    return [...allow].filter(n=>Number.isFinite(n)&&n>=0&&n<total).sort((a,b)=>a-b);
  }
  return [...Array(total).keys()];
}
function formatInvalidTip(invalid){
  if(!invalid?.length)return '';
  return `[${invalid.join(',')}]为invalid`;
}
function syncReplayLabelTipToLog(){
  const el=$('replayLog');if(!el)return;
  const tip=replayLabelTip||'';
  const body=(el.dataset.jobBody||el.textContent||'').replace(/^\[.*?\]为invalid\n?/,'').replace(/^未加载 01 标签[^\n]*\n?/,'').replace(/^已按 valid 筛选[^\n]*\n?/,'');
  if(tip){
    el.dataset.jobBody=body;
    el.textContent=body&&body!=='等待任务…'?`${tip}\n${body}`:tip;
  }else if(el.dataset.jobBody){
    el.textContent=el.dataset.jobBody||'等待任务…';
  }
}
function applyReplayLabelFilterState(x,{selectAllValid=true}={}){
  const valid=(x.valid||[]).map(Number).filter(Number.isFinite);
  const invalid=(x.invalid||[]).map(Number).filter(Number.isFinite);
  replayLabelFilter={valid,invalid,path:x.labels_path||($('replayLabelsPath')?.value||'').trim()};
  replayLabelTip=formatInvalidTip(invalid)||(valid.length?`已按 valid 筛选 ${valid.length} 条（invalid 不显示）`:'');
  syncReplayLabelTipToLog();
  if(selectAllValid)setReplaySelection(valid);
  else renderReplayEpGrid();
  updateReplayMetrics();
}
async function applyReplayLabelsFilter(silent=false){
  if(!replayDataset){if(!silent)toast('请先扫描数据集',true);return false}
  const path=($('replayLabelsPath')?.value||'').trim();
  if(!path){
    replayLabelFilter=null;
    replayLabelTip='未加载 01 标签，显示全部 Episode；填写 labels 后仅筛选 valid';
    syncReplayLabelTipToLog();
    renderReplayEpGrid();
    if(!silent)toast('未填写标签路径，暂显示全部',true);
    return false;
  }
  try{
    const total=replayDataset.total_episodes|0;
    const x=await post('/api/replay/parse-labels',{
      labels_path:path,
      host_id:resolveReplayHostId(),
      total_episodes:total,
      dataset_path:replayDataset.path||currentDatasetPath(),
      session_name:String(replayDataset.path||currentDatasetPath()||'').replace(/\/+$/,'').split('/').pop()
    });
    const valid=x.valid||[];
    if(!valid.length){
      if(!silent)toast('JSON 中 valid 为空',true);
      replayLabelFilter={valid:[],invalid:x.invalid||[],path};
      replayLabelTip=formatInvalidTip(x.invalid)||'valid 为空';
      syncReplayLabelTipToLog();
      setReplaySelection([]);
      return false;
    }
    applyReplayLabelFilterState(x,{selectAllValid:true});
    if(!silent)toast(`仅筛选 valid：${valid.length} 条`+(x.invalid?.length?` · invalid ${x.invalid.length} 条已隐藏`:''));
    return true;
  }catch(e){
    replayLabelFilter=null;
    replayLabelTip=`标签读取失败: ${e.message}`;
    syncReplayLabelTipToLog();
    if(!silent)toast(e.message,true);
    return false;
  }
}
function setReplayPickMode(mode,btn){
  replayPickMode=mode;
  document.querySelectorAll('[data-replay-mode]').forEach(x=>x.classList.toggle('active',x.dataset.replayMode===mode));
  if(btn?.classList)btn.classList.add('active');
  if($('replayPickHint'))$('replayPickHint').textContent=mode==='range'?'例：60-70（仅 valid 池内）':'点击下方 Episode 编号多选（仅 valid；invalid 不显示）';
  renderReplayEpGrid();
}
function setReplaySelection(eps){
  const pool=new Set(getReplayInventoryEps());
  const cleaned=[...new Set(eps.map(Number).filter(n=>pool.has(n)))].sort((a,b)=>a-b);
  replaySelectedEps=new Set(cleaned);
  $('replayEpisodes').value=episodesToArg(cleaned);
  if(cleaned.length)replayFocusEp=cleaned[0];
  renderReplayEpGrid();updateReplayMetrics();
}
async function applyReplayQuick(kind){
  if(!replayDataset)return toast('请先扫描数据集',true);
  const pool=getReplayInventoryEps();
  const issues=replayIssues?.issues||{};
  const failSet=new Set(Object.entries(issues).filter(([,v])=>v.status!=='resolved').map(([ep])=>+ep));
  if(kind==='all'){setReplaySelection(pool);toast(`已选全部 valid ${pool.length}`);return}
  if(kind==='fails'){
    const eps=[...failSet].filter(n=>pool.includes(n)).sort((a,b)=>a-b);
    if(!eps.length)return toast('公告栏暂无失败条目（valid 池内）',true);
    setReplayPickMode('pick');
    setReplaySelection(eps);toast(`已选 ${eps.length} 条失败`);return
  }
  if(kind==='unqa'){
    const eps=pool.filter(i=>!failSet.has(i));
    setReplaySelection(eps);toast(`已选未 QA / 非失败 ${eps.length} 条`);return
  }
  if(kind==='random'){
    const n=Math.max(1,Math.min(pool.length,+(prompt('随机抽取条数（valid 池）', '20')||0)));
    if(!n)return;
    const shuffled=[...pool];
    for(let i=shuffled.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]]}
    setReplaySelection(shuffled.slice(0,n));toast(`随机选中 ${n} 条 valid`);return
  }
  if(kind==='valid_json')return applyReplayLabelsFilter();
}
function setAnnotateFocus(ep){
  if(!replayDataset)return;
  const pool=new Set(getReplayInventoryEps());
  if(!Number.isFinite(ep)||!pool.has(ep))return;
  replayFocusEp=ep;
  if($('replayFocusEp'))$('replayFocusEp').textContent=String(ep);
  renderReplayEpGrid();
}
function shiftAnnotateFocus(delta){
  if(!replayDataset)return;
  const selected=getSelectedReplayEps();
  const pool=selected.length?selected:getReplayInventoryEps();
  if(!pool.length)return;
  let idx=pool.indexOf(replayFocusEp);
  if(idx<0)idx=0;else idx=(idx+delta+pool.length)%pool.length;
  setAnnotateFocus(pool[idx]);
}
async function markFocusFail(note=''){
  if(replayFocusEp==null)return toast('请先选择标注焦点 Episode',true);
  const path=replayDataset?.path||currentDatasetPath();
  if(!path)return toast('请先扫描数据集',true);
  const ep=replayFocusEp;
  try{
    const x=await post('/api/replay/issues',{dataset_path:path,issue_updates:{[ep]:{
      status:'fail',source:'manual',
      summary_zh:`Episode ${ep}：人工标记异常`,
      note:note||undefined
    }}});
    replayIssues=x.issues;state.replay_issues[path]=replayIssues;
    renderReplayIssues();renderReplayEpGrid();toast(`已标记 Fail · ep ${ep}`)
  }catch(e){toast(e.message,true)}
}
async function clearFocusFail(){
  if(replayFocusEp==null)return toast('请先选择标注焦点',true);
  await removeIssue(replayFocusEp);
}
async function markFocusSuspect(){
  if(replayFocusEp==null)return toast('请先选择标注焦点',true);
  const note=prompt(`Episode ${replayFocusEp} 可疑备注`,'可疑，待复核')||'';
  if(!note)return;
  await markFocusFail(note);
}
async function exportValidAllowlist(){
  try{
    const ds=await ensureReplayDataset();
    const labels=($('replayLabelsPath')?.value||'').trim();
    const body={
      dataset_path:ds.path,
      host_id:resolveReplayHostId(),
      source:labels?'labels':'issues',
      sync_cluster:true,
      write_allowlists:false,
      remote_path:currentRemote830Path()||undefined,
    };
    if(labels)body.labels_path=labels;
    const x=await post('/api/replay/export-valid-allowlist',body);
    toast(`已写回 labels：valid ${x.valid_count} · invalid ${x.invalid_count} → session + 父级 skill_N.json`);
    const msg=`session：${x.labels_path||x.allowlist_path}\n父级：${x.parent_path||''}\n\nvalid ${x.valid_count} · invalid ${x.invalid_count}\n\n是否跳转到 03 训练？`;
    if(confirm(msg))goTrainWithReplayDataset(ds.path);
  }catch(e){toast(e.message,true)}
}
async function syncValidToCluster830(){
  try{
    const ds=await ensureReplayDataset();
    const x=await post('/api/replay/sync-valid-to-cluster',{
      dataset_path:ds.path,
      host_id:resolveReplayHostId(),
      remote_path:currentRemote830Path()||undefined,
    });
    if(x.remote_path&&$('replayRemote830Path'))$('replayRemote830Path').value=x.remote_path;
    toast(`已同步 ${x.synced_count} 个 meta → ${x.remote_path}`);
  }catch(e){toast(e.message,true)}
}
function goTrainWithReplayDataset(path){
  path=path||replayDataset?.path||currentDatasetPath();
  document.querySelector('.nav[data-page="train"]').click();
  setTimeout(async()=>{
    await initTrainPage();
    const walk=(trainCatalog?.tasks||[])[0];
    if(walk)openTrainWorkspace(walk);
    if(path){$('trainDatasetPath').value=path;void onTrainDatasetChange()}
  },50);
}
document.addEventListener('keydown',e=>{
  if($('page-replay')?.classList.contains('active')){
    const tag=(e.target&&e.target.tagName||'').toLowerCase();
    if(tag==='input'||tag==='textarea'||tag==='select'||e.target?.isContentEditable)return;
    const k=e.key.toLowerCase();
    if(k==='f'){e.preventDefault();void markFocusFail()}
    else if(k==='u'){e.preventDefault();void clearFocusFail()}
    else if(k==='s'){e.preventDefault();void markFocusSuspect()}
    else if(e.key===']'){e.preventDefault();shiftAnnotateFocus(1)}
    else if(e.key==='['){e.preventDefault();shiftAnnotateFocus(-1)}
    return;
  }
  if($('page-infer')?.classList.contains('active')){
    const tag=(e.target&&e.target.tagName||'').toLowerCase();
    if(tag==='input'||tag==='textarea'||tag==='select'||e.target?.isContentEditable)return;
    // Number keys 1–9: same as publish_skill_prompt_keyboard KEY_TO_SKILL slots
    if(/^[1-9]$/.test(e.key)){
      const skills=inferSkillList();
      const idx=+e.key-1;
      if(idx>=skills.length)return;
      e.preventDefault();
      const running=(state.infer_jobs||[]).find(j=>j.status==='running');
      if(running||!$('inferSwitchPanel')?.hidden){
        void applyInferSkillSwitch(skills[idx].id);
      }else{
        selectInferSkill(skills[idx].id);
      }
    }
  }
});
function getSelectedReplayEps(){
  if(!replayDataset)return[];
  const pool=new Set(getReplayInventoryEps());
  if(replayPickMode==='pick')return[...replaySelectedEps].filter(n=>pool.has(n)).sort((a,b)=>a-b);
  const fromRange=parseReplayRange($('replayEpisodes').value,replayDataset.total_episodes).filter(n=>pool.has(n));
  if(fromRange.length)return fromRange;
  // 默认：valid 池前 20 条（或全部不足 20）
  return getReplayInventoryEps().slice(0,20);
}
function episodesToArg(eps){if(!eps.length)return'0';const sorted=[...eps].sort((a,b)=>a-b);let ranges=[],s=sorted[0],p=sorted[0];for(let i=1;i<sorted.length;i++){if(sorted[i]===p+1)p=sorted[i];else{ranges.push(s===p?`${s}`:`${s}-${p}`);s=p=sorted[i]}}ranges.push(s===p?`${s}`:`${s}-${p}`);return ranges.length===1&&ranges[0].includes('-')?ranges[0]:sorted.join(' ')}
function datasetFormatLabel(x){
  const fmt=String(x?.format||'');
  const ver=String(x?.codebase_version||'');
  if(fmt==='v3'||ver.startsWith('v3'))return ver&&ver.startsWith('v3')?`LeRobot ${ver}`:'LeRobot V3';
  if(fmt==='v2.1'||ver.startsWith('v2'))return ver&&ver.startsWith('v2')?`LeRobot ${ver}`:'LeRobot V2.1';
  return ver?`LeRobot ${ver}`:'LeRobot';
}
async function probeReplayDataset(){
  const path=currentDatasetPath();
  if(!path)return toast('请填写数据集路径',true);
  try{
    const x=await post('/api/replay/datasets/probe',{host_id:resolveReplayHostId(),dataset_path:path,qa_root:$('replayQaRoot').value});
    replayDataset=x;replayIssues=x.issues||state.replay_issues?.[x.path]||{issues:{},marked_delete:[],prune_status:'pending'};
    replayExperiments=x.experiments||[];replayExecution=x.execution_mode||'local';
    state.replay_issues=state.replay_issues||{};state.replay_issues[x.path]=replayIssues;
    replayLabelFilter=null;replayLabelTip='';
    syncReplayPresetSelect(x.path);
    const fmtLabel=datasetFormatLabel(x);
    const sonic=x.sonic_replay||{};
    const sonicOk=!!sonic.ready;
    const fmt=String(x.format||'');
    const isV21=fmt==='v2.1'||String(sonic.layout||'').startsWith('raw_v21')||String(x.codebase_version||'').startsWith('v2');
    const sonicPill=sonicOk?(isV21?'V2.1·Sonic回放可用':'Sonic回放可用'):(isV21?'V2.1·仅筛选':'筛选可用·无Sonic');
    const sonicHint=sonic.message?`<span class="${sonicOk?'':'warn-text'}">${esc(sonic.message)}</span>`:'';
    $('replayDatasetSummary').className='scan-preview';
    $('replayDatasetSummary').innerHTML=`<b>${esc(x.name)}</b><span><span class="pill">${esc(fmtLabel)}</span> <span class="pill ${sonicOk?'ready':(isV21?'':'warn')}">${esc(sonicPill)}</span> · ${x.total_episodes} 条 Episode · ${x.total_frames} 帧 · ${x.fps} FPS</span><span>${esc(x.robot_type||'robot')} · ${esc(x.task_prompt||'无 task prompt')}</span>${sonicHint}<span><strong>${replayExecution==='local'?'本机':'远端'}</strong> · 已加载 <code>${esc(x.path)}</code></span>`;
    replaySelectedEps.clear();
    const labelsOk=await applyReplayLabelsFilter(true);
    if(!labelsOk){renderReplayEpGrid();updateReplayMetrics()}
    renderReplayIssues();renderReplayExperiments();
    if(replayExperiments.length)await importLatestExperiment(true);
    const inv=getReplayInventoryEps().length;
    toast(labelsOk
      ?`已切换到 ${x.name} · 仅 valid ${inv} 条可筛`
      :`已切换到 ${x.name}（${fmtLabel} · ${x.total_episodes} 条 Episode）`)
  }catch(e){toast(e.message,true);$('replayDatasetSummary').className='scan-preview empty';$('replayDatasetSummary').textContent='扫描失败：'+e.message}
}
function renderReplayExperiments(){
  const el=$('replayExperimentList');if(!el)return;
  if(!replayExperiments.length){el.className='experiment-list empty';el.textContent='暂无历史实验';return}
  el.className='experiment-list';
  el.innerHTML=replayExperiments.map((exp,i)=>`<div class="experiment-item"><div><b>${esc(exp.name)}</b><small>${exp.n_ok} 合格 · ${exp.n_fail} 异常 · ${esc(exp.mtime)}</small></div><button class="ghost" onclick="importExperimentByIndex(${i})">导入</button></div>`).join('');
}
async function importExperimentByIndex(i){if(replayExperiments[i])await importExperiment(replayExperiments[i].path)}
async function importExperiment(path){
  try{
    const ds=await ensureReplayDataset();
    const x=await post('/api/replay/experiments/import',{dataset_path:ds.path,experiment_path:path});
    replayIssues=x.issues;renderReplayIssues();renderReplayEpGrid();toast(`已导入：${x.summary?.n_fail||0} 条异常`)
  }catch(e){toast(e.message,true)}
}
async function importLatestExperiment(silent=false){
  if(!replayExperiments.length)return;
  await importExperiment(replayExperiments[0].path);
  if(!silent)toast('已导入最新实验结果')
}
function renderReplayEpGrid(){
  const el=$('replayEpGrid'),allEps=replayDataset?.episodes||[],issues=replayIssues?.issues||{};
  const inventory=new Set(getReplayInventoryEps());
  const eps=allEps.filter(e=>inventory.has(e.index));
  const invCount=inventory.size;
  $('replayInventoryCount').textContent=invCount;
  if(!allEps.length){el.className='replay-ep-grid empty';el.textContent='先扫描数据集';return}
  if(replayLabelFilter&&!invCount){el.className='replay-ep-grid empty';el.textContent='valid 为空 · invalid 已从网格隐藏';$('replayEpSelected').textContent=0;return}
  el.className='replay-ep-grid';const selected=new Set(getSelectedReplayEps()),marked=new Set(replayIssues?.marked_delete||[]);
  if(replayFocusEp!=null&&!inventory.has(replayFocusEp))replayFocusEp=null;
  if(replayFocusEp==null&&selected.size)replayFocusEp=[...selected][0];
  if($('replayFocusEp'))$('replayFocusEp').textContent=replayFocusEp==null?'—':String(replayFocusEp);
  el.innerHTML=eps.map(e=>{
    const st=issues[String(e.index)]?.status||(issues[String(e.index)]?'fail':'');
    const cls=['ep-chip',selected.has(e.index)?'selected':'',st==='fail'||issues[String(e.index)]?.source==='auto'?'fail':'',marked.has(e.index)?'marked-delete':'',replayFocusEp===e.index?'focus':''].filter(Boolean).join(' ');
    return `<button type="button" class="${cls}" title="${e.length} 帧" onclick="toggleReplayEp(${e.index})">${e.index}</button>`
  }).join('');
  $('replayEpSelected').textContent=getSelectedReplayEps().length;
}
function toggleReplayEp(n){
  setAnnotateFocus(n);
  if(replayPickMode!=='pick'){$('replayEpisodes').value=String(n);renderReplayEpGrid();return}
  if(replaySelectedEps.has(n))replaySelectedEps.delete(n);else replaySelectedEps.add(n);
  $('replayEpisodes').value=episodesToArg([...replaySelectedEps]);renderReplayEpGrid();
}
function updateReplayMetrics(){
  $('replayEpTotal').textContent=replayDataset?(replayLabelFilter?getReplayInventoryEps().length:replayDataset.total_episodes):'—';
  $('replayEpSelected').textContent=replayDataset?getSelectedReplayEps().length:0;
  const issues=replayIssues?.issues||{},failCount=Object.values(issues).filter(x=>x.status!=='resolved').length;
  $('replayIssueCount').textContent=failCount;
  const prune=replayIssues?.prune_status||'pending';
  const pruneEl=$('replayPruneMetric'),statusEl=$('replayPruneStatus'),board=$('issueBoard'),boardStatus=$('issueBoardStatus');
  if(prune==='completed'){pruneEl.className='metric prune-done';statusEl.textContent='已完成';board?.classList.add('done');boardStatus.textContent='已完成';$('pruneConfirmBtn').disabled=true}
  else{pruneEl.className='metric';statusEl.textContent=prune==='running'?'处理中…':prune==='error'?'失败':'待处理';board?.classList.remove('done');boardStatus.textContent=prune==='running'?'清理中…':'待处理';$('pruneConfirmBtn').disabled=!(replayIssues?.marked_delete?.length)}
}
function renderReplayIssues(){
  const el=$('issueList'),issues=replayIssues?.issues||{},marked=new Set(replayIssues?.marked_delete||[]);
  const rows=Object.entries(issues).filter(([,v])=>v.status!=='resolved').sort((a,b)=>+a[0]-+b[0]);
  if(!rows.length){el.className='issue-list empty';el.textContent='暂无异常记录';updateReplayMetrics();return}
  el.className='issue-list';
  el.innerHTML=rows.map(([ep,v])=>{
    const checked=marked.has(+ep)?'checked':'';
    const cls=['issue-item',marked.has(+ep)?'marked':'',v.status==='resolved'?'resolved':''].filter(Boolean).join(' ');
    return `<div class="${cls}"><input type="checkbox" data-issue-ep="${ep}" ${checked} onchange="toggleIssueDelete(${ep},this.checked)"><div><b>${esc(v.summary_zh||`Episode ${ep}：异常`)}</b><small>${esc((v.reasons||[]).join(' · ')||'人工标记')}</small><textarea placeholder="补充问题描述（中文）" oninput="updateIssueNote(${ep},this.value)">${esc(v.note||'')}</textarea></div><div class="issue-item-actions"><span class="pill">${v.source==='auto'?'自动':'人工'}</span><button type="button" class="text-btn issue-remove-btn" title="从公告栏移除此条" onclick="removeIssue(${ep})">移除</button></div></div>`
  }).join('');
  updateReplayMetrics();
}
function toggleIssueDelete(ep,on){
  if(!replayIssues)return;replayIssues.marked_delete=replayIssues.marked_delete||[];
  if(on){if(!replayIssues.marked_delete.includes(ep))replayIssues.marked_delete.push(ep)}
  else replayIssues.marked_delete=replayIssues.marked_delete.filter(x=>x!==ep);
  replayIssues.marked_delete.sort((a,b)=>a-b);
  void saveReplayIssues();renderReplayEpGrid();renderReplayIssues();
}
async function updateIssueNote(ep,note){
  if(!replayIssues)return;replayIssues.issues[String(ep)]=replayIssues.issues[String(ep)]||{episode:+ep};
  replayIssues.issues[String(ep)].note=note;
  clearTimeout(updateIssueNote._t);updateIssueNote._t=setTimeout(saveReplayIssues,600);
}
async function saveReplayIssues(){
  const path=replayDataset?.path;
  if(!path||path!==currentDatasetPath())return;
  try{const x=await post('/api/replay/issues',{dataset_path:path,marked_delete:replayIssues?.marked_delete||[],issue_updates:replayIssues?.issues||{}});replayIssues=x.issues;state.replay_issues[path]=replayIssues}catch(e){toast(e.message,true)}
}
function unresolvedIssueEps(){
  const issues=replayIssues?.issues||{};
  return Object.entries(issues).filter(([,v])=>v.status!=='resolved').map(([ep])=>+ep).sort((a,b)=>a-b)
}
function selectAllIssues(on){
  if(!replayIssues)return;
  replayIssues.marked_delete=on?unresolvedIssueEps():[];
  void saveReplayIssues();renderReplayEpGrid();renderReplayIssues();
  toast(on?`已全选 ${replayIssues.marked_delete.length} 条`:'已取消全选')
}
async function removeIssue(ep){
  const path=replayDataset?.path||currentDatasetPath();
  if(!path)return toast('请先扫描数据集',true);
  try{
    const x=await post('/api/replay/issues',{dataset_path:path,remove_episodes:[+ep]});
    replayIssues=x.issues;state.replay_issues[path]=replayIssues;renderReplayIssues();renderReplayEpGrid();updateReplayMetrics();toast(`已移除 Episode ${ep}`)
  }catch(e){toast(e.message,true)}
}
async function clearReplayIssues(){
  const path=replayDataset?.path||currentDatasetPath();
  if(!path)return toast('请先扫描数据集',true);
  if(!confirm('确认清除当前数据集公告栏中的所有不合格记录？\n\n此操作不会删除数据集中的 Episode，仅清空界面记录。'))return;
  try{
    const x=await post('/api/replay/issues/clear',{dataset_path:path});
    replayIssues=x.issues;state.replay_issues=state.replay_issues||{};state.replay_issues[path]=replayIssues;
    renderReplayIssues();renderReplayEpGrid();updateReplayMetrics();toast('公告栏已清空')
  }catch(e){toast(e.message,true)}
}
function markSelectedIssuesDelete(){
  if(!replayIssues)return;const boxes=[...document.querySelectorAll('[data-issue-ep]')];
  replayIssues.marked_delete=boxes.filter(b=>b.checked).map(b=>+b.dataset.issueEp);void saveReplayIssues();renderReplayEpGrid();renderReplayIssues();toast(`已标记 ${replayIssues.marked_delete.length} 条待删除`)
}
async function confirmPrune(){
  const eps=replayIssues?.marked_delete||[];
  if(!eps.length)return toast('请先勾选要删除的 Episode',true);
  const fmt=datasetFormatLabel(replayDataset);
  if(!confirm(`确认从 ${fmt} 数据集中删除 ${eps.length} 条 Episode 并重新排列索引？\n\n支持 V2.1 / V3。\n删除：${eps.join(', ')}\n\n若勾选了已不存在的索引，会自动跳过。`))return;
  try{
    const ds=await ensureReplayDataset();
    replayIssues.prune_status='running';updateReplayMetrics();
    const x=await post('/api/replay/prune',{host_id:resolveReplayHostId(),dataset_path:ds.path,delete_episodes:eps});
    toast('清理任务已启动（远端重排可能需数十秒）');activeReplayJob=x.job.id;renderReplayJobs();
    if($('viserTunnelHint'))$('viserTunnelHint').textContent='数据集重排中：Viser 隧道已关闭；下次开始回放会自动重建。';
    const jobId=x.job.id;
    for(let i=0;i<90;i++){
      await new Promise(r=>setTimeout(r,2000));
      await loadState();
      const iss=state.replay_issues?.[ds.path];
      if(iss){replayIssues=iss;state.replay_issues[ds.path]=iss}
      renderReplayIssues();renderReplayEpGrid();updateReplayMetrics();renderReplayJobs();
      const job=state.replay_jobs?.find(j=>j.id===jobId);
      if(job&&!['running'].includes(job.status)){
        await probeReplayDataset();
        toast(job.status==='completed'?(job.message||'清理完成'):(job.message||'清理结束'),job.status!=='completed');
        return;
      }
      if(iss&&iss.prune_status&&iss.prune_status!=='running'){
        await probeReplayDataset();
        toast(iss.prune_message||'清理结束',iss.prune_status==='error');
        return;
      }
    }
    toast('清理仍在进行，可稍后刷新数据集查看状态',true);
  }catch(e){toast(e.message,true);if(replayIssues){replayIssues.prune_status='pending';updateReplayMetrics()}}
}
async function resetReplaySession(silent=false){
  const path=replayDataset?.path||currentDatasetPath();
  if(!path)return toast('请先扫描数据集',true);
  try{
    const x=await post('/api/replay/issues/reset-session',{dataset_path:path});
    replayIssues=x.issues;state.replay_issues=state.replay_issues||{};state.replay_issues[path]=replayIssues;
    replaySelectedEps.clear();renderReplayEpGrid();renderReplayIssues();updateReplayMetrics();
    if(!silent)toast('已重置：勾选/清理状态已清空，失败记录保留')
  }catch(e){toast(e.message,true)}
}
let viserWatchTimer=null,viserWatchUntil=0;

function stopViserWatch(){
  if(viserWatchTimer){clearInterval(viserWatchTimer);viserWatchTimer=null}
  viserWatchUntil=0;
}
async function watchViserUntilReady(opts={}){
  const timeoutMs=Number(opts.timeoutMs||120000);
  const everyMs=Number(opts.everyMs||2000);
  stopViserWatch();
  const ph=$('viserPlaceholder');
  if(ph)ph.innerHTML='<div class="viser-icon">…</div><p>回放已启动 · 正在连接 Viser…</p><small>检测会与回放一起自动进行</small>';
  const shell=document.querySelector('.viser-shell');
  shell?.classList.remove('live');
  // Immediate heal + detect
  await checkViser(true);
  viserWatchUntil=Date.now()+timeoutMs;
  viserWatchTimer=setInterval(async()=>{
    const running=(state.replay_jobs||[]).some(j=>j.kind==='replay'&&j.status==='running');
    if(!running || Date.now()>viserWatchUntil){
      stopViserWatch();
      if(!running)void checkViser(true);
      return;
    }
    try{
      await checkViser(true);
      // stop early once http_ok / live
      const badge=$('viserBadge');
      if(badge && /已连接/.test(badge.textContent||'') && viserFrameLoaded){
        stopViserWatch();
        toast('Viser 已随回放自动连接');
      }
    }catch(_){}
  },everyMs);
}

async function startReplay(){
  try{
    const ds=await ensureReplayDataset();
    if(ds.sonic_replay&&ds.sonic_replay.ready===false){
      return toast(ds.sonic_replay.message||'当前数据集缺少 action.motion_token（V2.1）或 action.unified，无法 Sonic QA 仿真回放',true);
    }
    await resetReplaySession(true);
    const eps=getSelectedReplayEps();if(!eps.length)return toast('请选择至少一条 Episode',true);
    $('replayStartBtn').disabled=true;$('replayStopBtn').disabled=false;
    // Pre-heal tunnel so Viser detect starts with playback
    try{await api(`/api/replay/viser-check?port=${replayConfig.config?.viser_port||8081}&auto=1&heal=1`)}catch(_){}
    const x=await post('/api/replay/sessions',{
      host_id:resolveReplayHostId(),
      dataset_path:ds.path,
      dataset_name:ds.name,
      episodes:episodesToArg(eps),
      qa_root:$('replayQaRoot').value,
      viser_port:+(replayConfig.config?.viser_port||8081),
      live_ui:true,
    });
    activeReplayJob=x.job.id;
    if($('viserTunnelHint')){
      $('viserTunnelHint').hidden=false;
      if(x.execution==='local'){
        $('viserTunnelHint').textContent=`本机模式：Viser 直接访问 ${viserUrl}（与回放一并启动）`;
      }else if(x.tunnel?.alive){
        $('viserTunnelHint').textContent=`自动隧道已建立 → ${viserUrl}（与回放一并启动）`;
      }else{
        $('viserTunnelHint').textContent=x.tunnel_hint||'隧道未就绪，正在自动重试检测…';
      }
    }
    toast(x.tunnel?.alive?'回放已启动，Viser 检测已自动开始':'回放已启动，正在自动检测 Viser…');
    renderReplayJobs();
    await loadState();
    void watchViserUntilReady({timeoutMs:150000,everyMs:2000});
  }catch(e){toast(e.message,true);$('replayStartBtn').disabled=false;$('replayStopBtn').disabled=true;stopViserWatch()}
}
async function stopReplay(){
  const job=state.replay_jobs?.find(j=>j.id===activeReplayJob)||state.replay_jobs?.find(j=>j.kind==='replay'&&j.status==='running');
  if(!job)return toast('没有运行中的回放',true);
  try{
    stopViserWatch();
    await post(`/api/replay/sessions/${job.id}/cancel`);
    toast('已发送停止信号');
    $('replayStartBtn').disabled=false;$('replayStopBtn').disabled=true;
    await loadState();
    void checkViser(true);
  }catch(e){toast(e.message,true)}
}
async function refreshReplaySummary(){
  const job=state.replay_jobs?.find(j=>j.id===activeReplayJob)||state.replay_jobs?.find(j=>j.kind==='replay');
  if(!job)return toast('暂无回放任务',true);
  try{
    const x=await post(`/api/replay/sessions/${job.id}/refresh-summary`);
    replayIssues=x.issues;state.replay_issues[job.dataset_path]=replayIssues;renderReplayIssues();renderReplayEpGrid();toast(`检测完成：${x.summary?.n_fail||0} 条异常`)
  }catch(e){toast(e.message,true)}
}
function renderReplayJobs(){
  const el=$('replayJobList'),jobs=state.replay_jobs||[],countEl=$('replayJobCount');
  if(!el)return;
  if(countEl)countEl.textContent=`${jobs.length} 个任务`;
  if(!jobs.length){
    el.className='jobs-board-list empty';el.textContent='暂无任务 — 选择 Episode 后点击「开始回放」';
    if($('replayLog')){$('replayLog').dataset.jobBody='等待任务…';syncReplayLabelTipToLog();if(!replayLabelTip)$('replayLog').textContent='等待任务…'}
    return;
  }
  el.className='jobs-board-list';
  el.innerHTML=jobs.slice(0,12).map(j=>{
    const err=j.status==='error';
    const running=j.status==='running';
    const kind=j.kind==='prune'?'数据清理':'SONIC 回放';
    const where=j.execution==='local'?'本机':j.execution==='remote'?'远端':'';
    const epText=j.episodes||(j.delete_episodes?.length?`删除 ${j.delete_episodes.join(', ')}`:'—');
    const active=activeReplayJob===j.id||(running&&!activeReplayJob)?' active':'';
    const sum=j.summary;
    const stats=sum?`<div class="job-card-stats"><span><strong>${sum.n_ok??'—'}</strong>合格</span><span><strong>${sum.n_fail??'—'}</strong>异常</span></div>`:'';
    return `<div class="job-card${active}${err?' is-error':''}${running?' is-running':''}" onclick="selectReplayJob('${j.id}')"><div class="job-card-head"><b>${esc(j.id)}</b><span class="session-state ${err?'error':''}">${statusLabel(j.status)}</span></div><div class="job-card-meta"><span class="pill">${esc(kind)}</span>${where?`<span class="pill">${where}</span>`:''}<span class="pill">${esc(epText)}</span></div><div class="job-card-body"><small>${esc(j.created_at||'')}</small><div class="job-msg">${esc(j.message||'—')}</div>${stats}</div><div class="job-card-actions" onclick="event.stopPropagation()">${running?`<button onclick="stopReplayJob('${j.id}')">停止</button>`:`<button onclick="showReplayLog('${j.id}')">完整日志</button>`}${j.kind==='replay'&&j.status==='completed'?`<button onclick="refreshReplaySummaryById('${j.id}')">导入结果</button>`:''}</div></div>`
  }).join('');
  const active=jobs.find(j=>j.id===activeReplayJob)||jobs.find(j=>j.status==='running')||jobs[0];
  if(active&&!activeReplayJob)activeReplayJob=active.id;
  if($('replayLog')){
    $('replayLog').dataset.jobBody=(active?.logs?.length?(active.logs||[]).slice(-50).join('\n'):'等待任务…');
    syncReplayLabelTipToLog();
  }
  $('replayStartBtn').disabled=!!jobs.find(j=>j.kind==='replay'&&j.status==='running');
  $('replayStopBtn').disabled=!jobs.find(j=>j.kind==='replay'&&j.status==='running');
}
function selectReplayJob(id){activeReplayJob=id;renderReplayJobs()}
async function stopReplayJob(id){try{await post(`/api/replay/sessions/${id}/cancel`);await loadState()}catch(e){toast(e.message,true)}}
async function refreshReplaySummaryById(id){activeReplayJob=id;await refreshReplaySummary();await loadState()}
async function showReplayLog(id){try{const x=await api(`/api/replay/sessions/${id}`);manifest={logs:x.job.logs};$('jsonTitle').textContent=`${id} 日志`;$('jsonPath').textContent=x.job.command||'';$('jsonPreview').textContent=(x.job.logs||[]).join('\n')||'暂无日志';$('jsonModal').showModal()}catch(e){toast(e.message,true)}}
async function checkViser(silent=false){
  const port=replayConfig.config?.viser_port||8081;
  try{
    const mode=($('replayExecMode')?.value||replayExecution||'local');
    const runningRemote=(state.replay_jobs||[]).some(j=>j.kind==='replay'&&j.status==='running'&&(j.execution_mode==='remote'||j.host_id));
    const replayRunning=(state.replay_jobs||[]).some(j=>j.kind==='replay'&&j.status==='running');
    const auto=(mode==='remote'||runningRemote||replayRunning)?'&auto=1&heal=1':'&heal=1';
    const x=await api(`/api/replay/viser-check?port=${port}${auto}`);
    viserUrl=x.url||`http://127.0.0.1:${port}/`;
    const badge=$('viserBadge'),shell=document.querySelector('.viser-shell'),frame=$('viserFrame'),ph=$('viserPlaceholder');
    const want=String(viserUrl).replace(/\/?$/,'/');
    const cur=String(frame?.src||'');
    const blank=!cur||cur==='about:blank'||cur.endsWith('about:blank');
    const sameUrl=!blank&&(cur===want||cur===viserUrl||cur.replace(/\/$/,'')===want.replace(/\/$/,''));
    const httpOk=!!x.http_ok;
    const reachable=!!x.reachable;

    // Real Viser page ready
    if(httpOk){
      viserFailStreak=0;
      badge.textContent=x.tunnel?.auto?'Viser 已连接（自动隧道）':'Viser 已连接';
      badge.className='connection-badge online';
      shell?.classList.add('live');
      if(frame && (blank||!viserFrameLoaded||!sameUrl)){
        if(ph)ph.innerHTML='<div class="viser-icon">…</div><p>正在加载 Isaac / Viser</p>';
        frame.onload=()=>{viserFrameLoaded=true;shell?.classList.add('live')};
        frame.src=want;
        viserFrameLoaded=true;
        if(!silent)toast('Viser 窗口已加载');
      }
      return;
    }

    // Busy but alive — keep iframe if already loaded
    if(reachable && replayRunning && viserFrameLoaded && !blank){
      badge.textContent='Viser 加载中…';
      badge.className='connection-badge online';
      shell?.classList.add('live');
      return;
    }

    // Dead tunnel / empty response — never leave Firefox error page in iframe
    viserFailStreak+=1;
    badge.textContent='Viser 未连接';
    badge.className='connection-badge offline';
    shell?.classList.remove('live');
    if(frame && !blank){frame.src='about:blank';viserFrameLoaded=false}
    if(ph){
      ph.innerHTML=replayRunning
        ?'<div class="viser-icon">…</div><p>正在等待远端 Viser 启动…</p><small>隧道已尝试重建</small>'
        :'<div class="viser-icon">▶</div><p>Viser 未运行</p><small>请选择 Episode 后点「开始回放」</small>';
    }
    if(!silent)toast(x.message||'Viser 未连接，请先开始回放',true);
  }catch(e){if(!silent)toast(e.message,true)}
}
function openViserWindow(){window.open(viserUrl,'viser_replay','noopener,noreferrer')}
function reloadViserFrame(){
  const frame=$('viserFrame'),shell=document.querySelector('.viser-shell'),ph=$('viserPlaceholder');
  if(!frame)return;
  const want=String(viserUrl||'http://127.0.0.1:8081/').replace(/\/?$/,'/');
  viserFrameLoaded=false;
  shell?.classList.add('live');
  if(ph)ph.innerHTML='<div class="viser-icon">…</div><p>正在重新加载 Viser…</p>';
  frame.src='about:blank';
  setTimeout(()=>{frame.src=want;viserFrameLoaded=true;toast('已强制刷新 Viser 画面')},80);
}

/* ===== Train (05) ===== */
let trainCatalog=null,trainTask=null,activeTrainJob=null,trainChartPoints=[],trainPollTimer=null;
let trainChartGeom=null;
let trainChartModule='overview';
const TRAIN_CHART_MODULES=[
  {
    id:'overview',label:'总览',hint:'整体 loss + loss_z + hand（skill1 默认启用项）',
    focus:true,
    series:[
      {key:'loss',label:'loss',color:'#dff24b',width:2.0},
      {key:'loss_z',label:'loss_z',color:'#6ec6ff',width:1.5},
      {key:'loss_hand',label:'hand',color:'#ff9f6b',width:1.5},
    ],
  },
  {
    id:'loss',label:'整体 loss',hint:'加权总损失 loss = Σ w_i·loss_i（当前默认 w_z=1, w_hand=1）',
    focus:true,
    series:[
      {key:'loss',label:'loss',color:'#dff24b',width:2.1},
    ],
  },
  {
    id:'loss_z',label:'loss_z',hint:'学生 latent z_pred 对齐老师 z*（z-scored MSE）；W_Z=1 默认开启',
    focus:true,
    series:[
      {key:'loss_z',label:'loss_z',color:'#6ec6ff',width:2.1},
    ],
  },
  {
    id:'loss_z_phys',label:'z_phys',hint:'sonic token 物理尺度诊断项（不进加权总 loss，日志 z_phys=）',
    focus:true,
    series:[
      {key:'loss_z_phys',label:'z_phys',color:'#a78bfa',width:2.1},
    ],
  },
  {
    id:'loss_hand',label:'手部 hand',hint:'手部 12 维相对专家；W_HAND=1 默认开启',
    focus:true,
    series:[
      {key:'loss_hand',label:'hand',color:'#ff9f6b',width:2.1},
    ],
  },
  {
    id:'loss_q_head',label:'q_head',hint:'关节 q_head vs ATM；当前脚本 W_Q_HEAD=0（通常无曲线）',
    focus:true,
    series:[
      {key:'loss_q_head',label:'q_head',color:'#5eead4',width:2.4},
    ],
  },
  {
    id:'loss_smpl',label:'smpl',hint:'SMPL pose；当前脚本 W_SMPL=0（通常无曲线）',
    focus:true,
    series:[
      {key:'loss_smpl',label:'smpl',color:'#f472b6',width:2.4},
    ],
  },
  {
    id:'aux',label:'辅助',hint:'grad_l2 / dagger_beta（jsonl 有则显示；自动纵轴）',
    focus:false,
    series:[
      {key:'grad_l2',label:'grad_l2',color:'#fbbf24',width:2},
      {key:'dagger_beta',label:'dagger_β',color:'#34d399',width:2},
    ],
  },
];
function currentTrainChartModule(){
  return TRAIN_CHART_MODULES.find(m=>m.id===trainChartModule)||TRAIN_CHART_MODULES[0];
}
function moduleHasData(mod,points){
  if(!points?.length)return false;
  return mod.series.some(s=>points.some(p=>p[s.key]!=null&&isFinite(p[s.key])));
}
function renderTrainChartTabs(){
  const el=$('trainChartTabs');if(!el)return;
  el.innerHTML=TRAIN_CHART_MODULES.map(m=>{
    const has=moduleHasData(m,trainChartPoints);
    const cls=['train-chart-tab',m.id===trainChartModule?'active':'',(!has&&trainChartPoints.length)?'dim':''].filter(Boolean).join(' ');
    return `<button type="button" role="tab" class="${cls}" data-mod="${m.id}" aria-selected="${m.id===trainChartModule}">${m.label}</button>`;
  }).join('');
  el.querySelectorAll('button[data-mod]').forEach(btn=>{
    btn.onclick=()=>{
      trainChartModule=btn.getAttribute('data-mod')||'overview';
      renderTrainChartTabs();
      drawTrainChart(trainChartPoints);
    };
  });
  const hint=$('trainChartModuleHint');
  const mod=currentTrainChartModule();
  if(hint)hint.textContent=mod?.hint||'';
}
let trainMixMode=false,trainMixSelected=new Set();
let trainBrowseTree=null,trainBrowseSelected=new Set(),trainBrowseCollapsed=new Set();

function parseTrainPathList(text){
  return [...new Set(String(text||'').split(/[\n,;]+/).map(s=>s.trim()).filter(Boolean))];
}
function getTrainSelectedPaths(){
  if(trainBrowseSelected.size){
    // Stable order: tree leaf order when possible (not Set insertion / leftover unified first)
    const ordered=[];
    const seen=new Set();
    const push=(p)=>{const n=normTrainPath(p);if(!n||seen.has(n))return;seen.add(n);ordered.push(p)};
    for(const f of (trainBrowseTree?.folders||[])){
      const kids=f.children||[];
      if(kids.length)kids.forEach(c=>{if(trainPathSelected(c.path))push(c.path)});
      else if(f.kind==='dataset'&&trainPathSelected(f.path))push(f.path);
    }
    for(const p of trainBrowseSelected)push(p);
    return ordered;
  }
  return parseTrainPathList($('trainDatasetPath')?.value||'');
}
function syncTrainPathBoxFromSelection(){
  const el=$('trainDatasetPath');if(!el)return;
  el.value=getTrainSelectedPaths().join('\n');
  if($('trainBrowseHint')){
    const n=trainBrowseSelected.size;
    $('trainBrowseHint').textContent=n?`已勾选 ${n} 项 · 可跨 skill5 / skill_5_throw… 合并 · 仅 valid · pack 保留到 Phi_0_train_data`:'勾选 session 或整个技能文件夹 · 可跨多个 830demo 目录合并训练';
  }
}
function clearTrainBrowseSelection(){
  trainBrowseSelected=new Set();
  syncTrainPathBoxFromSelection();
  renderTrainDatasetTree();
  void probeTrainSelectedPaths();
}
async function refreshTrainBrowse(){
  const el=$('trainDatasetTree');if(!el)return;
  el.className='train-dataset-tree';el.textContent='正在扫描 830demo…';
  try{
    const root=(state.shared?.root||state.skills_config?.remote_base||'/mnt/data2/wpy/workspace/830demo');
    const x=await api(`/api/train/browse-datasets?root=${encodeURIComponent(root)}`);
    trainBrowseTree=x;
    reconcileTrainBrowseSelection();
    renderTrainDatasetTree();
    syncTrainPathBoxFromSelection();
    if(!trainBrowseSelected.size&&$('trainBrowseHint'))$('trainBrowseHint').textContent=`${x.root||root} · ${(x.folders||[]).length} 个文件夹`;
  }catch(e){
    el.className='train-dataset-tree empty';el.textContent='扫描失败：'+e.message;
  }
}
function toggleTrainBrowseFolder(path, on){
  const folder=(trainBrowseTree?.folders||[]).find(f=>f.path===path);
  if(!folder)return;
  const kids=(folder.children||[]).map(c=>c.path);
  // Drop leftover skill-card unified paths when picking from 830demo tree
  const treePaths=new Set();
  for(const f of (trainBrowseTree?.folders||[])){
    const ck=f.children||[];
    if(ck.length)ck.forEach(c=>treePaths.add(normTrainPath(c.path)));
    else if(f.kind==='dataset')treePaths.add(normTrainPath(f.path));
  }
  trainBrowseSelected=new Set([...trainBrowseSelected].filter(p=>treePaths.has(normTrainPath(p))));
  if(!kids.length){
    if(on)trainBrowseSelected.add(path);else trainBrowseSelected.delete(path);
  }else if(on){
    kids.forEach(p=>trainBrowseSelected.add(p));
  }else{
    kids.forEach(p=>trainBrowseSelected.delete(p));
  }
  syncTrainPathBoxFromSelection();
  renderTrainDatasetTree();
  void probeTrainSelectedPaths();
}
function toggleTrainBrowseLeaf(path, on){
  const treePaths=new Set();
  for(const f of (trainBrowseTree?.folders||[])){
    const ck=f.children||[];
    if(ck.length)ck.forEach(c=>treePaths.add(normTrainPath(c.path)));
    else if(f.kind==='dataset')treePaths.add(normTrainPath(f.path));
  }
  trainBrowseSelected=new Set([...trainBrowseSelected].filter(p=>treePaths.has(normTrainPath(p))));
  if(on)trainBrowseSelected.add(path);else trainBrowseSelected.delete(path);
  syncTrainPathBoxFromSelection();
  renderTrainDatasetTree();
  void probeTrainSelectedPaths();
}
function toggleTrainBrowseCollapse(path){
  if(trainBrowseCollapsed.has(path))trainBrowseCollapsed.delete(path);
  else trainBrowseCollapsed.add(path);
  renderTrainDatasetTree();
}
function normTrainPath(p){return String(p||'').replace(/\/+$/,'')}
function trainPathSelected(path){
  const want=normTrainPath(path);
  if(!want)return false;
  if(trainBrowseSelected.has(path)||trainBrowseSelected.has(want))return true;
  for(const p of trainBrowseSelected){if(normTrainPath(p)===want)return true}
  return false;
}
/** After browse: if selected paths don't match tree leaves, map by basename under same skill. */
function reconcileTrainBrowseSelection(){
  const folders=trainBrowseTree?.folders||[];
  if(!folders.length||!trainBrowseSelected.size)return;
  const leafByPath=new Map();
  const leaves=[];
  for(const f of folders){
    const kids=f.children||[];
    if(kids.length)kids.forEach(c=>{leafByPath.set(normTrainPath(c.path),c.path);leaves.push(c)});
    else if(f.kind==='dataset'){leafByPath.set(normTrainPath(f.path),f.path);leaves.push(f)}
  }
  const next=new Set();
  let changed=false;
  for(const raw of trainBrowseSelected){
    const n=normTrainPath(raw);
    if(leafByPath.has(n)){next.add(leafByPath.get(n));if(leafByPath.get(n)!==raw)changed=true;continue}
    const base=n.split('/').pop();
    const hit=leaves.find(c=>c.name===base||normTrainPath(c.path).endsWith('/'+base));
    if(hit){next.add(hit.path);changed=true}
    else next.add(raw);
  }
  if(changed || next.size!==trainBrowseSelected.size || [...trainBrowseSelected].some(p=>!next.has(p))){
    trainBrowseSelected=next;
    syncTrainPathBoxFromSelection();
  }
}
function renderTrainDatasetTree(){
  const el=$('trainDatasetTree');if(!el)return;
  const folders=trainBrowseTree?.folders||[];
  if(!folders.length){el.className='train-dataset-tree empty';el.textContent='未发现数据集，点击「扫描」重试';return}
  el.className='train-dataset-tree';
  el.innerHTML=folders.map((f)=>{
    const kids=f.children||[];
    if(f.kind==='dataset'||(!kids.length&&f.kind!=='skill')){
      const checked=trainPathSelected(f.path)?'checked':'';
      return `<label class="train-tree-kid${checked?' is-selected':''}"><input type="checkbox" data-browse="leaf" data-path="${esc(f.path)}" ${checked}><span>${esc(f.name)}</span><span class="meta">${f.total_episodes!=null?f.total_episodes+' eps':''}</span></label>`;
    }
    const allOn=kids.length&&kids.every(c=>trainPathSelected(c.path));
    const someOn=!allOn&&kids.some(c=>trainPathSelected(c.path));
    const collapsed=trainBrowseCollapsed.has(f.path);
    const kidsHtml=kids.map(c=>{
      const checked=trainPathSelected(c.path)?'checked':'';
      const prompt=c.prompt&&c.prompt.toLowerCase()!=='demo'?c.prompt:'';
      return `<label class="train-tree-kid${checked?' is-selected':''}"><input type="checkbox" data-browse="leaf" data-path="${esc(c.path)}" ${checked}><span>${esc(c.name)}</span><span class="meta">${c.total_episodes!=null?c.total_episodes+' eps':''}${prompt?' · '+esc(prompt):''}</span></label>`;
    }).join('');
    return `<div class="train-tree-folder" data-folder-path="${esc(f.path)}"><div class="train-tree-folder-row${allOn||someOn?' is-selected':''}"><input type="checkbox" data-browse="folder" data-path="${esc(f.path)}" ${allOn?'checked':''} ${someOn?'data-partial="1"':''}><span class="train-tree-folder-name" data-browse="folder-label" data-path="${esc(f.path)}">${esc(f.name)}</span><span class="meta">${kids.length} sessions</span><button type="button" class="text-btn" data-browse="collapse" data-path="${esc(f.path)}">${collapsed?'展开':'收起'}</button></div>${collapsed?'':`<div class="train-tree-kids">${kidsHtml||'<span class="hint quiet">无 session</span>'}</div>`}</div>`;
  }).join('');
  el.querySelectorAll('input[data-browse="folder"][data-partial="1"]').forEach(inp=>{inp.indeterminate=true});
  if(!el._browseBound){
    el._browseBound=true;
    el.addEventListener('click',e=>{
      const btn=e.target.closest('[data-browse="collapse"]');
      if(btn){e.preventDefault();toggleTrainBrowseCollapse(btn.getAttribute('data-path')||'');return}
      const fname=e.target.closest('[data-browse="folder-label"]');
      if(fname){
        e.preventDefault();
        const path=fname.getAttribute('data-path')||'';
        const folder=(trainBrowseTree?.folders||[]).find(f=>f.path===path);
        const kids=folder?.children||[];
        const allOn=kids.length?kids.every(c=>trainPathSelected(c.path)):trainPathSelected(path);
        toggleTrainBrowseFolder(path, !allOn);
      }
    });
    el.addEventListener('change',e=>{
      const inp=e.target;
      if(!(inp instanceof HTMLInputElement)||!inp.matches('input[type="checkbox"][data-browse]'))return;
      const path=inp.getAttribute('data-path')||'';
      if(!path)return;
      if(inp.getAttribute('data-browse')==='folder')toggleTrainBrowseFolder(path, inp.checked);
      else toggleTrainBrowseLeaf(path, inp.checked);
    });
  }
}
async function initTrainPage(){
  try{
    const x=await api('/api/train/catalog');
    trainCatalog=x.catalog;state.train_jobs=x.jobs||state.train_jobs||[];
    if(x.skills)state.skills=x.skills;
    if(x.config)state.train_config=x.config;
    if(x.hosts)state.hosts=x.hosts;
    fillTrainHostSelect();
    if(x.config?.host_id&&$('trainExecMode')){
      const want=x.config.host_id;
      if([...$('trainExecMode').options].some(o=>o.value===want&&!o.disabled))$('trainExecMode').value=want;
    }else if(x.config?.execution_mode==='local'){
      $('trainExecMode').value='local';
    }
    trainMixMode=false;trainMixSelected=new Set();
    if($('trainMixBar'))$('trainMixBar').hidden=true;
    renderTrainCards();renderTrainJobs();
    const runningJobs=(state.train_jobs||[]).filter(j=>j.status==='running');
    if(runningJobs.length){
      if(!activeTrainJob||!runningJobs.some(j=>j.id===activeTrainJob))activeTrainJob=runningJobs[0].id;
      const job=runningJobs.find(j=>j.id===activeTrainJob)||runningJobs[0];
      const task=(trainCatalog.tasks||[]).find(t=>t.id===job.task_id);
      if(task)openTrainWorkspace(task,true);
      updateTrainMonitorBanner(job);
      await refreshTrainMetrics();await refreshTrainLogs();await refreshTrainCkpts();
    }
    startTrainPolling();
    void refreshTrainBrowse();
  }catch(e){toast(e.message,true);$('trainCardGrid').textContent='加载失败：'+e.message}
}
function renderTrainCards(){
  const el=$('trainCardGrid'),tasks=trainCatalog?.tasks||[];
  el.className='train-card-grid';
  if(!tasks.length){
    el.className='train-card-grid empty';
    el.innerHTML='暂无训练配方。请检查 phi0_pipeline/train_catalog.json，或在 01 / SK 创建技能卡。';
  }
  const skillIds=new Set((state.skills||[]).map(s=>s.id));
  const hiddenN=(trainCatalog?.hidden_task_ids||[]).length;
  const cards=tasks.map(t=>{
    const readyN=(t.datasets||[]).filter(d=>d.probe?.ready).length;
    const accent=t.accent||'#1d6b4b';
    const stop=`e${t.defaults?.epochs??4}`;
    const mixOn=trainMixMode&&trainMixSelected.has(t.id);
    const click=trainMixMode?`toggleTrainMixSelect('${t.id}')`:`openTrainWorkspaceById('${t.id}')`;
    const delKind=t.custom?'custom':(skillIds.has(t.id)&&!t.builtin?'skill':'builtin');
    const delBtn=(!trainMixMode)
      ?`<button type="button" class="train-card-del" title="从列表移除" onclick="event.stopPropagation();deleteTrainCard('${esc(t.id)}','${delKind}')">删除</button>`
      :'';
    const kindPill=t.custom?'<span class="pill">自定义</span>':(t.builtin?'<span class="pill">内置</span>':(skillIds.has(t.id)?'<span class="pill">本机技能</span>':''));
    const profilePill=t.train_profile?`<span class="pill">${esc(t.train_profile)}</span>`:'';
    const blockPill=t.train_blocked?`<span class="pill not-ready" title="${esc(t.train_block_reason||'不可训练')}">需 REF</span>`:(t.train_pack_on_start?`<span class="pill" title="${esc(t.train_hint||'启动时 pack')}">启动 pack</span>`:'');
    return `<div class="train-card${mixOn?' mix-selected':''}${t.train_blocked?' train-card-blocked':''}" style="border-top-color:${accent}" onclick="${click}">
      <div class="train-card-top"><span class="badge">${esc(t.badge||t.skill||'SKILL')}</span>${delBtn}</div>
      <h3>${esc(t.title)}</h3><p>${esc(t.subtitle||'')}</p><p>${esc(t.description||'')}</p>
      <div class="meta"><span class="pill ${readyN?'ready':'not-ready'}">${readyN}/${(t.datasets||[]).length||0} 数据集</span><span class="pill">B=${t.defaults?.num_envs||32}×${t.defaults?.ngpu||8}</span><span class="pill">${esc(stop)}</span>${profilePill}${blockPill}${kindPill}${mixOn?'<span class="pill ready">已选</span>':''}</div>
    </div>`;
  }).join('');
  const mixCard=trainMixMode?'':`<button type="button" class="train-card train-card-mix" onclick="enterTrainMixMode()"><span class="badge">MIX</span><h3>混合训练</h3><p>点选 ≥2 张技能卡：每技能独立 prompt + raw session，再合并成 mix 训。</p><p>超参共用；也可回退用已有 830mix_*_unified。</p><div class="meta"><span class="pill">mix_vision_isaac</span><span class="pill">分技能 pack</span></div></button>`;
  const addCard=trainMixMode?'':`<button type="button" class="train-card train-card-add" onclick="openTrainCardCreateModal()"><span class="badge">NEW</span><h3>新建技能卡</h3><p>自定义训练配方（标题 / 数据集路径 / epochs）。</p><p>也可在 SK 页创建全链路技能。</p><div class="meta"><span class="pill">自定义</span></div></button>`;
  const restoreCard=(!trainMixMode&&hiddenN)?`<button type="button" class="train-card train-card-add" onclick="restoreHiddenTrainCards()"><span class="badge">UNDO</span><h3>恢复已隐藏</h3><p>已整理隐藏 ${hiddenN} 张内置/列表卡片，点此全部恢复。</p><div class="meta"><span class="pill">整理</span></div></button>`:'';
  el.innerHTML=(cards||'')+mixCard+addCard+restoreCard;
  updateTrainMixBar();
}
async function deleteTrainCard(id, kind){
  const t=(trainCatalog?.tasks||[]).find(x=>x.id===id);
  const title=t?.title||id;
  if(kind==='custom'){
    if(!confirm(`确认删除自定义训练卡「${title}」？\n\n不会删除 cluster 上的数据集或 ckpt。`))return;
    try{
      const x=await del(`/api/train/tasks/${encodeURIComponent(id)}`);
      if(x.catalog)trainCatalog=x.catalog;
      toast('已删除自定义训练卡');
      if(trainTask?.id===id)backToTrainCards();
      await initTrainPage();
    }catch(e){toast(e.message,true)}
    return;
  }
  if(kind==='skill'){
    if(!confirm(`确认删除技能卡「${title}」？\n\n会删除本机技能目录与挂载记录；不会删除 cluster 上的数据集文件。`))return;
    try{
      await del(`/api/skills/${encodeURIComponent(id)}`);
      toast('技能卡已删除');
      if(trainTask?.id===id)backToTrainCards();
      try{const x=await api('/api/skills');state.skills=x.skills||[]}catch(_){}
      await initTrainPage();
    }catch(e){toast(e.message,true)}
    return;
  }
  if(!confirm(`从列表移除「${title}」？\n\n仅整理 03 卡片列表（隐藏内置配方），不修改 train_catalog.json，也不删远端数据。\n需要时可点「恢复已隐藏」。`))return;
  try{
    const x=await del(`/api/train/tasks/${encodeURIComponent(id)}`);
    if(x.catalog)trainCatalog=x.catalog;
    toast('已从列表隐藏');
    if(trainTask?.id===id)backToTrainCards();
    await initTrainPage();
  }catch(e){toast(e.message,true)}
}
async function restoreHiddenTrainCards(){
  const n=(trainCatalog?.hidden_task_ids||[]).length;
  if(!n)return toast('没有已隐藏的卡片');
  if(!confirm(`恢复全部 ${n} 张已隐藏的训练卡到列表？`))return;
  try{
    const x=await post('/api/train/tasks/restore-hidden',{});
    if(x.catalog)trainCatalog=x.catalog;
    toast(`已恢复 ${((x.restored)||[]).length} 张`);
    await initTrainPage();
  }catch(e){toast(e.message,true)}
}
function openTrainCardCreateModal(){
  if($('trainCardCreateModal'))$('trainCardCreateModal').showModal();
}
async function submitTrainCardCreate(){
  const title=($('trainNewTitle')?.value||'').trim();
  const path=($('trainNewPath')?.value||'').trim();
  const prompt=($('trainNewPrompt')?.value||'').trim();
  const badge=($('trainNewBadge')?.value||'').trim();
  const epochs=Math.max(1,+$('trainNewEpochs')?.value||4);
  if(!title)return toast('请填写标题',true);
  if(!path)return toast('请填写数据集绝对路径',true);
  try{
    const x=await post('/api/train/tasks',{
      title,badge,prompt,default_prompt:prompt,
      dataset_path:path,remote_path:path,epochs,
    });
    if(x.catalog)trainCatalog=x.catalog;
    $('trainCardCreateModal')?.close();
    ['trainNewTitle','trainNewPath','trainNewPrompt','trainNewBadge'].forEach(id=>{if($(id))$(id).value=''});
    if($('trainNewEpochs'))$('trainNewEpochs').value='4';
    toast('已创建自定义训练卡');
    renderTrainCards();
    if(x.task)openTrainWorkspace(x.task);
  }catch(e){toast(e.message,true)}
}
function enterTrainMixMode(){
  if(!(trainCatalog?.tasks||[]).length)return toast('请先在 01 上传创建技能卡',true);
  trainMixMode=true;trainMixSelected=new Set();
  if($('trainMixBar'))$('trainMixBar').hidden=false;
  renderTrainCards();
  toast('已进入混合训练：点选至少 2 张技能卡');
}
function cancelTrainMixMode(){trainMixMode=false;trainMixSelected=new Set();if($('trainMixBar'))$('trainMixBar').hidden=true;renderTrainCards()}
function toggleTrainMixSelect(id){
  if(trainMixSelected.has(id))trainMixSelected.delete(id);else trainMixSelected.add(id);
  renderTrainCards();
}
function updateTrainMixBar(){
  const n=trainMixSelected.size;
  if($('trainMixHint'))$('trainMixHint').textContent=`已选 ${n} 个技能`+(n>=2?' · 可确认混训':' · 至少再选 '+(2-n)+' 个');
  if($('trainMixConfirmBtn'))$('trainMixConfirmBtn').disabled=n<2;
}
async function confirmTrainMix(){
  if(trainMixSelected.size<2)return toast('请至少选择 2 个技能',true);
  try{
    const x=await post('/api/train/mix',{skill_ids:[...trainMixSelected]});
    trainMixMode=false;if($('trainMixBar'))$('trainMixBar').hidden=true;
    openTrainWorkspace(x.task);
    const n=(x.task?.mix_slots||[]).length;
    const compose=x.task?.mix_compose;
    toast(compose
      ?`已组合混训 · ${n} 技能槽 · 每技能填 prompt/数据集后启动`
      :`已组合混训 · 回退预置包 REF=${(x.task?.ref_root||'').split('/').pop()||'?'}`);
  }catch(e){toast(e.message,true)}
}
function openTrainWorkspaceById(id){const t=(trainCatalog?.tasks||[]).find(x=>x.id===id);if(t)openTrainWorkspace(t)}
function resolveTrainPrompt(task, ds, probePrompt){
  const cands=[probePrompt, ds?.probe?.prompt, ds?.prompt, task?.default_prompt];
  for(const c of cands){
    const t=String(c||'').trim();
    if(t && !['demo','task','none','null','n/a','-'].includes(t.toLowerCase()))return t;
  }
  return String(task?.default_prompt||'').trim();
}
function isTrainMixCompose(task){
  return !!(task?.mix && Array.isArray(task.mix_slots) && task.mix_slots.length>=2 && (task.mix_compose || task.mix_slots.some(s=>(s.candidates||[]).length||(s.dataset_paths||[]).length)));
}
function setTrainMixComposeUI(on){
  const hideIds=['trainDatasetLabel','trainPromptLabel','trainDatasetPathLabel','trainSingleBrowseBlock'];
  hideIds.forEach(id=>{const el=$(id);if(el)el.hidden=!!on;});
  if($('trainMixSlots'))$('trainMixSlots').hidden=!on;
  if($('trainParamsHint')){
    $('trainParamsHint').textContent=on
      ?'混训：下方每技能各自 prompt + 数据集；Epochs/CUDA 等超参共用'
      :'仅使用 02 导出的 valid allowlist';
  }
}
function renderTrainMixSlots(){
  const el=$('trainMixSlots');
  if(!el)return;
  const slots=trainTask?.mix_slots||[];
  if(!isTrainMixCompose(trainTask)||!slots.length){
    el.hidden=true;el.innerHTML='';return;
  }
  el.hidden=false;
  el.innerHTML=slots.map((slot,idx)=>{
    const sid=esc(slot.skill_id||`skill${idx}`);
    const title=esc(slot.title||slot.skill_id||`技能 ${idx+1}`);
    const badge=esc(slot.badge||slot.skill_id||'SK');
    const prompt=esc(slot.prompt||'');
    const selected=new Set((slot.dataset_paths||[]).map(String));
    const cands=slot.candidates||[];
    const rows=cands.length
      ?cands.map(c=>{
        const p=String(c.path||c.remote_path||'');
        const checked=selected.has(p)?' checked':'';
        const vc=c.valid_count!=null?` · valid ${c.valid_count}`:'';
        return `<label><input type="checkbox" data-mix-skill="${sid}" data-mix-path="${esc(p)}"${checked} onchange="onMixSlotPathChange('${sid}', this)"><span><b>${esc(c.label||c.id||p.split('/').pop())}</b>${vc}<br><span class="quiet">${esc(p)}</span></span></label>`;
      }).join('')
      :`<p class="train-mix-slot-empty">该技能暂无 raw session 候选。请先在技能卡挂载 830demo 并在 02 导出 valid。</p>`;
    return `<div class="train-mix-slot" data-skill="${sid}">
      <div class="train-mix-slot-head"><span class="badge">${badge}</span><h3>${title}</h3></div>
      <label>Prompt<input type="text" data-mix-prompt="${sid}" value="${prompt}" oninput="onMixSlotPromptChange('${sid}', this)"></label>
      <div class="train-mix-slot-cands">${rows}</div>
    </div>`;
  }).join('');
}
function findMixSlot(skillId){
  return (trainTask?.mix_slots||[]).find(s=>String(s.skill_id)===String(skillId));
}
function onMixSlotPromptChange(skillId, el){
  const slot=findMixSlot(skillId);
  if(slot)slot.prompt=String(el?.value||'');
}
function onMixSlotPathChange(skillId, el){
  const slot=findMixSlot(skillId);
  if(!slot)return;
  const path=el?.getAttribute?.('data-mix-path')||'';
  const cur=new Set((slot.dataset_paths||[]).map(String));
  if(el?.checked)cur.add(path);else cur.delete(path);
  slot.dataset_paths=[...cur];
  updateMixSlotsReadyHint();
}
function collectMixSlotsFromUI(){
  return (trainTask?.mix_slots||[]).map(slot=>{
    const sid=String(slot.skill_id||'');
    const promptEl=document.querySelector(`input[data-mix-prompt="${CSS.escape(sid)}"]`);
    const boxes=[...document.querySelectorAll(`input[data-mix-skill="${CSS.escape(sid)}"]`)];
    const paths=boxes.filter(b=>b.checked).map(b=>b.getAttribute('data-mix-path')||'').filter(Boolean);
    return {
      skill_id:sid,
      title:slot.title||sid,
      prompt:String(promptEl?.value??slot.prompt??'').trim()||'task',
      dataset_paths:paths.length?paths:(slot.dataset_paths||[]),
    };
  });
}
function updateMixSlotsReadyHint(){
  const ready=$('trainDatasetReady');
  if(!ready||!isTrainMixCompose(trainTask))return;
  const slots=collectMixSlotsFromUI();
  const ok=slots.filter(s=>(s.dataset_paths||[]).length>0);
  const missing=slots.filter(s=>(s.dataset_paths||[]).length===0).map(s=>s.skill_id||s.title);
  ready.className='scan-preview';
  ready.innerHTML=`<b>混训组成</b><span>${ok.length}/${slots.length} 技能已选数据集 · 超参共用</span>`+
    (missing.length?`<span style="color:#b45309">缺数据集：${esc(missing.join(', '))}</span>`:'')+
    `<span>启动后分技能 pack → merge → mix_vision_isaac；pack 保留在 Phi_0_train_data/mix_&lt;时间&gt;</span>`;
  if($('trainStartBtn'))$('trainStartBtn').disabled=ok.length<2;
}
function openTrainWorkspace(task,keepJob=false){
  if(task?.train_blocked && !keepJob){
    toast(task.train_block_reason||'该技能卡当前不可训练：缺少 train_profile 或单技能 unified REF',true);
  }
  trainTask=task;$('trainCardView').hidden=true;$('trainWorkspace').hidden=false;
  $('trainWsTitle').textContent=task.title;$('trainWsBadge').textContent=(task.badge||'TRAIN');$('trainWsSkill').textContent=task.skill||task.id;
  const profileHint=task.train_profile?` · ${task.train_profile}`:'';
  if($('trainWsSkill'))$('trainWsSkill').textContent=`${task.skill||task.id}${profileHint}`;
  const mixCompose=isTrainMixCompose(task);
  setTrainMixComposeUI(mixCompose);
  if(mixCompose){
    if(!Array.isArray(task.mix_slots))task.mix_slots=[];
    task.mix_slots.forEach(s=>{
      if(!Array.isArray(s.dataset_paths))s.dataset_paths=[];
      if(!Array.isArray(s.candidates))s.candidates=[];
    });
    renderTrainMixSlots();
    updateMixSlotsReadyHint();
  }else{
    renderTrainMixSlots();
  }
  const ds=(task.datasets||[]).filter(d=>{
    const p=String(d.remote_path||d.path||'');
    if(task.train_profile==='mix_vision_isaac'||task.mix) return true;
    // Hide mix packs from single-skill dropdown to avoid accidental wrong REF.
    return !(p.toLowerCase().includes('830mix')||p.toLowerCase().includes('_mix_'));
  });
  if($('trainDataset')){
    $('trainDataset').innerHTML=ds.map(d=>{
      const usePath=(d.remote_path&&String(d.remote_path).startsWith('/mnt'))?d.remote_path:d.path;
      const mark=String(usePath||d.label||'').toLowerCase().includes('unified')?' · unified':'';
      return `<option value="${esc(d.path)}" data-remote="${esc(d.remote_path||'')}">${esc(d.label||d.id)}${d.probe?.ready?' ✓':''}${mark}</option>`;
    }).join('')+(ds.length?'':`<option value="">暂无单技能 *_unified（请先 pack / 挂载）</option>`);
  }
  const preferred=task.preferred_dataset&&ds.some(d=>d.path===task.preferred_dataset.path)
    ?task.preferred_dataset
    :(ds.find(d=>String(d.path||d.remote_path||'').toLowerCase().includes('unified')&&!String(d.path||'').toLowerCase().includes('830mix'))||ds[0]);
  if(!mixCompose){
    if(preferred){
      $('trainDataset').value=preferred.path;
      const usePath=(preferred.remote_path&&String(preferred.remote_path).startsWith('/mnt'))?preferred.remote_path:preferred.path;
      trainBrowseSelected=new Set(usePath?[usePath]:[]);
      syncTrainPathBoxFromSelection();
      $('trainPrompt').value=resolveTrainPrompt(task, preferred, preferred.probe?.prompt);
    }else{$('trainDatasetPath').value='';trainBrowseSelected=new Set();$('trainPrompt').value=task.default_prompt||''}
  }else if($('trainPrompt')){
    $('trainPrompt').value=task.default_prompt||'';
  }
  const p=task.defaults||{};
  $('trainLr').value=p.lr??'1e-4';
  $('trainWarmupRatio').value=p.warmup_ratio??0.05;
  $('trainNumEnvs').value=p.num_envs??32;
  $('trainEpochs').value=Math.max(1,+(p.epochs??4));
  const epochCkpt=Math.max(0,+(p.ckpt_every_epoch??1));
  if($('trainCkptEveryEpoch'))$('trainCkptEveryEpoch').value=epochCkpt;
  // Default is epoch-boundary saves; step cadence only when epoch==0.
  $('trainCkptEvery').value=epochCkpt>0?0:(p.ckpt_every??0);
  if($('trainCkptStepKeep'))$('trainCkptStepKeep').value=p.ckpt_step_keep??3;
  syncTrainCkptMode();
  $('trainHand').value=p.hand_mode||'dex3';
  $('trainCuda').value=p.cuda_devices||[...Array(+(p.ngpu||8)).keys()].join(',');
  // Always start a new run dir under Phi_0_model_zoo; do not reuse last train_out_dir.
  $('trainOutDir').value='';
  if(!keepJob)activeTrainJob=null;
  updateTrainEffBatch();
  if(!mixCompose){
    if(!trainBrowseTree)void refreshTrainBrowse();else renderTrainDatasetTree();
    void onTrainDatasetChange();
  }
  void refreshTrainCkpts();
}
function backToTrainCards(){
  $('trainWorkspace').hidden=true;$('trainCardView').hidden=false;trainTask=null;
  setTrainMixComposeUI(false);
  if($('trainMixSlots')){$('trainMixSlots').hidden=true;$('trainMixSlots').innerHTML='';}
  renderTrainCards();
}
function markTrainDatasetManual(){
  const paths=parseTrainPathList($('trainDatasetPath')?.value||'');
  trainBrowseSelected=new Set(paths);
  renderTrainDatasetTree();
  void probeTrainSelectedPaths();
}
/** NGPU always follows CUDA list length — never a separate control. */
function parseTrainCudaIds(raw){
  return String(raw||'').split(',').map(s=>s.trim()).filter(Boolean);
}
function trainNgpuFromCuda(){
  const ids=parseTrainCudaIds($('trainCuda')?.value||'');
  return ids.length||0;
}
function syncTrainCkptMode(){
  const epochEl=$('trainCkptEveryEpoch');
  const stepEl=$('trainCkptEvery');
  if(!epochEl||!stepEl)return;
  const epochN=Math.max(0,+epochEl.value||0);
  if(epochN>0){
    stepEl.value='0';
    stepEl.disabled=true;
    stepEl.title='epoch 模式生效中：按 step 存 ckpt 已关闭（将「每 N epoch」设为 0 可启用）';
  }else{
    stepEl.disabled=false;
    stepEl.title='按 step 存 ckpt；建议仅在关闭 epoch 模式时使用';
    if(!(+stepEl.value>0))stepEl.value='10000';
  }
}
function updateTrainEffBatch(){
  const numEnvs=+$('trainNumEnvs').value||0;
  const ngpu=trainNgpuFromCuda();
  const epochs=Math.max(1,+$('trainEpochs').value||1);
  const eff=numEnvs&&ngpu?numEnvs*ngpu:'—';
  $('trainEffBatch').textContent=ngpu
    ?`GPU×${ngpu} · effective batch ${eff}`
    :'effective batch —（请填写 CUDA）';
  if($('trainCuda')&&!($('trainCuda').value||'').trim()){
    const defN=+(trainTask?.defaults?.ngpu||8)||8;
    $('trainCuda').placeholder=[...Array(defN).keys()].join(',')||'0';
  }
  const frames=+(window._trainProbeFrames||0);
  const hint=$('trainStepsHint');
  if(!hint)return;
  if(frames>0&&numEnvs>0){
    const spe=Math.max(1,Math.ceil(frames/numEnvs));
    hint.innerHTML=`脚本对齐：<b>steps/epoch ≈ ceil(${frames} / ${numEnvs}) = ${spe}</b> · <b>总 steps ≈ ${epochs}×${spe} = ${epochs*spe}</b>（step 只是进度，停准则是 Epochs）`;
  }else{
    hint.textContent='选择数据集后显示预估 steps/epoch 与总 steps。';
  }
}
async function onTrainDatasetChange(){
  // Shortcut <select> only: replace tree selection with that one dataset.
  const selPath=$('trainDataset')?.value||'';
  const ds=(trainTask?.datasets||[]).find(d=>d.path===selPath);
  if(selPath&&ds){
    const usePath=(ds.remote_path&&String(ds.remote_path).startsWith('/mnt'))?ds.remote_path:ds.path;
    if(usePath){
      trainBrowseSelected=new Set([usePath]);
      syncTrainPathBoxFromSelection();
      renderTrainDatasetTree();
    }
    $('trainPrompt').value=resolveTrainPrompt(trainTask, ds, ds?.probe?.prompt);
  }
  await probeTrainSelectedPaths(ds);
}
function isTrainUnifiedPath(p){
  const s=String(p||'').toLowerCase();
  return s.includes('unified')||s.includes('/datasets/830/');
}
function isTrainRawSessionPath(p){
  const s=String(p||'').replace(/\\/g,'/');
  if(!s||isTrainUnifiedPath(s))return false;
  const name=s.split('/').filter(Boolean).pop()||'';
  if(/^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$/.test(name))return true;
  return s.includes('/830demo/')||s.includes('/820demo/');
}
/** Pass through all selected paths. Backend packs raw sessions (valid-only) or uses one unified. */
function resolveTrainJobRefPaths(paths){
  const list=(paths||[]).map(String).filter(Boolean);
  const mix=!!(trainTask?.mix||trainTask?.train_profile==='mix_vision_isaac');
  if(mix||!list.length)return {paths:list, redirected:false, from:null, unified:null, pack:false};
  if(list.some(isTrainRawSessionPath)){
    return {paths:list, redirected:false, from:null, unified:null, pack:true};
  }
  if(list.some(isTrainUnifiedPath))return {paths:list, redirected:false, from:null, unified:null, pack:false};
  return {paths:list, redirected:false, from:list[0], unified:null, pack:false};
}
function suggestTrainUnifiedPath(){
  const cands=[
    trainTask?.ref_root,
    trainTask?.preferred_dataset?.remote_path,
    trainTask?.preferred_dataset?.path,
    ...((trainTask?.datasets||[]).map(d=>d.remote_path||d.path)),
  ].filter(Boolean).map(String);
  return cands.find(p=>isTrainUnifiedPath(p)&&!/830mix|_mix_/i.test(p))||'';
}
function resolvedPackHint(paths){
  return resolveTrainJobRefPaths(paths||[]).pack;
}
async function probeTrainSelectedPaths(ds){
  const paths=getTrainSelectedPaths();
  const path=paths[0]||'';
  if(!path){
    if($('trainDatasetReady')){$('trainDatasetReady').className='scan-preview empty';$('trainDatasetReady').textContent='勾选下方数据集后显示就绪检查'}
    return;
  }
  try{
    const exec=$('trainExecMode')?.value||'cluster_0';
    const x=await post('/api/train/datasets/probe',{dataset_path:path,execution_mode:exec==='local'?'local':'remote',host_id:exec==='local'?'cluster_0':exec});
    $('trainPrompt').value=resolveTrainPrompt(trainTask, ds, x.prompt);
    window._trainProbeFrames=+(x.total_frames||0);
    const checks=Object.entries(x.checks||{}).map(([k,ok])=>`${ok?'✓':'✗'} ${k}`).join(' · ');
    const screened=x.screened?`已筛选 valid ${x.valid_count||0}`:`未导出 valid（请先在 02 导出）`;
    const invTip=x.invalid_count?` · invalid ${x.invalid_count} 已排除`:'';
    const multi=paths.length>1?` · 已选 ${paths.length} 个（全部用于训练）`:'';
    const rawTip=resolvedPackHint(paths)
      ?`<span style="color:#b45309">已选 ${paths.length} 个 raw session：启动后合并 <b>valid</b> 打包到 <b>Phi_0_train_data/&lt;skill&gt;_&lt;时间&gt;</b> → distill；训完保留，原始数据不动。</span>`
      :(!isTrainUnifiedPath(path)
      ?`<span style="color:#b45309">当前路径不是 *_unified。请勾选 raw session（将 pack 到 Phi_0_train_data），或选择一个已有 unified。</span>`
      :'');
    $('trainDatasetReady').className='scan-preview';
    $('trainDatasetReady').innerHTML=`<b>${esc(x.name)}</b><span>${x.train_ready?'可训练':(x.ready?'文件就绪·缺筛选':'未就绪')} · <strong>${esc(screened)}</strong>${invTip}${multi} · 全量 ${x.total_episodes_all||x.total_episodes||0} eps · ${x.total_frames||0} frames · ${esc(x.robot_type||'')}</span><span>${esc(checks)}</span>${rawTip}<span>prompt: ${esc($('trainPrompt').value||'—')}</span>`;
    $('trainStartBtn').disabled=false;
    if($('trainExecMode').value==='local'&&!(x.train_ready||x.ready)&&isTrainUnifiedPath(path))$('trainStartBtn').disabled=true;
    if($('trainExecMode').value==='local'&&x.ready&&!x.screened)$('trainStartBtn').disabled=false;
    updateTrainEffBatch();
  }catch(e){$('trainDatasetReady').className='scan-preview empty';$('trainDatasetReady').textContent=e.message}
}
async function startTrainJob(){
  if(!trainTask)return toast('请先选择任务卡片',true);
  try{
    const exec=$('trainExecMode')?.value||'cluster_0';
    await post('/api/train/config',exec==='local'?{execution_mode:'local',host_id:'cluster_0'}:{execution_mode:'remote',host_id:exec});
    const d=trainTask.defaults||{};
    const epochs=Math.max(1,+$('trainEpochs').value||+(d.epochs||4));
    if(!epochs)return toast('请设置 Epochs',true);
    const mixCompose=isTrainMixCompose(trainTask);
    let mixSlots=null;
    let paths=[];
    let resolved={paths:[], redirected:false, from:null, unified:null, pack:false};
    if(mixCompose){
      mixSlots=collectMixSlotsFromUI();
      const filled=mixSlots.filter(s=>(s.dataset_paths||[]).length>0);
      if(filled.length<2)return toast('混训请至少为 2 个技能各选一个 raw session',true);
      const empty=mixSlots.filter(s=>(s.dataset_paths||[]).length===0);
      if(empty.length)return toast(`请为技能补齐数据集：${empty.map(s=>s.skill_id||s.title).join(', ')}`,true);
      mixSlots=filled;
      paths=mixSlots.flatMap(s=>s.dataset_paths||[]);
      resolved={paths, redirected:false, from:null, unified:null, pack:true};
    }else{
      const selected=getTrainSelectedPaths();
      if(!selected.length)return toast('请先勾选或填写至少一个数据集路径',true);
      resolved=resolveTrainJobRefPaths(selected);
      if(resolved.redirected){
        toast(`已自动改用 unified REF：${resolved.unified.split('/').pop()}`);
        trainBrowseSelected=new Set([resolved.unified]);
        syncTrainPathBoxFromSelection();
        if($('trainDataset')){
          const hit=(trainTask.datasets||[]).find(d=>[d.path,d.remote_path].includes(resolved.unified));
          if(hit)$('trainDataset').value=hit.path;
        }
        void probeTrainSelectedPaths();
      }else if(resolved.pack){
        // raw sessions → backend packs valid-only then runs distill launcher
      }else if(!resolved.paths.some(isTrainUnifiedPath)&&!(trainTask?.mix||trainTask?.train_profile==='mix_vision_isaac')){
        return toast('请勾选 raw session（将按 valid pack），或直接选择一个 *_unified',true);
      }
      paths=resolved.paths;
    }
    const params={
      lr:$('trainLr').value,
      lr_scheduler:d.lr_scheduler||'cosine',
      warmup_ratio:+$('trainWarmupRatio').value,
      warmup_steps:d.warmup_steps??0,
      num_envs:+$('trainNumEnvs').value,
      ngpu:trainNgpuFromCuda(),
      epochs,
      extra_steps:0,train_steps:0,
      horizon:d.horizon??32,ckpt_every_epoch:Math.max(0,+$('trainCkptEveryEpoch')?.value||1),
      // Epoch mode forces step cadence off (UI may still show stale numbers).
      ckpt_every:(Math.max(0,+$('trainCkptEveryEpoch')?.value||1)>0)?0:(+$('trainCkptEvery').value||0),
      ckpt_step_keep:Math.max(0,+$('trainCkptStepKeep')?.value||3),
      hand_mode:$('trainHand').value,deploy_policy:d.deploy_policy||'sonic_v1_1',
      cuda_devices:parseTrainCudaIds($('trainCuda').value).join(','),
      prompt:mixCompose?(mixSlots.map(s=>`${s.skill_id}:${s.prompt}`).join(' | ')):$('trainPrompt').value,
      dataset_paths:paths,
    };
    if(mixCompose){
      params.mix_slots=mixSlots;
      params.mix_compose=true;
    }
    if(!params.ngpu)return toast('请填写 CUDA 设备列表（张数由此决定）',true);
    const hostId=exec==='local'?'cluster_0':exec;
    const body={
      task_id:trainTask.id,
      skill_id:trainTask.skill||trainTask.id,
      dataset_path:paths[0],
      dataset_paths:paths,
      ref_root:paths[0],
      execution_mode:exec==='local'?'local':'remote',
      host_id:hostId,
      train_profile:trainTask.train_profile||'',
      params,
    };
    if(mixCompose){
      body.mix_slots=mixSlots;
      body.mix_compose=true;
    }
    const ds=(trainTask.datasets||[]).find(d=>d.path===$('trainDataset')?.value||d.path===paths[0]||d.remote_path===paths[0]);
    if(ds?.remote_path&&exec!=='local')body.remote_dataset_path=ds.remote_path;
    if(trainTask.valid_json)body.labels_path=trainTask.valid_json;
    if(ds?.labels_path)body.labels_path=ds.labels_path;
    else if(trainTask.labels_path)body.labels_path=trainTask.labels_path;
    if(trainHostHasRunning(hostId)){
      return toast(`节点 ${hostId} 已有训练在跑，请换其他 SSH 或先停止该节点任务`,true);
    }
    $('trainStartBtn').disabled=true;
    const x=await post('/api/train/jobs',body);
    if(x.job?.out_dir&&$('trainOutDir'))$('trainOutDir').value=x.job.out_dir;
    activeTrainJob=x.job.id;state.train_jobs=state.train_jobs||[];state.train_jobs.unshift(x.job);
    clearTrainMonitorView();
    updateTrainMonitorBanner(x.job);
    renderTrainJobs();
    const pack=x.job?.pack;
    if(pack?.mix_slots_mode){
      toast(`已在 ${hostId} 启动混训 · ${pack.slots?.length||mixSlots?.length||0} 技能 / ${pack.session_count||paths.length} session / valid ${pack.valid_count||'?'}`);
    }else if(pack){
      const roots=(pack.raw_parents||[]).map(p=>String(p).split('/').pop()).filter(Boolean);
      const rootTip=roots.length>1?` · 合并目录 ${roots.join('+')}`:'';
      toast(`已在 ${hostId} 启动 · pack ${pack.session_count||paths.length} session / valid ${pack.valid_count||'?'}${rootTip}`);
    }else{
      toast(`已在 ${hostId} 启动 · ${paths.length>1?paths.length+' 个路径':String(paths[0]).split('/').pop()}`);
    }
    await refreshTrainLogs();startTrainPolling();
  }catch(e){toast(e.message,true);renderTrainJobs()}
}
function selectedTrainHostId(){
  const exec=$('trainExecMode')?.value||'cluster_0';
  return exec==='local'?'cluster_0':exec;
}
function trainHostHasRunning(hostId){
  const hid=String(hostId||'');
  return (state.train_jobs||[]).some(j=>j.status==='running'&&String(j.host_id||'')===hid);
}
function updateTrainMonitorBanner(job){
  const ban=$('trainMonitorBanner');
  const logHint=$('trainLogHostHint');
  if(!job){
    if(ban)ban.textContent='未选任务 · 点击任务列表切换节点监控';
    if(logHint)logHint.textContent='点击任务切换对应节点日志';
    if($('tmHostHint'))$('tmHostHint').textContent='监控 host：—';
    return;
  }
  const host=job.host_id||job.execution||'—';
  const title=job.task_title||job.task_id||'';
  const msg=(job.message||'').replace(/\s+/g,' ').slice(0,80);
  if(ban)ban.textContent=`监控中 · ${host} · ${job.id} · ${title} · ${statusLabel(job.status)}${msg?` · ${msg}`:''}`;
  if(logHint)logHint.textContent=`${host} · ${job.id} · ${statusLabel(job.status)}`;
  if($('tmHostHint'))$('tmHostHint').textContent=`监控 host：${host}`;
}
function clearTrainMonitorView(){
  trainChartPoints=[];
  trainChartGeom=null;
  ['tmEpoch','tmStep','tmLoss','tmLossZ','tmLossHand','tmLossZPhys','tmEta','tmSpeed'].forEach(id=>{
    if($(id))$(id).textContent='—';
  });
  const canvas=$('trainChart');
  if(canvas?.getContext){
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
  }
  if($('trainTerminal'))$('trainTerminal').textContent='切换任务中…';
  if($('trainCkptList')){$('trainCkptList').className='ckpt-list empty-state';$('trainCkptList').textContent='—';}
}
async function stopTrainJob(){
  const id=activeTrainJob||(state.train_jobs||[]).find(j=>j.status==='running')?.id;
  if(!id)return toast('请先在任务列表选中要停止的训练',true);
  const job=(state.train_jobs||[]).find(j=>j.id===id);
  if(job&&job.status!=='running')return toast('当前选中任务未在运行',true);
  try{
    // Keep VLM cache / ephemeral pack by default — re-encode is expensive.
    await post(`/api/train/jobs/${id}/cancel`,{keep_vlm_cache:true});
    toast(`已停止 ${job?.host_id||''} · ${id}（已保留 VLM cache）`);
    await loadState();renderTrainJobs();
    if(activeTrainJob){await refreshTrainMetrics();await refreshTrainLogs();await refreshTrainCkpts()}
  }catch(e){toast(e.message,true)}
}
function renderTrainJobs(){
  const el=$('trainJobList'),jobs=state.train_jobs||[];
  if(!el)return;
  if(!jobs.length){el.className='train-job-list empty';el.textContent='暂无训练任务';updateTrainMonitorBanner(null);return}
  el.className='train-job-list';
  el.innerHTML=jobs.slice(0,24).map(j=>{
    const host=esc(j.host_id||j.execution||'—');
    const running=j.status==='running';
    return `<div class="train-job-item${j.id===activeTrainJob?' active':''}" onclick="selectTrainJob('${j.id}')">
      <div class="train-job-item-top"><span class="train-job-host">${host}</span><span class="train-job-status${running?' is-running':''}">${statusLabel(j.status)}</span></div>
      <b>${esc(j.task_title||j.task_id||j.id)}</b>
      <small>${esc(j.id)} · batch ${j.effective_batch||'—'} · ${esc((j.message||'').replace(/\s+/g,' ').slice(0,100))}</small>
    </div>`;
  }).join('');
  const hostId=selectedTrainHostId();
  const hostBusy=trainHostHasRunning(hostId);
  if($('trainStartBtn')){
    $('trainStartBtn').disabled=hostBusy;
    $('trainStartBtn').title=hostBusy
      ?`当前节点 ${hostId} 已有训练在跑，请换其他 SSH 节点或先停止`
      :'在当前所选 SSH 节点启动训练（可与其他节点并行）';
  }
  const active=jobs.find(j=>j.id===activeTrainJob);
  if($('trainStopBtn'))$('trainStopBtn').disabled=!(active&&active.status==='running');
  updateTrainMonitorBanner(active||null);
}
async function selectTrainJob(id){
  if(!id)return;
  activeTrainJob=id;
  clearTrainMonitorView();
  const job=(state.train_jobs||[]).find(j=>j.id===id);
  if(job?.out_dir&&$('trainOutDir'))$('trainOutDir').value=job.out_dir;
  updateTrainMonitorBanner(job||null);
  renderTrainJobs();
  await refreshTrainMetrics();
  await refreshTrainLogs();
  await refreshTrainCkpts();
  startTrainPolling();
}
function fmtEta(sec){
  if(sec==null||!isFinite(sec)||sec<0)return'—';
  sec=Math.round(sec);
  if(sec<60)return sec+'s';
  if(sec<3600)return Math.floor(sec/60)+'m'+String(sec%60).padStart(2,'0')+'s';
  const h=Math.floor(sec/3600),m=Math.floor((sec%3600)/60);
  return h+'h'+String(m).padStart(2,'0')+'m';
}
async function refreshTrainMetrics(){
  if(!activeTrainJob)return;
  try{
    const x=await api(`/api/train/jobs/${activeTrainJob}/metrics`);
    trainChartPoints=x.points||[];const live=x.live||{};
    if($('tmHostHint')){
      const hid=x.host_id||(state.train_jobs||[]).find(j=>j.id===activeTrainJob)?.host_id||'—';
      const via=x.remote_monitor?'SSH 远端':'本机/cluster_0';
      $('tmHostHint').textContent=`监控 host：${hid}（${via}${x.source?` · ${x.source}`:''}）`;
    }
    if($('tmEpoch'))$('tmEpoch').textContent=live.epoch_frac!=null?live.epoch_frac.toFixed(2):(live.epochs??'—');
    $('tmStep').textContent=live.steps_done??'—';
    $('tmLoss').textContent=live.loss!=null?fmtLossAxis(live.loss):'—';
    if($('tmLossZ'))$('tmLossZ').textContent=live.loss_z!=null?fmtLossAxis(live.loss_z):'—';
    if($('tmLossHand'))$('tmLossHand').textContent=live.loss_hand!=null?fmtLossAxis(live.loss_hand):'—';
    if($('tmLossZPhys'))$('tmLossZPhys').textContent=live.loss_z_phys!=null?fmtLossAxis(live.loss_z_phys):'—';
    const etaWall=fmtEta(live.eta_seconds);
    $('tmEta').textContent=live.eta_seconds!=null?etaWall:(live.eta_steps!=null?`${live.eta_steps} steps`:'—');
    if($('tmSpeed'))$('tmSpeed').textContent=live.steps_per_sec!=null?`${Number(live.steps_per_sec).toFixed(2)}/s`:'—';
    window._trainMetricsWaiting=!!x.waiting_current;
    window._trainMetricsMessage=x.message||'';
    renderTrainChartTabs();
    drawTrainChart(trainChartPoints);
  }catch(e){/* quiet while waiting for first metrics */}
}
function renderCkptList(el,ckpts,opts={}){
  if(!el)return;
  if(!ckpts?.length){el.className='ckpt-list empty';el.textContent=opts.empty||'暂无 ckpt';return}
  el.className='ckpt-list';
  el.innerHTML=ckpts.map((c,i)=>{
    const loss=c.loss!=null?` · loss=${Number(c.loss).toFixed(4)}`:'';
    const n=c.episode_count!=null?` · ${c.episode_count} eps`:'';
    return `<div class="ckpt-item"><div><b>${esc(c.name)}</b><small>${esc(c.path)}${loss}${n}</small></div><div class="ckpt-actions"><button type="button" class="ghost" onclick="useCkptForInfer(0,${i})">推理用此</button></div></div>`;
  }).join('');
  el._ckpts=ckpts;
}
async function refreshTrainCkpts(){
  const el=$('trainCkptList');if(!el)return;
  const skillId=trainTask?.skill||trainTask?.id;
  if(activeTrainJob){
    try{
      const x=await api(`/api/train/jobs/${activeTrainJob}/ckpts`);
      renderCkptList(el,x.ckpts||[],{empty:'尚无 phi0_student_*.pt'});
      return;
    }catch(e){/* fall through to skill-bound */}
  }
  if(skillId){
    try{
      const x=await api(`/api/skills/${encodeURIComponent(skillId)}/ckpts`);
      renderCkptList(el,x.ckpts||[],{empty:'该技能卡尚无绑定 ckpt（训练完成后会自动写入）'});
      return;
    }catch(e){el.className='ckpt-list empty';el.textContent=e.message;return}
  }
  el.className='ckpt-list empty';el.textContent='选择训练任务或技能后显示产物';
}
function useCkptForInfer(_probe,i){
  const el=$('trainCkptList');
  const c=el?._ckpts?.[i];if(!c)return;
  const skillId=trainTask?.skill||trainTask?.id||'';
  document.querySelector('.nav[data-page="infer"]').click();
  setTimeout(async()=>{
    await initInferPage();
    if(skillId&&$('inferSkill')){$('inferSkill').value=skillId;onInferSkillChange()}
    if($('inferCkpt'))$('inferCkpt').value=c.path;
    const parts=String(c.path||'').replace(/\/+$/,'').split('/');parts.pop();
    if($('inferCkptDir'))$('inferCkptDir').value=parts.join('/');
    syncInferCkptSelected();
    if(skillId){
      try{await api(`/api/skills/${encodeURIComponent(skillId)}`,{method:'PATCH',body:JSON.stringify({student_ckpt:c.path})})}catch(_){}
    }
    void scanInferCkpts({quiet:true});
    toast(`已绑定技能并填入推理 ckpt：${c.name}`);
  },50);
}
async function refreshTrainLogs(){
  if(!activeTrainJob)return;
  try{const x=await api(`/api/train/jobs/${activeTrainJob}/logs`);const el=$('trainTerminal');el.textContent=x.text||'（日志为空）';el.scrollTop=el.scrollHeight}catch(e){}
}
function fmtLossAxis(v){
  // Always plain decimal — never 1e-2 style.
  if(v==null||!isFinite(v))return '—';
  const a=Math.abs(v);
  if(a>=10)return v.toFixed(2);
  if(a>=1)return v.toFixed(3);
  if(a>=0.01)return v.toFixed(4);
  return v.toFixed(5);
}
function trainLossYWindow(vals,{useFocus=true,focusTop=0.1}={}){
  // Broken Y scale: bottom 1/3 = [ymin, 0.1], top 2/3 = (0.1, ymax] (linear).
  const dataMin=Math.min(...vals);
  const dataMax=Math.max(...vals);
  let ymin,ymax,hasUpper=false;
  if(!(dataMax>dataMin)){
    ymin=0;ymax=useFocus?focusTop:1;
  }else if(!useFocus){
    const span=Math.max(dataMax-dataMin,1e-6);
    ymin=Math.max(0,dataMin-span*0.08);
    ymax=dataMax+span*0.12;
  }else if(dataMin>=focusTop){
    // All above 0.1 — plain auto range (no empty lower band).
    const span=Math.max(dataMax-dataMin,0.01);
    ymin=Math.max(0,dataMin-span*0.08);
    ymax=dataMax+span*0.12;
  }else{
    ymin=Math.max(0,dataMin-Math.max((focusTop-dataMin)*0.04,0.002));
    ymax=Math.max(focusTop,dataMax);
    hasUpper=dataMax>focusTop+1e-9;
    if(hasUpper){
      const up=Math.max(ymax-focusTop,1e-6);
      ymax=focusTop+up*1.06; // small headroom above max
    }
    if(ymax<=ymin){ymin=0;ymax=focusTop;}
  }
  return {ymin,ymax,focusTop,hasUpper,clipped:hasUpper};
}
/** Piecewise-linear Y: [ymin,0.1] → bottom 1/3; (0.1,ymax] → top 2/3. */
function trainLossYAt(v, ymin, ymax, focusTop, y0, y1, useFocus, hasUpper){
  const vv=Math.min(ymax,Math.max(ymin,v));
  const plain=!useFocus||!hasUpper||ymin>=focusTop-1e-12||ymax<=focusTop+1e-12;
  if(plain){
    return y1-(y1-y0)*((vv-ymin)/Math.max(1e-12,ymax-ymin));
  }
  const lowShare=1/3;
  const yFocus=y1-(y1-y0)*lowShare; // 0.1 sits at 1/3 height from bottom
  if(vv<=focusTop){
    const t=(vv-ymin)/Math.max(1e-12,focusTop-ymin);
    return y1-(y1-yFocus)*t;
  }
  const t=Math.min(1,Math.max(0,(vv-focusTop)/Math.max(1e-12,ymax-focusTop)));
  return yFocus-(yFocus-y0)*t;
}
/** Smooth polyline with Catmull-Rom in value space, then map Y (avoids broken-axis wiggles). */
function strokeSmoothMapped(ctx, pts /* {x,v} */, yAt){
  if(!pts.length)return;
  const Y=p=>({x:p.x,y:yAt(p.v)});
  if(pts.length===1){
    const a=Y(pts[0]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(a.x+0.01,a.y);ctx.stroke();return;
  }
  if(pts.length===2){
    const a=Y(pts[0]),b=Y(pts[1]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();return;
  }
  const p0=Y(pts[0]);
  ctx.beginPath();ctx.moveTo(p0.x,p0.y);
  for(let i=0;i<pts.length-1;i++){
    const a=pts[Math.max(0,i-1)],b=pts[i],c=pts[i+1],d=pts[Math.min(pts.length-1,i+2)];
    // Control points in (x,v), then map — keeps shape across the 0.1 break.
    const cp1={x:b.x+(c.x-a.x)/6, v:b.v+(c.v-a.v)/6};
    const cp2={x:c.x-(d.x-b.x)/6, v:c.v-(d.v-b.v)/6};
    const A=Y(cp1),B=Y(cp2),C=Y(c);
    ctx.bezierCurveTo(A.x,A.y,B.x,B.y,C.x,C.y);
  }
  ctx.stroke();
}
function emaTrainSeries(points, key, decay){
  // Heavy EMA for trend only — raw jitter is never drawn.
  const n=points.length;
  // Effective window ≈ 18% of series (clamp 50..220) → α≈2/(span+1).
  const span=Math.min(220,Math.max(50,Math.round(n*0.18)));
  const d=decay!=null?decay:(1-2/(span+1));
  const pass=(src)=>{
    const out=new Array(n).fill(null);
    let ema=null;
    for(let i=0;i<n;i++){
      const raw=src?src[i]:points[i][key];
      if(raw==null||!isFinite(raw)){out[i]=ema;continue}
      const v=Number(raw);
      ema=ema==null?v:d*ema+(1-d)*v;
      out[i]=ema;
    }
    return out;
  };
  // Double-pass EMA further kills residual zigzag while keeping direction.
  const once=pass(null);
  const twice=pass(once);
  return {values:twice,decay:d,span,alpha:1-d};
}
function drawTrainChart(points){
  const c=$('trainChart');if(!c)return;
  const tip=$('trainChartTip');
  const mod=currentTrainChartModule();
  const series=mod.series||[];
  const cssW=c.clientWidth||1400,cssH=520;
  const dpr=Math.min(window.devicePixelRatio||1,2);
  if(c.width!==Math.round(cssW*dpr)||c.height!==Math.round(cssH*dpr)){
    c.width=Math.round(cssW*dpr);c.height=Math.round(cssH*dpr);
  }
  const ctx=c.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  const w=cssW,h=cssH;
  ctx.fillStyle='#0f1512';ctx.fillRect(0,0,w,h);
  if(!points.length){
    trainChartGeom=null;
    if(tip)tip.hidden=true;
    ctx.fillStyle='#6a7a70';ctx.font='14px sans-serif';
    const msg=window._trainMetricsWaiting
      ? (window._trainMetricsMessage || '当前训练尚未进入 distill，不显示旧 out_dir 曲线')
      : '等待当前训练写出 metrics / distill 进度 …';
    ctx.fillText(msg,24,40);
    return;
  }
  const emaByKey={};
  let emaMeta=null;
  for(const s of series){
    const r=emaTrainSeries(points,s.key);
    emaByKey[s.key]=r.values;
    emaMeta=r;
  }
  const vals=series.flatMap(s=>(emaByKey[s.key]||[]).filter(v=>v!=null&&isFinite(v)));
  if(!vals.length){
    trainChartGeom=null;
    ctx.fillStyle='#6a7a70';ctx.font='14px sans-serif';
    ctx.fillText(`模块「${mod.label}」暂无数据（权重为 0 或尚未写入）`,24,40);
    return;
  }
  const useFocus=!!mod.focus;
  const {ymin,ymax,focusTop,hasUpper,clipped}=trainLossYWindow(vals,{useFocus});
  const x0=64,y0=28,x1=w-20,y1=h-36;
  const yAt=v=>trainLossYAt(v,ymin,ymax,focusTop,y0,y1,useFocus,hasUpper);
  ctx.strokeStyle='#2a3530';ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0,y1);ctx.lineTo(x1,y1);ctx.stroke();
  ctx.fillStyle='#5a6a62';ctx.font='11px monospace';
  const tickVals=[];
  if(useFocus&&hasUpper&&ymin<focusTop){
    for(let i=0;i<=3;i++)tickVals.push(ymin+(focusTop-ymin)*(i/3));
    for(let i=1;i<=4;i++)tickVals.push(focusTop+(ymax-focusTop)*(i/4));
  }else{
    for(let i=0;i<=5;i++)tickVals.push(ymin+(ymax-ymin)*(i/5));
  }
  for(const yv of tickVals){
    const y=yAt(yv);
    const upper=useFocus&&hasUpper&&yv>focusTop+1e-9;
    ctx.strokeStyle=upper?'rgba(42,53,48,.4)':'rgba(42,53,48,.7)';
    ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();
    ctx.fillStyle=upper?'#6a7a72':'#5a6a62';
    ctx.fillText(fmtLossAxis(yv),4,y+3);
  }
  const steps=points.map(p=>p.step);const smin=steps[0],smax=steps[steps.length-1]||smin+1;
  const xAt=step=>x0+(x1-x0)*((step-smin)/Math.max(1,smax-smin));
  if(useFocus&&hasUpper&&ymin<focusTop){
    const yFocus=yAt(focusTop);
    ctx.fillStyle='rgba(110,198,255,.045)';
    ctx.fillRect(x0,yFocus,x1-x0,y1-yFocus);
    ctx.fillStyle='rgba(255,180,90,.04)';
    ctx.fillRect(x0,y0,x1-x0,yFocus-y0);
    ctx.strokeStyle='rgba(110,198,255,.4)';ctx.lineWidth=1;ctx.setLineDash([5,4]);
    ctx.beginPath();ctx.moveTo(x0,yFocus);ctx.lineTo(x1,yFocus);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='rgba(110,198,255,.7)';ctx.font='10px monospace';
    ctx.fillText('0.1',x1-22,yFocus-4);
  }
  ctx.lineJoin='round';ctx.lineCap='round';
  // Draw sparsified trend polyline (every ~max(1,n/180) points) for a clean curve.
  const stride=Math.max(1,Math.floor(points.length/180));
  for(const s of series){
    const smoothed=emaByKey[s.key]||[];
    const pts=[];
    for(let i=0;i<points.length;i++){
      if(i!==0&&i!==points.length-1&&(i%stride)!==0)continue;
      const v=smoothed[i];if(v==null||!isFinite(v))continue;
      pts.push({x:xAt(points[i].step),v});
    }
    if(!pts.length)continue;
    ctx.save();
    ctx.strokeStyle=s.color;
    ctx.globalAlpha=0.16;
    ctx.lineWidth=(s.width||2.4)+2.4;
    strokeSmoothMapped(ctx,pts,yAt);
    ctx.globalAlpha=0.96;
    ctx.lineWidth=s.width||2.4;
    strokeSmoothMapped(ctx,pts,yAt);
    ctx.restore();
  }
  ctx.fillStyle='#8fa097';ctx.font='11px monospace';
  let lx=x0;
  ctx.fillText(`step ${smin} → ${smax}`,lx,h-10);lx+=150;
  if(emaMeta){
    ctx.fillText(`EMA 趋势 span≈${emaMeta.span} · α=${emaMeta.alpha.toFixed(3)}`,lx,h-10);
    lx+=210;
  }
  if(useFocus&&hasUpper){
    ctx.fillText('Y: 下1/3=最低→0.1 · 上2/3=0.1→最大',lx,h-10);
    lx+=260;
  }else if(useFocus){
    ctx.fillText('Y: 最低→0.1',lx,h-10);lx+=110;
  }
  for(const s of series){
    const lab=`${s.label} · EMA`;
    ctx.fillStyle='#8fa097';ctx.fillText(lab,lx,h-10);
    ctx.fillStyle=s.color;ctx.globalAlpha=0.9;ctx.fillRect(lx+ctx.measureText(lab).width+6,h-18,10,4);ctx.globalAlpha=1;
    lx+=ctx.measureText(lab).width+28;
  }
  trainChartGeom={points,series,emaByKey,x0,y0,x1,y1,smin,smax,ymin,ymax,xAt,yAt,w,h,focusTop,clipped,hasUpper,useFocus,modId:mod.id};
}
function onTrainChartMove(ev){
  const c=$('trainChart'),tip=$('trainChartTip'),g=trainChartGeom;
  if(!c||!tip||!g?.points?.length){if(tip)tip.hidden=true;return}
  const rect=c.getBoundingClientRect();
  const x=(ev.clientX-rect.left);
  if(x<g.x0||x>g.x1){tip.hidden=true;return}
  const ratio=(x-g.x0)/Math.max(1,g.x1-g.x0);
  const stepTarget=g.smin+ratio*Math.max(1,g.smax-g.smin);
  let bestIdx=0,bestD=Infinity;
  for(let i=0;i<g.points.length;i++){
    const d=Math.abs(g.points[i].step-stepTarget);
    if(d<bestD){bestD=d;bestIdx=i}
  }
  const best=g.points[bestIdx];
  const cx=g.xAt(best.step);
  drawTrainChart(g.points);
  const g2=trainChartGeom||g;
  const ctx=c.getContext('2d');
  const dpr=Math.min(window.devicePixelRatio||1,2);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.strokeStyle='rgba(255,255,255,.28)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(cx,g2.y0);ctx.lineTo(cx,g2.y1);ctx.stroke();ctx.setLineDash([]);
  ctx.lineJoin='round';ctx.lineCap='round';
  for(const s of g2.series){
    const v=g2.emaByKey?.[s.key]?.[bestIdx];
    if(v==null||!isFinite(v))continue;
    const y=g2.yAt(v);
    ctx.fillStyle=s.color;
    ctx.globalAlpha=0.35;ctx.beginPath();ctx.arc(cx,y,7,0,Math.PI*2);ctx.fill();
    ctx.globalAlpha=1;ctx.beginPath();ctx.arc(cx,y,3.5,0,Math.PI*2);ctx.fill();
  }
  const lines=[`<b>step ${best.step}</b>`];
  if(best.epoch!=null)lines.push(`<span>epoch ${Number(best.epoch).toFixed(3)}</span>`);
  for(const s of g2.series){
    const raw=best[s.key];
    if(raw==null||!isFinite(raw))continue;
    lines.push(`<span style="color:${s.color}">${s.label}: ${fmtLossAxis(raw)}</span>`);
  }
  tip.innerHTML=lines.join('');
  tip.hidden=false;
  const tipW=tip.offsetWidth||160,tipH=tip.offsetHeight||80;
  let left=cx+12,top=(ev.clientY-rect.top)+12;
  if(left+tipW>g2.w-8)left=cx-tipW-12;
  if(top+tipH>g2.h-8)top=g2.h-tipH-8;
  tip.style.left=`${Math.max(8,left)}px`;
  tip.style.top=`${Math.max(8,top)}px`;
}
function onTrainChartLeave(){
  const tip=$('trainChartTip');if(tip)tip.hidden=true;
  if(trainChartGeom?.points)drawTrainChart(trainChartGeom.points);
}
document.getElementById('trainChart')?.addEventListener('mousemove',onTrainChartMove);
document.getElementById('trainChart')?.addEventListener('mouseleave',onTrainChartLeave);
window.addEventListener('resize',()=>{if(trainChartPoints?.length)drawTrainChart(trainChartPoints)});
renderTrainChartTabs();
function startTrainPolling(){
  if(trainPollTimer)return;
  trainPollTimer=setInterval(async()=>{
    if(!$('page-train')?.classList.contains('active'))return;
    try{const st=await api('/api/state');state.train_jobs=st.train_jobs||[];renderTrainJobs()}catch(_){}
    if(activeTrainJob){await refreshTrainMetrics();await refreshTrainLogs();await refreshTrainCkpts()}
  },5000);
}
document.getElementById('trainExecMode')?.addEventListener('change',()=>{
  updateTrainHostHint();
  renderTrainJobs();
  const v=$('trainExecMode').value;
  const payload=v==='local'
    ?{execution_mode:'local',host_id:'cluster_0'}
    :{execution_mode:'remote',host_id:v};
  void post('/api/train/config',payload).catch(()=>{});
});

/* ===== Infer (06) MuJoCo CL ===== */
let inferCatalog=null,activeInferJob=null,inferPollTimer=null,inferVideoLoopTimer=null,inferPauseS=1,inferMonTimer=null,inferCkptFiles=[];
let inferProcLogTab='sim';
let inferProcLogs={sim:'',policy:'',deploy:''};
let inferCkptPull=null;
let inferSkillEpMap={},inferSkillEpList={}; // skillId -> ep / episode list
async function initInferPage(){
  try{
    const x=await api('/api/infer/catalog');
    inferCatalog=x.catalog;state.infer_jobs=x.jobs||[];
    const d=inferCatalog.defaults||{};
    const cfg=x.config||{};
    if($('inferHorizon'))$('inferHorizon').value=d.horizon??32;
    if($('inferRtcDelay'))$('inferRtcDelay').value=d.rtc_inference_delay??6;
    if($('inferRtcExec'))$('inferRtcExec').value=d.rtc_execution_horizon??26;
    if($('inferCuda'))$('inferCuda').value=d.cuda_devices||'4';
    if($('inferUseRtc'))$('inferUseRtc').value=(d.use_rtc===false||d.use_rtc==='0')?'0':'1';
    if($('inferHandObs'))$('inferHandObs').value=d.hand_obs||'commanded';
    if($('inferVlmSrc'))$('inferVlmSrc').value=d.vlm_source||'frame_cache';
    if($('inferMonHost'))$('inferMonHost').value=cfg.monitor_host||'127.0.0.1';
    if($('inferCkptDir'))$('inferCkptDir').value=cfg.last_ckpt_dir||cfg.remote_ckpt_base||'/mnt/data2/wpy/workspace/Phi_0_model_zoo';
    inferPauseS=+(d.video_loop_pause_s||1);
    const skills=inferCatalog.skills||[];
    // Don't auto-pick a skill — keep top row as clean cards until user confirms.
    if($('inferSkill').value)onInferSkillChange();
    else{updateInferSkillTools();renderInferSkillCards()}
    if(!skills.length&&$('inferCkptDir')?.value)void scanInferCkpts({quiet:true});
    else if(!$('inferSkill').value)renderInferSkillCards();
    renderInferJobs();
    const running=(state.infer_jobs||[]).find(j=>j.status==='running');
    if(running){activeInferJob=running.id;await refreshInferLogs();attachInferMjpeg(running.id);await refreshInferPreview();await refreshInferMp4()}
    if($('inferExecMode'))$('inferExecMode').value='remote';
    void refreshInferDisplayHint();
    startInferPolling();
    await refreshInferMonitor();
    startInferMonPolling();
  }catch(e){toast(e.message,true)}
}
function inferSkillList(){return inferCatalog?.skills||[]}
function skillEp(skillId){
  if(skillId&&inferSkillEpMap[skillId]!=null)return +inferSkillEpMap[skillId];
  const s=inferSkillList().find(x=>x.id===skillId);
  return +(s?.ep??0);
}
function setSkillEp(skillId,ep){
  if(skillId)inferSkillEpMap[skillId]=+ep;
  if(skillId===$('inferSkill')?.value)$('inferEp').value=String(+ep);
}
function renderInferSkillCards(){
  const skills=inferSkillList();
  const cur=$('inferSkill')?.value||'';
  const mk=(mode)=>{
    if(!skills.length)return `<div class="empty-hint">暂无通用技能卡 — 请先在 01 上传创建</div>`;
    return skills.map((s,i)=>{
      const n=i+1;
      const sel=s.id===cur?' selected':'';
      const click=mode==='switch'
        ?`applyInferSkillSwitch('${esc(s.id)}')`
        :`selectInferSkill('${esc(s.id)}')`;
      // Cards stay clean: badge + title + prompt only (video/ckpt live outside).
      return `<button type="button" class="infer-skill-card${sel}" data-skill="${esc(s.id)}" data-key="${n}" onclick="${click}"><span class="infer-skill-key">${n}</span><span class="badge">${esc(s.badge||s.id)}</span><h3>${esc(s.title||s.id)}</h3><p>${esc(s.prompt||'')}</p></button>`;
    }).join('');
  };
  const selEl=$('inferSkillCards');
  if(selEl){selEl.className='infer-skill-card-grid';selEl.innerHTML=mk('select')+`<button type="button" class="infer-skill-card infer-skill-card-custom${cur?'':' selected'}" onclick="selectInferSkill('')"><span class="badge">CUSTOM</span><h3>自定义</h3><p>手动填 ckpt / REF_ROOT</p></button>`}
  const swEl=$('inferSwitchCards');
  if(swEl){swEl.className='infer-skill-card-grid';swEl.innerHTML=mk('switch')}
  updateInferSkillTools();
}
function updateInferSkillTools(){
  const tools=$('inferSkillTools');if(!tools)return;
  const id=$('inferSkill')?.value||'';
  const s=id?inferSkillList().find(x=>x.id===id):null;
  if(!s){tools.hidden=true;return}
  tools.hidden=false;
  if($('inferToolsSkillBadge'))$('inferToolsSkillBadge').textContent=s.badge||s.id;
  if($('inferToolsSkillTitle'))$('inferToolsSkillTitle').textContent=s.title||s.id;
  const ep=skillEp(s.id);
  const eps=inferSkillEpList[s.id];
  if($('inferToolsEpLabel'))$('inferToolsEpLabel').textContent=eps?.length?`回放 ep${ep} / ${eps.length}`:`回放 ep${ep}`;
}
async function ensureInferEpisodes(skill){
  if(!skill)return[];
  // 04 默认全远端：只探 cluster REF，不用 local_ref_root。
  const ref=skill.ref_root||'';
  if(!ref)return[];
  try{
    const x=await post('/api/infer/datasets/probe',{
      ref_root:ref,
      ep:skillEp(skill.id),
      local_ref_root:'',
      host_id:'cluster_0',
    });
    const eps=Array.isArray(x.episodes)?x.episodes:[];
    if(!eps.length){
      toast('该训练集暂无回放视频（远端未找到 mp4）',true);
    }else{
      toast(`远端训练集 ${eps.length} 个回放 episode`,false);
    }
    inferSkillEpList[skill.id]=eps.length?eps:[];
    const cur=skillEp(skill.id);
    if(eps.length&&!eps.includes(cur))setSkillEp(skill.id,eps[0]);
    else if(inferSkillEpMap[skill.id]==null)setSkillEp(skill.id,skill.ep??eps[0]??0);
    return inferSkillEpList[skill.id];
  }catch(_){
    inferSkillEpList[skill.id]=[];
    return [];
  }
}
async function cycleInferReplayVideo(skillId){
  const skill=inferSkillList().find(x=>x.id===skillId);
  if(!skill)return toast('未知技能',true);
  const eps=await ensureInferEpisodes(skill);
  if(!eps.length)return toast('该技能暂无回放视频',true);
  const cur=skillEp(skillId);
  let idx=eps.indexOf(cur);
  if(idx<0)idx=0;
  const next=eps[(idx+1)%eps.length];
  setSkillEp(skillId,next);
  if(skillId!==$('inferSkill')?.value){
    selectInferSkill(skillId,{quiet:true});
  }else{
    $('inferEp').value=String(next);
    void refreshInferVideos();
  }
  updateInferSkillTools();
  renderInferSkillCards();
  toast(`已更换回放 → ep${next}`);
}
function cycleCurrentInferReplayVideo(){
  const id=$('inferSkill')?.value||'';
  if(!id)return toast('请先选择技能卡',true);
  return cycleInferReplayVideo(id);
}
function selectInferSkill(id,opts={}){
  if($('inferSkill'))$('inferSkill').value=id||'';
  onInferSkillChange();
  renderInferSkillCards();
  if(!opts.quiet&&id){
    const s=inferSkillList().find(x=>x.id===id);
    if(s)toast(`已确认技能：${s.title||s.id}`);
  }
}
function toggleInferSwitchPanel(force){
  const panel=$('inferSwitchPanel');if(!panel)return;
  const open=force==null?panel.hidden:!!force;
  panel.hidden=!open;
  if($('inferSwitchBtn'))$('inferSwitchBtn').classList.toggle('btn-next',open);
  if(open)renderInferSkillCards();
}
async function applyInferSkillSwitch(id){
  selectInferSkill(id,{quiet:true});
  toggleInferSwitchPanel(false);
  await switchInferSkill();
}
async function switchInferSkill(){
  const running=(state.infer_jobs||[]).find(j=>j.status==='running');
  const hadJob=!!(running||activeInferJob);
  if(running){
    if(!confirm('将终止当前推理，并用新技能的 ckpt / 回放视频重新启动，继续？'))return;
    try{await post(`/api/infer/jobs/${running.id}/cancel`)}catch(_){}
  }else if(hadJob){
    if(!confirm('用当前选中技能（新 ckpt / 回放）重新启动推理？'))return;
  }else{
    toast('已应用技能参数；点 Init Sim 开始');
    return;
  }
  try{
    const skillId=$('inferSkill').value;
    const body={
      skill_id:skillId||undefined,
      student_ckpt:$('inferCkpt').value.trim(),
      ref_root:$('inferRefRoot').value.trim(),
      ep:+$('inferEp').value||0,
      params:inferParams(),
      action:'switch',
      interactive:true,
      force:true,
      execution_mode:'remote',
      host_id:'cluster_0',
      viewer_interactive:true,
      open_terminal:true,
    };
    const skill=(inferCatalog?.skills||[]).find(x=>x.id===skillId);
    if(skill?.valid_json)body.labels_path=skill.valid_json;
    if($('inferOutDir').value.trim())body.out_dir=$('inferOutDir').value.trim();
    if(!body.student_ckpt||!body.ref_root)return toast('需要 ckpt 与 REF_ROOT',true);
    const x=await post('/api/infer/jobs',body);
    activeInferJob=x.job.id;state.infer_jobs=state.infer_jobs||[];state.infer_jobs.unshift(x.job);
    renderInferJobs();startInferDualFromJob(x.job);toast('已在 cluster_0 切换技能并重启');await refreshInferLogs();
    attachInferMjpeg(x.job.id);
    $('inferStopBtn').disabled=false;
  }catch(e){toast(e.message,true)}
}
function onInferSkillChange(){
  const id=$('inferSkill').value;const s=(inferCatalog?.skills||[]).find(x=>x.id===id);
  if(!s){
    if($('inferSkillMeta')){$('inferSkillMeta').className='infer-hero-card empty';$('inferSkillMeta').textContent='手动选择权重目录，或在高级里填 ckpt / REF_ROOT'}
    if($('inferVideoProbe'))$('inferVideoProbe').textContent='选择技能后显示';
    updateInferSkillTools();
    renderInferSkillCards();
    return;
  }
  const ckpt=s.student_ckpt||'';
  $('inferCkpt').value=ckpt;
  // 04 全远端：始终用 cluster REF，忽略 local_ref_root。
  $('inferRefRoot').value=s.ref_root||'';
  $('inferPrompt').value=s.prompt||'';
  if($('inferHandObs'))$('inferHandObs').value=s.hand_obs||inferCatalog?.defaults?.hand_obs||'commanded';
  const ep=skillEp(s.id);
  $('inferEp').value=String(ep);
  // Ckpt dir: skill train_out_dir → parent of student_ckpt → last_ckpt_dir
  if($('inferCkptDir')){
    if(s.train_out_dir)$('inferCkptDir').value=s.train_out_dir;
    else if(ckpt){
      const parts=ckpt.replace(/\/+$/,'').split('/');
      parts.pop();
      $('inferCkptDir').value=parts.join('/')||$('inferCkptDir').value;
    }
  }
  syncInferCkptSelected();
  if($('inferSkillMeta')){
    $('inferSkillMeta').className='infer-hero-card';
    const ckptTail=(ckpt||'').split('/').slice(-2).join('/')||'—';
    const refTail=($('inferRefRoot').value||'').split('/').slice(-2).join('/')||'—';
    const nCkpt=(s.ckpts||[]).length;
    $('inferSkillMeta').innerHTML=`<div class="infer-hero-eyebrow">当前技能</div><h3>${esc(s.title||s.id)}</h3><div class="infer-hero-meta"><span>ckpt …/${esc(ckptTail)}</span><span>ref …/${esc(refTail)}</span><span>回放 ep${ep}</span>${nCkpt?`<span>${nCkpt} 个绑定 ckpt</span>`:''}</div>`;
  }
  updateInferSkillTools();
  void (async()=>{
    await ensureInferEpisodes(s);
    updateInferSkillTools();
    renderInferSkillCards();
    await refreshInferVideos();
    await loadSkillBoundCkpts(id);
  })();
  renderInferSkillCards();
}
async function loadSkillBoundCkpts(skillId){
  if(!skillId)return;
  try{
    const x=await api(`/api/skills/${encodeURIComponent(skillId)}/ckpts`);
    if(x.train_out_dir&&$('inferCkptDir')&&!$('inferCkptDir').value.trim())
      $('inferCkptDir').value=x.train_out_dir;
    if(x.train_out_dir&&$('inferCkptDir'))$('inferCkptDir').value=x.train_out_dir;
    if(x.ckpts?.length){
      inferCkptFiles=x.ckpts.map(c=>({
        path:c.path,name:c.name||String(c.path||'').split('/').pop(),
        rel:c.name||c.path,remote:String(c.path||'').startsWith('/mnt'),size:c.size||0
      }));
      renderInferCkptList();
    }else if($('inferCkptDir')?.value){
      void scanInferCkpts({quiet:true});
    }
    if(x.student_ckpt&&$('inferCkpt')&&!$('inferCkpt').value.trim())
      $('inferCkpt').value=x.student_ckpt;
    if(x.ref_root&&$('inferRefRoot')&&!$('inferRefRoot').value.trim())
      $('inferRefRoot').value=x.ref_root;
    syncInferCkptSelected();
  }catch(_){
    if($('inferCkptDir')?.value)void scanInferCkpts({quiet:true});
  }
}
function syncInferCkptSelected(){
  const path=$('inferCkpt')?.value?.trim()||'';
  const el=$('inferCkptSelected');
  if(!el)return;
  if(!path){el.className='ckpt-selected';el.textContent='未选择权重';return}
  el.className='ckpt-selected has-ckpt';
  el.innerHTML=`已选：<code>${esc(path)}</code>`;
  const list=$('inferCkptList');
  if(list){
    list.querySelectorAll('.ckpt-item').forEach(n=>{
      n.classList.toggle('selected',n.dataset.path===path);
    });
  }
}
async function scanInferCkpts(opts={}){
  const dir=$('inferCkptDir')?.value?.trim();
  const el=$('inferCkptList');
  if(!dir){if(!opts.quiet)toast('请先填写云端权重目录',true);return}
  if(el){el.className='ckpt-list empty';el.textContent='扫描中…'}
  try{
    const x=await api(`/api/infer/ckpts/scan?source=remote&dir=${encodeURIComponent(dir)}`);
    inferCkptFiles=x.files||[];
    renderInferCkptList();
    if(!opts.quiet)toast(`云端找到 ${inferCkptFiles.length} 个 .pt`);
  }catch(e){
    inferCkptFiles=[];
    if(el){el.className='ckpt-list empty';el.textContent=e.message}
    if(!opts.quiet)toast(e.message,true);
  }
}
function renderInferCkptList(){
  const el=$('inferCkptList');if(!el)return;
  if(!inferCkptFiles.length){el.className='ckpt-list empty';el.textContent='该目录下未找到 .pt';return}
  const cur=$('inferCkpt')?.value?.trim()||'';
  el.className='ckpt-list';
  el.innerHTML=inferCkptFiles.map((f,i)=>{
    const sel=f.path===cur||f.local_path===cur?' selected':'';
    const rel=f.rel||f.name;
    const src=f.remote?'云端':'本地';
    const pulling=inferCkptPull&&inferCkptPull.path===f.path;
    const busy=pulling?' is-pulling':'';
    const bar=pulling?ckptPullProgressHtml(inferCkptPull):'';
    const btnLabel=pulling?(inferCkptPull.status==='done'?'已完成':'下拉中…'):'下拉并选用';
    return `<div class="ckpt-item${sel}${busy}" data-path="${esc(f.path)}" data-i="${i}">
      <div class="ckpt-item-main">
        <b>${esc(rel)}</b>
        <small>${src} · ${esc(f.path)} · ${fmtBytes(f.size||0)}</small>
        ${bar}
      </div>
      <div class="ckpt-actions">
        <button type="button" class="ghost" ${pulling?'disabled':''} onclick="event.stopPropagation();pullAndSelectInferCkpt(${i})">${btnLabel}</button>
      </div>
    </div>`;
  }).join('');
  el.querySelectorAll('.ckpt-item').forEach(node=>{
    node.addEventListener('click',e=>{
      if(e.target.closest('button'))return;
      const i=+(node.dataset.i||'-1');
      if(i>=0)void pullAndSelectInferCkpt(i);
    });
  });
  syncInferCkptSelected();
}
function ckptPullProgressHtml(p){
  if(!p)return '';
  const pct=Number(p.pct);
  const known=Number.isFinite(pct)&&pct>=0;
  const width=known?Math.max(0,Math.min(100,pct)):0;
  const done=Number(p.bytes_done||0),total=Number(p.bytes_total||0);
  const label=p.status==='error'
    ?`失败：${esc(p.error||p.message||'')}`
    :(known
      ?`${Math.round(width)}% · ${fmtBytes(done)}${total?` / ${fmtBytes(total)}`:''}${p.speed?` · ${esc(p.speed)}`:''}${p.eta?` · ETA ${esc(p.eta)}`:''}`
      :`下拉中…${total?` · 共 ${fmtBytes(total)}`:''}`);
  return `<div class="upload-progress ckpt-pull-progress${known?'':' is-indeterminate'}${p.status==='error'?' is-error':''}">
    <div class="upload-progress-track"><i style="width:${width}%"></i></div>
    <small>${label}</small>
  </div>`;
}
function updateInferCkptPullBar(){
  const el=$('inferCkptList');if(!el||!inferCkptPull)return;
  const item=[...el.querySelectorAll('.ckpt-item')].find(n=>n.dataset.path===inferCkptPull.path);
  if(!item){renderInferCkptList();return}
  const main=item.querySelector('.ckpt-item-main');
  if(!main){renderInferCkptList();return}
  const html=ckptPullProgressHtml(inferCkptPull);
  const bar=main.querySelector('.ckpt-pull-progress');
  if(bar)bar.outerHTML=html;
  else main.insertAdjacentHTML('beforeend',html);
  const btn=item.querySelector('button');
  if(btn){
    btn.disabled=inferCkptPull.status==='running';
    btn.textContent=inferCkptPull.status==='running'?'下拉中…':(inferCkptPull.status==='done'?'已完成':'下拉并选用');
  }
}
async function finishInferCkptSelect(f, localPath){
  if($('inferCkpt'))$('inferCkpt').value=localPath;
  f.local_path=localPath;
  const skillId=$('inferSkill')?.value;
  if(skillId){
    try{
      await api(`/api/skills/${encodeURIComponent(skillId)}`,{method:'PATCH',body:JSON.stringify({
        student_ckpt:f.path,
        train_out_dir:$('inferCkptDir')?.value?.trim()||undefined,
      })});
    }catch(_){}
  }
  syncInferCkptSelected();
  renderInferCkptList();
  toast(`已下拉并绑定到技能：${f.name}`);
}
async function pullAndSelectInferCkpt(i){
  const f=inferCkptFiles[i];if(!f)return;
  if(inferCkptPull&&inferCkptPull.status==='running')return toast('已有权重正在下拉',true);
  try{
    inferCkptPull={path:f.path,status:'running',pct:0,bytes_done:0,bytes_total:f.size||0,speed:'',eta:'',message:'开始下拉…'};
    renderInferCkptList();
    const x=await post('/api/infer/ckpts/pull',{path:f.path,size:f.size||0});
    if(x.cached||x.status==='done'){
      inferCkptPull={...inferCkptPull,status:'done',pct:100,bytes_done:x.size||f.size||0};
      updateInferCkptPullBar();
      await finishInferCkptSelect(f, x.local_path);
      inferCkptPull=null;
      return;
    }
    const jobId=x.job_id;
    if(!jobId)throw new Error('未返回下拉任务 id');
    // Poll progress
    for(;;){
      await new Promise(r=>setTimeout(r,400));
      const st=await api(`/api/infer/ckpts/pull/${encodeURIComponent(jobId)}`);
      inferCkptPull={
        path:f.path,
        status:st.status||'running',
        pct:st.pct??0,
        bytes_done:st.bytes_done||0,
        bytes_total:st.bytes_total||f.size||0,
        speed:st.speed||'',
        eta:st.eta||'',
        message:st.message||'',
        error:st.error||'',
      };
      updateInferCkptPullBar();
      if(st.status==='done'){
        await finishInferCkptSelect(f, st.local_path||x.local_path);
        inferCkptPull=null;
        return;
      }
      if(st.status==='error')throw new Error(st.error||st.message||'下拉失败');
    }
  }catch(e){
    if(inferCkptPull){inferCkptPull.status='error';inferCkptPull.error=e.message;updateInferCkptPullBar()}
    toast(e.message,true);
    setTimeout(()=>{inferCkptPull=null;renderInferCkptList()},1800);
  }
}
function selectInferCkpt(i){
  void pullAndSelectInferCkpt(i);
}
async function refreshInferVideos(){
  const ref=$('inferRefRoot').value.trim(),ep=+$('inferEp').value||0;
  const skill=(inferCatalog?.skills||[]).find(x=>x.id===$('inferSkill').value);
  if(!ref){if($('inferVideoProbe'))$('inferVideoProbe').textContent='选择技能后显示';return}
  try{
    // 04 默认全远端：不探本地、不阻塞等本机缓存；预览走 /api/infer/video 远端流。
    const x=await post('/api/infer/datasets/probe',{
      ref_root:ref,ep,
      local_ref_root:'',
      host_id:'cluster_0',
    });
    if(skill?.id&&Array.isArray(x.episodes))inferSkillEpList[skill.id]=x.episodes;
    const ego=x.paths?.ego||x.remote_paths?.ego,wrist=x.paths?.wrist||x.remote_paths?.wrist;
    if($('inferVideoProbe')){
      const n=x.episode_count||x.episodes?.length||0;
      const tag=x.screened?`valid ${n}`:(x.allowlist_source==='vision_allowlist'?`打包 allowlist ${n}`:`训练集 ${n}`);
      const tip=x.invalid_tip?` · ${x.invalid_tip}`:'';
      $('inferVideoProbe').textContent=`ep${ep} · ${tag}${tip} · 远端预览 · ego ${x.exists?.ego?'✓':'✗'} · wrist ${x.exists?.wrist?'✓':'✗'}`;
    }
    setInferCacheBar(false);
    const paths=[ego,wrist].filter(Boolean);
    // 后台预热缓存（不阻塞 UI）；播放仍用远端 path，接口侧有缓存会自动加速。
    if(paths.length){
      void post('/api/infer/videos/cache',{paths,host_id:'cluster_0'}).catch(()=>{});
    }
    const bust=`&t=${Date.now()}`;
    const egoUrl=ego?`/api/infer/video?path=${encodeURIComponent(ego)}${bust}`:'';
    const wristUrl=wrist?`/api/infer/video?path=${encodeURIComponent(wrist)}${bust}`:'';
    setupInferDualLoop(egoUrl,wristUrl);
    renderInferSkillCards();
  }catch(e){if($('inferVideoProbe'))$('inferVideoProbe').textContent=e.message}
}
function setInferCacheBar(show,ready,text){
  const bar=$('inferCacheBar'),fill=$('inferCacheFill'),lab=$('inferCacheText');
  if(!bar)return;
  bar.hidden=!show;
  bar.classList.toggle('ready',!!ready);
  if(lab&&text)lab.textContent=text;
}
async function waitInferVideoCache(paths,opts={}){
  const timeoutMs=opts.timeoutMs||90000;
  const t0=Date.now();
  while(Date.now()-t0<timeoutMs){
    const q=paths.map(p=>'path='+encodeURIComponent(p)).join('&');
    const x=await api('/api/infer/videos/cache?'+q);
    const items=x.items||[];
    const ready=items.length&&items.every(it=>it.ready||it.status==='done');
    const failed=items.some(it=>it.status==='error');
    if(ready)return items;
    if(failed)throw new Error(items.find(it=>it.status==='error')?.message||'缓存失败');
    const n=items.filter(it=>it.ready).length;
    setInferCacheBar(true,false,`缓存中 ${n}/${items.length||paths.length}…`);
    await new Promise(r=>setTimeout(r,500));
  }
  throw new Error('缓存超时');
}
function setupInferDualLoop(egoUrl,wristUrl){
  const ego=$('inferEgoVideo'),wrist=$('inferWristVideo');
  if(inferVideoLoopTimer){clearTimeout(inferVideoLoopTimer);inferVideoLoopTimer=null}
  const bind=(v,url)=>{
    if(!v)return;
    if(!url){v.removeAttribute('src');v.load();v.dataset.url='';delete v.dataset.base;return}
    const base=url.split('&t=')[0];
    if(v.dataset.base===base&&v.dataset.url===url&&v.readyState>=2){void v.play().catch(()=>{});return}
    v.dataset.base=base;
    v.dataset.url=url;
    v.preload='auto';
    v.muted=true;
    v.playsInline=true;
    v.src=url;
    v.onerror=()=>{if($('inferVideoProbe'))$('inferVideoProbe').textContent='视频加载失败（可点更换回放或刷新）'};
    v.load();
  };
  bind(ego,egoUrl);bind(wrist,wristUrl);
  const pauseMs=Math.max(0,inferPauseS)*1000;
  const onEnded=()=>{
    ego.pause();wrist.pause();
    inferVideoLoopTimer=setTimeout(()=>{
      try{ego.currentTime=0;wrist.currentTime=0}catch(_){}
      void ego.play().catch(()=>{});void wrist.play().catch(()=>{});
    },pauseMs);
  };
  ego.onended=onEnded;wrist.onended=null;
  const kick=()=>{void ego.play().catch(()=>{});void wrist.play().catch(()=>{})};
  if(ego.readyState>=3)kick();else ego.addEventListener('canplay',kick,{once:true});
  const sync=()=>{
    if(!wrist||wrist.paused&&ego.paused)return;
    if(Math.abs((ego.currentTime||0)-(wrist.currentTime||0))>0.25){
      try{wrist.currentTime=ego.currentTime}catch(_){}
    }
  };
  ego.ontimeupdate=sync;
}
function inferParams(){
  return{
    prompt:$('inferPrompt').value,
    horizon:+($('inferHorizon')?.value||32),
    use_rtc:($('inferUseRtc')?.value||'1')==='1',
    rtc_inference_delay:+($('inferRtcDelay')?.value||6),
    rtc_execution_horizon:+($('inferRtcExec')?.value||26),
    hand_obs:$('inferHandObs')?.value||'commanded',
    vlm_source:$('inferVlmSrc')?.value||'frame_cache',
    cuda_devices:$('inferCuda')?.value||'4'
  };
}
async function inferAction(action){
  try{
    const skillId=$('inferSkill').value;
    if($('inferExecMode'))$('inferExecMode').value='remote';
    const body={
      skill_id:skillId||undefined,
      student_ckpt:$('inferCkpt').value.trim(),
      ref_root:$('inferRefRoot').value.trim(),
      ep:+$('inferEp').value||0,
      params:inferParams(),
      action,
      interactive:action!=='start',
      force:false,
      execution_mode:'remote',
      host_id:'cluster_0',
      viewer_interactive:true,
      open_terminal:true,
    };
    const skill=(inferCatalog?.skills||[]).find(x=>x.id===skillId);
    if(skill?.valid_json)body.labels_path=skill.valid_json;
    if($('inferOutDir')?.value.trim())body.out_dir=$('inferOutDir').value.trim();
    if(!body.student_ckpt||!body.ref_root)return toast('需要 ckpt 与 REF_ROOT（选技能或展开高级填写）',true);
    $('inferInitBtn').disabled=true;if($('inferStartBtn'))$('inferStartBtn').disabled=true;$('inferStopBtn').disabled=false;
    const x=await post('/api/infer/jobs',body);
    activeInferJob=x.job.id;state.infer_jobs=state.infer_jobs||[];state.infer_jobs.unshift(x.job);
    renderInferJobs();
    toast(action==='start'
      ?'已在 cluster_0 启动全自动闭环'
      :'已启动：本机将弹 MuJoCo 窗 + 日志终端；Sim 就绪后 Deploy 按 y');
    if(x.terminal&&x.terminal.ok===false)toast('桌面终端：'+(x.terminal.message||'打开失败'),true);
    else if(x.terminal&&x.terminal.ok)toast(`已打开桌面终端（${x.terminal.launcher||'terminal'}）`);
    startInferDualFromJob(x.job);await refreshInferLogs();startInferPolling();
    void attachInferMjpeg(x.job.id);
  }catch(e){toast(e.message,true);$('inferInitBtn').disabled=false;if($('inferStartBtn'))$('inferStartBtn').disabled=false}
}
async function openInferMujocoTerminal(){
  try{
    const body={host_id:'cluster_0'};
    if(activeInferJob)body.job_id=activeInferJob;
    const x=await post('/api/infer/open-mujoco-terminal',body);
    toast(x.message||'已打开桌面终端');
  }catch(e){toast(e.message,true)}
}
async function refreshInferDisplayHint(){
  const el=$('inferDisplayHint');
  if(!el)return;
  try{
    const x=await api('/api/infer/display-status');
    if(!x.ready){
      el.innerHTML='当前无 DISPLAY：请在<strong>跑 Studio 的图形桌面</strong>启动服务，或 <code>export DISPLAY=:0</code> 后重启。点「打开桌面终端」需本机有 gnome-terminal/xterm。';
    }else{
      el.innerHTML=`一键：本机桌面弹出可拖动 <b>MuJoCo</b>（ssh -Y · DISPLAY=${esc(x.display)}）并开日志终端；网页只控 Deploy（y → ] → Enter）。浏览器无法嵌真实 MuJoCo。`;
    }
  }catch(_){}
}
async function inferCmd(cmd){
  if(!activeInferJob)return toast('请先初始化 Sim',true);
  try{
    await post(`/api/infer/jobs/${activeInferJob}/cmd`,{cmd});
    const labels={deploy:'已发送 Deploy',stand:'已发送 ] 站立',policy:'已发送 Policy',stream:'已发送 Enter',stream_p:'已发送 P + Enter'};
    toast(labels[cmd]||`已发送 ${cmd}`);
    await refreshInferLogs();
  }catch(e){toast(e.message,true)}
}
function mapInferTtyToCmd(raw){
  const t=String(raw||'').trim();
  if(!t)return 'stream';
  const low=t.toLowerCase();
  if(low==='y')return 'deploy';
  if(t===']'||low==='stand')return 'stand';
  if(low==='enter'||low==='stream'||t==='')return 'stream';
  if(low==='p'||low==='stream_p')return 'stream_p';
  if(low==='policy'||low==='arm')return 'policy';
  if(low==='deploy'||low==='go')return 'deploy';
  if(low.startsWith('key:'))return t;
  // single char → raw deploy key
  if(t.length===1)return `key:${t}`;
  return t;
}
async function sendInferTtyKey(key){
  await inferCmd(mapInferTtyToCmd(key));
}
async function submitInferTty(){
  const el=$('inferTtyInput');if(!el)return;
  const raw=el.value;
  el.value='';
  await inferCmd(mapInferTtyToCmd(raw));
}
async function startInferMonitor(){
  try{
    const host=$('inferMonHost').value.trim()||'127.0.0.1';
    await post('/api/infer/config',{monitor_host:host});
    await post('/api/infer/monitor/start',{monitor_host:host});
    toast('Monitor 已启动');
    await refreshInferMonitor();
    startInferMonPolling();
  }catch(e){toast(e.message,true)}
}
async function stopInferMonitor(){
  try{await post('/api/infer/monitor/stop');toast('Monitor 已停止');await refreshInferMonitor()}catch(e){toast(e.message,true)}
}
async function refreshInferMonitor(){
  try{
    const x=await api('/api/infer/monitor/status');
    const badge=$('inferMonBadge');
    if(badge){badge.textContent=x.alive?'Monitor 运行中':'未启动';badge.classList.toggle('live',!!x.alive)}
    if($('inferMonStart'))$('inferMonStart').disabled=!!x.alive;
    if($('inferMonStop'))$('inferMonStop').disabled=!x.alive;
    if(x.zmq_host&&$('inferMonHost')&&document.activeElement!==$('inferMonHost'))$('inferMonHost').value=x.zmq_host;
    const ports=x.status?.ports||{};
    document.querySelectorAll('#inferMonHz [data-port]').forEach(el=>{
      const p=ports[el.dataset.port]||{};
      const b=el.querySelector('b');
      if(b)b.textContent=p.hz!=null?`${Number(p.hz).toFixed(1)} Hz`:'—';
      el.classList.toggle('alive',!!p.alive);
    });
    if($('inferMonHint'))$('inferMonHint').textContent=`pose: ${x.status?.pose_hint||'—'} · keys: ${(x.status?.camera_keys||[]).join(',')||'—'}`;
    const img=$('inferMonCam'),ph=$('inferMonCamPh');
    if(x.alive&&x.status?.has_camera_jpeg){
      if(ph)ph.hidden=true;if(img){img.hidden=false;
        const token=String(x.status?.camera_mtime||Date.now());
        if(img.dataset.mtime!==token){img.dataset.mtime=token;img.src='/api/infer/monitor/camera.jpg?t='+token}
      }
    }else if(!x.alive){
      if(img){img.hidden=true;img.removeAttribute('src');delete img.dataset.mtime}if(ph)ph.hidden=false;
    }
  }catch(_){}
}
function startInferMonPolling(){
  if(inferMonTimer)return;
  inferMonTimer=setInterval(()=>{
    if(!$('page-infer')?.classList.contains('active'))return;
    void refreshInferMonitor();
  },1500);
}
function startInferDualFromJob(job){
  const ego=job.videos?.ego,wrist=job.videos?.wrist;
  // 04 全远端：不用 local_ref_root 改写路径。
  setupInferDualLoop(
    ego?`/api/infer/video?path=${encodeURIComponent(ego)}`:'',
    wrist?`/api/infer/video?path=${encodeURIComponent(wrist)}`:''
  );
}
async function killG1Deploy(){
  try{
    const r=await post('/api/infer/kill-deploy');
    toast(r.message||'已停止 g1_deploy');
  }catch(e){toast(e.message,true)}
}
async function stopInferJob(){
  const id=activeInferJob||(state.infer_jobs||[]).find(j=>j.status==='running')?.id;
  if(!id)return toast('没有运行中的推理',true);
  try{
    await post(`/api/infer/jobs/${id}/cancel`);
    try{await post('/api/infer/kill-deploy')}catch(_){/* best-effort */}
    toast('已发送终止');
    await loadState();
    renderInferJobs();
    $('inferInitBtn').disabled=false;
    $('inferStartBtn').disabled=false;
    $('inferStopBtn').disabled=true;
  }catch(e){toast(e.message,true)}
}
function renderInferJobs(){
  const el=$('inferJobList'),jobs=state.infer_jobs||[];
  if(!el)return;
  if(!jobs.length){el.className='train-job-list empty';el.textContent='暂无推理任务';return}
  el.className='train-job-list';
  el.innerHTML=jobs.slice(0,10).map(j=>`<div class="train-job-item${j.id===activeInferJob?' active':''}" onclick="selectInferJob('${j.id}')"><b>${esc(j.id)} · ${esc(j.skill_title||'')}</b><small>${statusLabel(j.status)} · ${esc(j.phase||'')} · ${esc(j.message||'')}</small></div>`).join('');
  const running=jobs.some(j=>j.status==='running');
  if($('inferStopBtn'))$('inferStopBtn').disabled=!running;
  if($('inferInitBtn'))$('inferInitBtn').disabled=running;
  if($('inferStartBtn'))$('inferStartBtn').disabled=running;
  const job=jobs.find(j=>j.id===activeInferJob)||jobs.find(j=>j.status==='running');
  if(job)setInferPhase(job.phase||'starting');
}
async function selectInferJob(id){activeInferJob=id;renderInferJobs();await refreshInferLogs();await refreshInferPreview();await refreshInferMp4()}
function renderInferStackStatus(stack){
  const el=$('inferStackStatus');if(!el)return;
  const lights=(stack&&stack.lights)||{};
  el.querySelectorAll('[data-k]').forEach(node=>{
    const key=node.dataset.k;
    let st=lights[key];
    if(!st){
      if(stack&&stack[key]===true)st='ok';
      else if(stack&&stack[key]===false)st='idle';
      else st='idle';
    }
    node.classList.remove('ok','error','idle','skip','on','off');
    node.classList.add('lamp', st);
  });
}
function selectInferProcLog(which){
  inferProcLogTab=which||'sim';
  document.querySelectorAll('#inferProcTabs [data-log]').forEach(b=>{
    b.classList.toggle('active',b.dataset.log===inferProcLogTab);
  });
  const map={sim:inferProcLogs.sim,policy:inferProcLogs.policy,deploy:inferProcLogs.deploy};
  const empty={sim:'等待 MuJoCo Sim…',policy:'等待策略推理（Deploy Terminal 按 y 后启动）…',deploy:'等待 Deploy(sim)…'};
  paintInferLogView($('inferProcLog'), map[inferProcLogTab]||empty[inferProcLogTab]||'（日志为空）');
}
function setInferPhase(phase){
  const order=['starting','sim_ready','deploy_starting','init_done','standing','policy_starting','policy_ready','streaming','done'];
  const alias={
    queued:'starting',sim:'sim_ready',publisher:'deploy_starting',deploy:'init_done',
    armed:'standing',policy:'streaming',policy_ready:'standing',policy_starting:'standing'
  };
  phase=alias[phase]||phase||'starting';
  // Collapse fine-grained policy phases into standing/streaming for the simplified bar
  const barPhase=phase==='policy_starting'||phase==='policy_ready'?'standing'
    :phase==='done'?'streaming':phase;
  document.querySelectorAll('#inferPhaseBar span').forEach(s=>{
    s.classList.toggle('on',s.dataset.p===barPhase);
    const i=order.indexOf(barPhase),j=order.indexOf(s.dataset.p);
    s.classList.toggle('done',j>=0&&i>=0&&j<i);
  });
  const labels={
    starting:'等待一键启动',
    sim_ready:'Sim 就绪 — 请点 y',
    deploy_starting:'Deploy / 策略启动中…',
    init_done:'Init Done — 请点 ]',
    standing:'已站立 — 可 Enter',
    policy_starting:'Policy 武装中…',
    policy_ready:'Policy 就绪 — 按 Enter',
    streaming:'流式运行中',
    done:'完成',
  };
  if($('inferDeployPhase')){
    $('inferDeployPhase').textContent=labels[phase]||phase;
    $('inferDeployPhase').className='status-pill'+(phase==='init_done'?' ready':'');
  }
  if($('inferDeployHint')){
    $('inferDeployHint').textContent=phase==='init_done'
      ?'已检测到 Init Done，请点 ] 站立'
      :(phase==='sim_ready'?'Sim 已就绪：按 y 启动策略推理 + Deploy(sim)':'Sim 就绪后：y → 等 Init Done → ] → Enter');
  }
  const stepBtns=['inferDeployBtn','inferStandBtn','inferStreamBtn','inferStreamPBtn','inferPolicyBtn'];
  stepBtns.forEach(id=>{const b=$(id);if(!b)return;b.disabled=true;b.classList.remove('btn-next','primary','green')});
  const running=!!$('inferStopBtn')&&!$('inferStopBtn').disabled;
  if(!running&&phase==='starting')return;
  if($('inferDeployBtn'))$('inferDeployBtn').disabled=false;
  if(phase==='sim_ready'&&$('inferDeployBtn')){
    $('inferDeployBtn').classList.add('btn-next','primary','green');
  }
  if(phase==='init_done'&&$('inferStandBtn')){
    $('inferStandBtn').disabled=false;
    $('inferStandBtn').classList.add('btn-next','primary','green');
  }
  if(['standing','policy_starting','policy_ready','streaming','done'].includes(phase)){
    if($('inferStandBtn'))$('inferStandBtn').disabled=false;
    if($('inferStreamBtn')){
      $('inferStreamBtn').disabled=false;
      if(phase==='standing'||phase==='policy_ready')$('inferStreamBtn').classList.add('btn-next','primary','green');
    }
    if($('inferStreamPBtn'))$('inferStreamPBtn').disabled=false;
    if($('inferPolicyBtn'))$('inferPolicyBtn').disabled=false;
  }
}
function paintInferLogView(el, text){
  if(!el)return;
  const nearBottom=el.scrollHeight-el.scrollTop-el.clientHeight<96;
  // Cap DOM size — huge innerHTML rewrites every 2s OOM Edge tabs.
  const lines=String(text||'（日志为空）').split('\n');
  const raw=lines.slice(-400).join('\n');
  if(el.dataset.logHash===String(raw.length)+':'+raw.slice(-120))return;
  el.dataset.logHash=String(raw.length)+':'+raw.slice(-120);
  el.innerHTML=raw.split('\n').map(line=>{
    const t=esc(line);
    if(/Init Done|ZMQ STREAMING MODE: ENABLED|STUDIO got cmd=/i.test(line))
      return `<span class="log-ok">${t}</span>`;
    if(/STUDIO phase=|STUDIO waiting|wait_log: OK|sim loop starting|bound tcp/i.test(line))
      return `<span class="log-wait">${t}</span>`;
    if(/ERROR|TIMEOUT|Traceback|FAIL/i.test(line))
      return `<span class="log-hl">${t}</span>`;
    return t;
  }).join('\n');
  if(nearBottom||!el.dataset.userScroll)el.scrollTop=el.scrollHeight;
}
async function refreshInferLogs(){
  if(!activeInferJob)return;
  try{
    const x=await api(`/api/infer/jobs/${activeInferJob}/logs`);
    inferProcLogs={
      sim:x.sim_text||'',
      policy:x.policy_text||x.replay_text||'',
      deploy:x.deploy_text||'',
    };
    selectInferProcLog(inferProcLogTab);
    // Deploy Terminal focuses on deploy.log (01-style), not the merged blob
    const deployView=x.deploy_text||x.text||'（等待 Deploy…）';
    paintInferLogView($('inferDeployLog'),deployView);
    paintInferLogView($('inferTerminal'),x.engine_text||x.text||deployView);
    renderInferStackStatus(x.stack||x.job?.stack||{});
    const phase=x.phase?.phase||x.job?.phase||'queued';setInferPhase(phase);
    if(x.job){const i=(state.infer_jobs||[]).findIndex(j=>j.id===activeInferJob);if(i>=0)state.infer_jobs[i]=x.job;renderInferJobs()}
  }catch(_){}
}
async function refreshInferPreview(){
  if(!activeInferJob||document.hidden)return;
  try{
    const x=await api(`/api/infer/jobs/${activeInferJob}/preview-status`);
    const img=$('inferSimPreview'),v=$('inferSimVideo'),ph=$('inferSimPlaceholder');
    // Prefer single JPEG frames only — never attach multipart MJPEG in <img>.
    if(x.exists&&x.url){
      if(ph)ph.hidden=true;
      if(v)v.hidden=true;
      if(img){
        img.hidden=false;
        img.classList.remove('live-mjpeg');
        delete img.dataset.mjpeg;
        const token=String(x.mtime||Date.now());
        if(img.dataset.mtime!==token){
          img.dataset.mtime=token;
          img.src=x.url+'?t='+token;
        }
      }
    }
  }catch(_){}
}
function stopInferPreviewMedia(){
  const img=$('inferSimPreview'),v=$('inferSimVideo');
  if(img){img.removeAttribute('src');delete img.dataset.mjpeg;delete img.dataset.mtime;img.classList.remove('live-mjpeg')}
  if(v){try{v.pause()}catch(_){}v.removeAttribute('src');delete v.dataset.base}
}
function attachInferMjpeg(_jobId){
  // No-op: MJPEG multipart streams crash Edge/Chromium tabs under load.
  void refreshInferPreview();
}
async function refreshInferMp4(){
  if(!activeInferJob||document.hidden)return;
  try{
    const x=await api(`/api/infer/jobs/${activeInferJob}/mp4-status`);
    const v=$('inferSimVideo'),ph=$('inferSimPlaceholder'),img=$('inferSimPreview');
    if(x.exists&&x.url){
      // Prefer live JPEG preview when available; mp4 is archival fallback when idle.
      if(img&&!img.hidden&&img.dataset.mtime)return;
      if(ph)ph.hidden=true;if(img)img.hidden=true;if(v){v.hidden=false;
      const url=x.url+'?t='+Date.now();
      if(v.dataset.base!==x.url||v.ended){v.dataset.base=x.url;v.src=url;v.load();void v.play().catch(()=>{})}}
    }
  }catch(_){}
}
function startInferPolling(){
  if(inferPollTimer)return;
  inferPollTimer=setInterval(async()=>{
    if(document.hidden||!$('page-infer')?.classList.contains('active'))return;
    try{const st=await api('/api/state');state.infer_jobs=st.infer_jobs||[];renderInferJobs()}catch(_){}
    if(activeInferJob){
      await refreshInferLogs();
      await refreshInferPreview();
      await refreshInferMp4();
    }
  },2500);
}
function openInferSkillModal(){
  const s=inferSkillList().find(x=>x.id===$('inferSkill')?.value);
  $('saveInferTitle').value=s?.title||'';
  $('saveInferBadge').value=s?.badge||'SKILL';
  inferSkillModal.showModal();
}
async function submitInferSkill(){
  try{
    const x=await post('/api/infer/skills',{
      title:$('saveInferTitle').value.trim(),badge:$('saveInferBadge').value.trim(),
      prompt:$('inferPrompt').value,student_ckpt:$('inferCkpt').value.trim(),
      ref_root:$('inferRefRoot').value.trim(),ep:+$('inferEp').value||0
    });
    inferCatalog=x.catalog;inferSkillModal.close();toast('技能已保存');await initInferPage();
  }catch(e){toast(e.message,true)}
}