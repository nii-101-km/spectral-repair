/* Exercise ownership boundaries independently of Gradio and audio generation. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {harness}=require('./controller_harness.cjs');
let count=0;const test=(name,fn)=>{fn();count++;console.log('PASS',name);};
test('editor state instances do not share mutable arrays or mask/view state',()=>{
  const context=vm.createContext({});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'editor_state.js'),'utf8')+`
    const first=createEditorState(), second=createEditorState();
    first.images.push('one');first.plots.push({x:1});first.source={signature:'first'};
    first.view={t0:1,t1:2};first.tool='eraser';
    if(second.images.length||second.plots.length||second.source||second.view||second.tool!=='rectangle')
      throw Error('shared editor state');`,context);
});
test('audio registry finds shadow players and excludes audition/empty players',()=>{
  const preview={src:'source.wav'},shadowAudio={currentSrc:'shadow.wav'},audition={src:'audition.wav'};
  const empty={},roots=[],frames=[];let observedCallback,refreshes=0;
  const shadow={querySelectorAll:s=>s==='audio'?[shadowAudio]:[]};
  const host={querySelectorAll:s=>s==='audio'?[preview,audition,empty]:[{shadowRoot:shadow}]};
  class Observer{constructor(callback){observedCallback=callback;}observe(target){roots.push(target);}}
  const scope=vm.createContext({host,Observer,exclude:a=>a===audition,
    onChange:()=>refreshes++,frame:fn=>{frames.push(fn);return frames.length;}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'audio_registry.js'),'utf8')+
    'const registry=createAudioRegistry({host,exclude,onChange,frame,Observer});',scope);
  assert.deepEqual(Array.from(vm.runInContext('registry.all()',scope)),[preview,shadowAudio]);
  vm.runInContext('registry.all()',scope);assert.equal(roots.length,2);
  observedCallback([{type:'childList',addedNodes:[{nodeType:3}]}]);assert.equal(frames.length,0);
  observedCallback([{type:'attributes'}]);observedCallback([{type:'attributes'}]);
  assert.equal(frames.length,1);frames.shift()();assert.equal(refreshes,1);
});
test('repeated player discovery does not register duplicate event handlers',()=>{
  const h=harness(__dirname);h.deliver(h.source);
  const before=h.preview.listeners.play.length;
  vm.runInContext('playback.bind();playback.bind()',h.scope);
  assert.equal(h.preview.listeners.play.length,before);
});
test('seeking full audio leaves the audition player and mask unchanged',()=>{
  const h=harness(__dirname);h.deliver(h.source);h.preview.onloadedmetadata();
  h.audition.src='audition.wav';h.audition.currentTime=.13;
  const packet=h.host.niiSpectralEditor.serialize();
  h.field('[data-role="seek-ruler"]').emit('keydown',{key:'End',shiftKey:false,preventDefault(){}});
  assert.equal(h.preview.currentTime,1);assert.equal(h.other.currentTime,1);
  assert.equal(h.audition.currentTime,.13);assert.equal(h.host.niiSpectralEditor.serialize(),packet);
});
test('playing version switch resumes at its saved position after metadata',()=>{
  const h=harness(__dirname);h.deliver(h.source);h.preview.onloadedmetadata();
  h.preview.currentTime=.6;h.preview.play();
  h.deliver({...h.source,revision:'result',history_only:true,history_select:'one',history:[
    {id:'one',method:'ACE',images:['one.png','two.png'],preview_url:'one.wav',details:'ACE'}]});
  assert.equal(h.preview.paused,true);h.preview.onloadedmetadata();
  assert.equal(h.preview.currentTime,.6);assert.equal(h.preview.paused,false);
  h.preview.pause();h.flush();
});
test('unsynchronized players keep their positions when seeking',()=>{
  const h=harness(__dirname);h.deliver(h.source);h.preview.onloadedmetadata();
  h.other.currentTime=.2;h.field('[data-role="sync"]').checked=false;
  h.field('[data-role="seek-ruler"]').emit('keydown',{key:'End',shiftKey:false,preventDefault(){}});
  assert.equal(h.preview.currentTime,1);assert.equal(h.other.currentTime,.2);
});
console.log(count+' ownership regressions passed');
