const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
function setup() {
  const handlers = {}, fields = {};
  let top = 563, commits = 0;
  const canvas = {style:{}, addEventListener:(n,f)=>handlers[n]=f,
    getBoundingClientRect:()=>({left:0,top}), setPointerCapture:()=>{},
    focus:opts=>{if (!opts?.preventScroll) top -= 173;}};
  const root = {querySelectorAll:()=>[]};
  const $ = s => fields[s] ||= {value:s.includes('brush-ms')?'100':'100',addEventListener:()=>{}};
  const context = {canvas,root,$,ResizeObserver:class {observe(){}},invalidate:()=>{},
    selectionChanged:()=>commits++,updateView:()=>{},clampView:v=>v,
    editor:{plots:[{x:0,y:0,w:1000,h:200},{x:0,y:250,w:1000,h:200}],
    view:{t0:0,t1:1,f0:0,f1:5000}, state:{duration:1,stft:{sample_rate:10000}},
    drag:null,tool:'brush',cursor:null,dirty:false}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'grid.js'),'utf8')+'\neditor.grid=new SpectralGrid([101,51],.01,100,1);',context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'events.js'),'utf8') + `
    const pointer=createPointerController({editor,canvas,root,$,
      drawing:{invalidate,selectionChanged,updateView,clampView},
      selectAudition:(a,b)=>selectAudition(a,b)});`,context);
  const event=(x,y,id=1)=>({clientX:x,clientY:y+563,button:0,pointerId:id,preventDefault(){}});
  return {context,handlers,event,top:()=>top,commits:()=>commits};
}
let tests=0;
function test(name,fn){fn();tests++;console.log('PASS',name);}
test('partly visible canvas stays still and fast stroke reaches release position',()=>{
  const s=setup();s.handlers.pointerdown(s.event(100,100));s.handlers.pointerup(s.event(900,100));
  assert.equal(s.top(),563);assert.equal(s.context.editor.grid.values[90*51+25],1);
  assert.equal(s.context.editor.grid.past.length,1);
});
test('drag remains on the starting stereo plot',()=>{
  const s=setup();s.handlers.pointerdown(s.event(100,100));s.handlers.pointermove(s.event(900,350));
  assert.equal(s.context.editor.drag.last.f,0);s.handlers.pointerup(s.event(900,350));
});
test('lost capture finishes once and next gesture works',()=>{
  const s=setup();s.handlers.pointerdown(s.event(100,100));s.handlers.lostpointercapture();
  s.handlers.pointerup(s.event(900,100));assert.equal(s.commits(),1);
  s.handlers.pointerdown(s.event(900,100));s.handlers.pointerup(s.event(950,100));
  assert.equal(s.commits(),2);assert.equal(s.context.editor.grid.past.length,2);
});
test('other pointer cannot move or end active stroke',()=>{
  const s=setup();s.handlers.pointerdown(s.event(100,100));s.handlers.pointermove(s.event(900,100,2));
  s.handlers.pointerup(s.event(900,100,2));assert.ok(s.context.editor.drag);assert.equal(s.context.editor.drag.last.t,.1);
});
test('audition rectangle keeps mask and undo untouched and uses release coordinates',()=>{
  const s=setup();let box;
  s.context.editor.tool='audition';s.context.selectAudition=(a,b)=>{box=[a,b];};
  const before=s.context.editor.grid.values.slice();
  s.handlers.pointerdown(s.event(100,100));s.handlers.pointerup(s.event(900,180));
  assert.equal(box[0].t,.1);assert.equal(box[1].t,.9);assert.equal(box[1].f,500);
  assert.deepEqual(s.context.editor.grid.values,before);assert.equal(s.context.editor.grid.past.length,0);
});
console.log(tests+' pointer regression tests passed');
