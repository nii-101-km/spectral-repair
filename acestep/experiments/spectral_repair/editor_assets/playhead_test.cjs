/* Playback mapping and gestures use the real frontend functions without GPU/DOM dependencies. */
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const path = require('node:path');
const source = name => fs.readFileSync(path.join(__dirname, name), 'utf8');
function element() {
  return {children:[], style:{}, attrs:{}, listeners:{}, checked:true,
    setAttribute(k,v){this.attrs[k]=v;}, appendChild(x){this.children.push(x);},
    replaceChildren(){this.children=[];}, addEventListener(k,v){this.listeners[k]=v;},
    focus(options){assert.equal(options.preventScroll,true);},
    setPointerCapture(id){this.capture=id;}, hasPointerCapture(id){return this.capture===id;},
    releasePointerCapture(){this.capture=null;}};
}
function audio(time=0) {
  return {currentTime:time, duration:67, paused:true, ended:false, isConnected:true,
    pause(){this.paused=true;}, play(){this.paused=false; return Promise.resolve();},
    addEventListener(){}, src:'source.wav', getAttribute(){return this.src;}, load(){},
    removeAttribute(){this.src='';}};
}
const nodes = {}, a=audio(20), b=audio(5), frames=[];
const context = vm.createContext({console, Math, Number, String,
  document:{createElement:element}, $:s=>nodes[s]||(nodes[s]=element()),
  editor:{source:{duration:67,offset:193}, view:{t0:15,t1:40}, plots:[{x:65,y:17,w:800,h:200}],drag:null},
  canvas:{getBoundingClientRect:()=>({left:10}),clientWidth:882},
  preview:a, registry:{all:()=>[a,b]}, audition:{active:false,box:null,audio:{}},
  clamp:(x,lo,hi)=>Math.max(lo,Math.min(hi,x)),requestAnimationFrame:f=>{frames.push(f);return frames.length;},
  queueMicrotask, message:assert.fail});
vm.runInContext(source('playhead_math.js')+source('playback.js')+source('playhead.js')+source('seek_events.js')+`
  const playback=createPlaybackController({editor,$,preview,registry,stopAudition(){},watchPlayhead(){},
    message, defer:queueMicrotask});
  const playhead=createPlayheadController({editor,$,canvas,preview,playback,audition,
    getSeekDrag:()=>seeking.gesture, frame:requestAnimationFrame});
  const seeking=createSeekController({editor,canvas,playhead,message});`,context);
const run = code => vm.runInContext(code, context);
assert.equal(run('playheadTime(475,10,editor.plots[0],editor.view,67)'),27.5);
assert.equal(run('playheadTime(-100,10,editor.plots[0],editor.view,67)'),15);
assert.equal(run('playheadTime(2000,10,editor.plots[0],editor.view,67)'),40);
assert.equal(run('playheadX(27.5,editor.plots[0],editor.view)'),465);
assert.equal(run('playheadInView(5,editor.view)'),false);
run('playhead.refresh()');
assert.equal(nodes['[data-role="playheads"]'].children.length,1);
assert.equal(nodes['[data-role="seek-ruler"]'].attrs['aria-valuenow'],'20.000');
const ruler=nodes['[data-role="seek-ruler"]'];
const event = x => ({clientX:x,button:0,isPrimary:true,pointerId:3,
  preventDefault(){},stopPropagation(){}});
ruler.listeners.pointerdown(event(475));
assert.equal(a.currentTime,27.5); assert.equal(b.currentTime,27.5); assert.equal(a.paused,true);
ruler.listeners.pointermove(event(555)); ruler.listeners.pointerup(event(555));
assert.equal(a.currentTime,30); assert.equal(a.paused,true);
a.paused=false; ruler.listeners.pointerdown(event(395));
assert.equal(a.paused,true); ruler.listeners.pointerup(event(395)); assert.equal(a.paused,false);
ruler.listeners.keydown({key:'ArrowRight',shiftKey:true,preventDefault(){}});
assert.equal(a.currentTime,26);
nodes['[data-role="sync"]'].checked=false;
b.currentTime=5; ruler.listeners.pointerdown(event(475)); ruler.listeners.pointerup(event(475));
assert.equal(b.currentTime,5);
run("playback.switchVersion({preview_url:'next.wav'},()=>true); playhead.seek(0); preview.onloadedmetadata()");
assert.equal(a.currentTime,0); // A seek while a version loads also preserves zero.
run('editor.drag={mask:true}'); const previous=a.currentTime;
ruler.listeners.pointerdown(event(675)); assert.equal(a.currentTime,previous);
run('editor.drag=null; editor.source=null; playhead.refresh()');
assert.equal(nodes['[data-role="playheads"]'].children.length,0);
assert.equal(ruler.attrs['aria-disabled'],'true');
assert.equal(nodes['[data-role="seek-thumb"]'].style.display,'none');
console.log('playhead: mapping, sync, paused/playing drag, keyboard, mask isolation and clear passed');
