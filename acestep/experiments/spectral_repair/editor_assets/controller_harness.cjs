/* Minimal DOM/media fixture: runs the complete editor bootstrap without a browser or network. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
function harness(directory,modern=true) {
  const frames=[],observers=[],nodes=new Map();
  let mediaMutation=()=>{};
  function node() {
    return {children:[],style:{},dataset:{},attrs:{},listeners:{},value:'',checked:false,
      clientWidth:882,clientHeight:510,width:882,height:510,scrollLeft:0,isConnected:true,
      addEventListener(name,callback){(this.listeners[name]??=[]).push(callback);},
      emit(name,event={}){for(const callback of this.listeners[name]||[])callback(event);},
      click(){this.emit('click');},focus(){},setAttribute(k,v){this.attrs[k]=v;},
      getAttribute(k){return this.attrs[k]??null;},removeAttribute(k){delete this.attrs[k];},
      replaceChildren(){this.children=[];},appendChild(child){this.children.push(child);},
      querySelector(s){return this.children.find(c=>c.attrs['aria-selected']==='true')||field(s);},
      querySelectorAll(){return [];},getBoundingClientRect(){return {left:0,top:0,right:882,width:882};},
      setPointerCapture(id){this.capture=id;},hasPointerCapture(id){return this.capture===id;},
      releasePointerCapture(){this.capture=null;}};
  }
  function field(selector) {if(!nodes.has(selector))nodes.set(selector,node());return nodes.get(selector);}
  function audio() {
    const a=node();a.currentTime=0;a.duration=1;a.paused=true;a.ended=false;
    Object.defineProperty(a,'src',{get(){return this.attrs.src||'';},set(v){this.attrs.src=v;mediaMutation();}});
    a.pause=function(){const changed=!this.paused;this.paused=true;if(changed)this.emit('pause');};
    a.play=function(){const changed=this.paused;this.paused=false;if(changed)this.emit('play');return Promise.resolve();};
    a.load=function(){this.currentTime=0;this.pause();};
    return a;
  }
  const preview=audio(),other=audio(),audition=audio();other.src='control.wav';
  nodes.set('[data-role="audio"]',preview);nodes.set('[data-role="audition-audio"]',audition);
  field('[data-role="channel"]').value='both';field('[data-role="overlay"]').checked=true;
  field('[data-role="sync"]').checked=true;field('[data-role="operation"]').value='add';
  field('[data-role="brush-ms"]').value=100;field('[data-role="brush-hz"]').value=100;
  field('[data-role="soft"]').value=30;
  const context2d=new Proxy({createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})},
    {get:(target,key)=>target[key]??(()=>{})});
  const canvas=node();canvas.getContext=()=>context2d;
  const tools=['rectangle','brush','eraser','zoom','pan','audition'].map(tool=>Object.assign(node(),{dataset:{tool}}));
  const actions=['undo','redo','clear','invert','soften','reset'].map(action=>Object.assign(node(),{dataset:{action}}));
  const root=node();root.querySelector=s=>s==='canvas'?canvas:field(s);
  root.querySelectorAll=s=>s==='[data-tool]'?tools:s==='[data-action]'?actions:
    s==='[data-view]'?['t0','t1','f0','f1'].map(k=>field(`[data-view="${k}"]`)):[];
  const body=node();body.querySelectorAll=s=>s==='audio'?[preview,other,audition]:[];
  mediaMutation=()=>{for(const observer of observers)if(observer.target===body)observer.callback([{type:'attributes'}]);};
  const document={body,createElement:tag=>tag==='canvas'?Object.assign(node(),{getContext:()=>context2d}):node()};
  class Observer {constructor(callback){this.callback=callback;observers.push(this);}observe(target){this.target=target;}}
  const host={devicePixelRatio:1},scope=vm.createContext({console,window:host,document,
    element:{querySelector:()=>root},MutationObserver:Observer,ResizeObserver:class{observe(){}},
    Node:{ELEMENT_NODE:1},Image:class{constructor(){this.complete=true;}},
    requestAnimationFrame:f=>{frames.push(f);return frames.length;},queueMicrotask,
    getComputedStyle:()=>({color:'#fff'}),atob:s=>Buffer.from(s,'base64').toString('binary'),
    btoa:s=>Buffer.from(s,'binary').toString('base64'),Blob,URL,DataView,ArrayBuffer,
    Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Math,Number,String});
  const names=modern?['grid','playhead_math','editor_state','drawing','events','history','audition_dsp',
    'audition','audio_registry','playback','playhead','seek_events','bridge','bootstrap']:
    ['grid','playhead_math','drawing','events','history','audition_dsp','audition','playhead','seek_events','bridge'];
  vm.runInContext(names.map(n=>fs.readFileSync(path.join(directory,n+'.js'),'utf8')).join('\n'),scope);
  function flush(){let limit=100;while(frames.length&&limit--)frames.shift()();if(frames.length)throw Error('unbounded frames');}
  function deliver(data){field('[data-role="payload"]').textContent=JSON.stringify(data);
    for(const observer of observers)if(observer.target===field('[data-role="payload"]'))observer.callback([]);
    flush();}
  const source={signature:'source',revision:'first',shape:[101,51],stft:{sample_rate:10000,fft_size:100,hop_length:100},
    duration:1,offset:193,channels:2,images:['left.png','right.png'],mask_f32:'',preview_url:'source.wav'};
  const event=(x,y,id=1)=>({clientX:x,clientY:y,button:0,isPrimary:true,pointerId:id,
    preventDefault(){},stopPropagation(){}});
  const snapshot=()=>({packet:JSON.parse(host.niiSpectralEditor.serialize()),
    src:preview.src,time:preview.currentTime,paused:preview.paused,
    tabs:field('[data-role="versions"]').children.map(x=>({label:x.textContent,selected:x.attrs['aria-selected']})),
    view:['t0','t1','f0','f1'].map(k=>field(`[data-view="${k}"]`).value)});
  return {scope,host,source,root,canvas,preview,other,audition,tools,actions,field,flush,deliver,event,snapshot};
}
module.exports={harness};
