const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const context=vm.createContext({WeakSet,Map});
vm.runInContext(fs.readFileSync(path.join(__dirname,'layout.js'),'utf8').split('const layoutPanelIds')[0],context);
const data=new Map(),frames=[],panels={};
const storage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v)};
function panel(id,parent=null){
 const header={nextElementSibling:{style:{display:'none'}},handlers:[],
  addEventListener(type,fn){this.handlers.push(fn);},
  click(){this.handlers.forEach(f=>f());this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none';}};
 return panels[id]={header,parent,querySelector:()=>header};
}
function open(id){return panels[id].header.nextElementSibling.style.display!=='none';}
function flush(){let count=0;while(frames.length){assert.ok(count++<30,'must converge');frames.shift()();}}
const ids=['ace','generation','mask'].map(n=>'nii-settings-'+n);
panel(ids[0]);panel(ids[1],ids[0]);panel(ids[2]);
const host={getElementById:id=>{const p=panels[id];return p&&(!p.parent||open(p.parent))?p:null;}};
const controller=context.createLayoutController(ids,host,storage,f=>frames.push(f));
controller.sync();assert.equal(open(ids[0]),false);
controller.setAll(true);flush();ids.forEach(id=>assert.equal(open(id),true));
controller.setAll(false);flush();assert.equal(open(ids[0]),false);assert.equal(open(ids[2]),false);
panels[ids[2]].header.click();flush();assert.equal(open(ids[2]),true);
assert.equal(data.get('nii.spectral.layout.v2.'+ids[2]),'1');
// New page defaults closed; restoring honors the saved mask state.
Object.values(panels).forEach(p=>{p.header.nextElementSibling.style.display='none';p.header.handlers=[];});
context.createLayoutController(ids,host,storage,f=>frames.push(f)).sync();flush();
assert.equal(open(ids[2]),true);assert.equal(open(ids[0]),false);
// Manual close must stay closed instead of fighting the remembered state.
panels[ids[2]].header.click();flush();assert.equal(open(ids[2]),false);
// Storage can be unavailable; controls still work by toggling current UI state.
const noStorage={getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}};
Object.values(panels).forEach(p=>p.header.handlers=[]);
const isolated=context.createLayoutController(ids,host,noStorage,f=>frames.push(f));isolated.sync();
isolated.setAll(true);flush();ids.forEach(id=>assert.equal(open(id),true));
isolated.setAll(false);flush();assert.equal(open(ids[0]),false);
console.log('layout: global expand/collapse, lazy nested panels, reload restoration and manual close passed');
