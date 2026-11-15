const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const code=fs.readFileSync(require('node:path').join(__dirname,'history.js'),'utf8');
const scope={module:{exports:{}}};
vm.runInNewContext(code.split('const versions =')[0]+'\nmodule.exports=SpectralVersions;',scope);
const Versions=scope.module.exports;
const source={signature:'a',images:['original'],preview_url:'source.wav'};
const result={id:'one',method:'ACE',images:['repaired'],preview_url:'one.wav'};
let tests=0;
function test(name,fn){fn();tests++;console.log('PASS',name);}
test('original first and restored history ordered',()=>{
 const v=new Versions();v.reset({...source,history:[result]});
 assert.equal(v.current().preview_url,'source.wav');assert.equal(v.items[0].id,'source');
 assert.equal(v.label(v.items[1]),'Репеинт 1 · ACE');
});
test('tab switches audio and images together without changing source',()=>{
 const v=new Versions();v.reset({...source,history:[result]});v.select('one');
 assert.equal(v.current().images[0],'repaired');assert.equal(v.current().preview_url,'one.wav');
 v.select('source');assert.equal(v.current().images[0],'original');assert.equal(source.preview_url,'source.wav');
});
test('duplicate completion and stale context cannot corrupt history',()=>{
 const v=new Versions();v.reset(source);v.merge('a',[result]);v.merge('a',[result]);
 assert.equal(v.items.length,2);assert.equal(v.merge('b',[{id:'stale'}]),false);
 assert.equal(v.select('stale'),false);assert.equal(v.current().id,'source');
});
test('new source resets old tabs and active version',()=>{
 const v=new Versions();v.reset({...source,history:[result]});v.select('one');
 v.reset({...source,signature:'b'});assert.equal(v.items.length,1);assert.equal(v.current().id,'source');
});
test('damage previews do not renumber real repaint results',()=>{
 const v=new Versions();v.reset(source);
 const hole={id:'hole',kind:'damage-preview',method:'DSP',images:['hole'],preview_url:'hole.wav'};
 const next={...result,id:'two'};
 v.merge('a',[result,hole,next]);
 assert.equal(v.label(hole),'Провал 1');assert.equal(v.label(result),'Репеинт 1 · ACE');
 assert.equal(v.label(next),'Репеинт 2 · ACE');
 v.select('hole');assert.equal(v.current().preview_url,'hole.wav');
});
console.log(tests+' history frontend tests passed');
