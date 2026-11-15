const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const context=vm.createContext({Float32Array,Float64Array,Uint8Array,Math});
vm.runInContext(fs.readFileSync(path.join(__dirname,'audition_dsp.js'),'utf8'),context);
const rate=48000, count=48000;
const tone=(hz,gain=1)=>Float32Array.from({length:count},(_,i)=>gain*Math.sin(2*Math.PI*hz*i/rate));
const a=tone(1000,.3),noise=tone(4000,.25);
const mixed=a.map((v,i)=>v+noise[i]),saved=mixed.slice();
const render=(channels,low,high,from=12000,length=24000)=>context.spectralAuditionJob({channels,sampleRate:rate,fftSize:2048,low,high,from,length});
const result=render([mixed],900,1100)[0];
function amplitude(signal,hz){
 let re=0,im=0,n=0;
 for(let i=2400;i<signal.length-2400;i++){re+=signal[i]*Math.cos(2*Math.PI*hz*i/rate);im+=signal[i]*Math.sin(2*Math.PI*hz*i/rate);n++;}
 return 2*Math.hypot(re,im)/n;
}
assert.ok(amplitude(result,1000)>.29);
assert.ok(amplitude(result,4000)<.0001);
assert.deepEqual(mixed,saved);assert.equal(result.length,24000);
assert.equal(Math.abs(result[0]),0);assert.equal(Math.abs(result.at(-1)),0);
const stereo=render([mixed,mixed.map(v=>-v)],0,24000);
let error=0;
for(let i=1000;i<23000;i++){error=Math.max(error,Math.abs(stereo[0][i]-mixed[12000+i]));assert.ok(Math.abs(stereo[0][i]+stereo[1][i])<1e-7);}
assert.ok(error<1e-6);
const short=render([mixed],0,24000,0,10)[0];assert.equal(short.length,10);assert.ok(short.every(Number.isFinite));
const dc=render([new Float32Array(count).fill(.2)],0,100)[0];assert.ok(Math.abs(dc[12000]-.2)<1e-5);
console.log('audition: passband, stopband, exact time crop, stereo polarity, full-band reconstruction, source isolation and edges passed');
