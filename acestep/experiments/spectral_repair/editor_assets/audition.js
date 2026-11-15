/* createAuditionController: explicit dependencies, state owned by this editor instance. */
function createAuditionController({editor, $, canvas, versions, allAudio, invalidate, refreshPlayhead}) {
  const ctx = canvas.getContext('2d');
  /* Selected-version decoding is cached; only the chosen interval is sent to the DSP worker. */
  let auditionBox=null, auditionActive=false, auditionToken=0, auditionWorker=null;
  let auditionCache=null, auditionURL=null;
  const auditionAudio=$('[data-role="audition-audio"]');
  function stopAudition(clear=false) {
    auditionToken++; auditionActive=false; auditionAudio.pause();
    if(auditionWorker){auditionWorker.terminate();auditionWorker=null;}
    if(clear){
      auditionBox=null; auditionAudio.removeAttribute('src');
      if(auditionURL){URL.revokeObjectURL(auditionURL);auditionURL=null;}
      $('[data-role="audition-status"]').textContent='Выберите «Прослушать область» и растяните прямоугольник.';
    }
    refreshPlayhead();
  }
  function auditionWav(channels, sampleRate) {
    const count=channels.length,length=channels[0].length;
    const bytes=new ArrayBuffer(44+length*count*4),data=new DataView(bytes);
    const text=(at,value)=>[...value].forEach((c,i)=>data.setUint8(at+i,c.charCodeAt(0)));
    text(0,'RIFF');data.setUint32(4,bytes.byteLength-8,true);text(8,'WAVE');text(12,'fmt ');
    data.setUint32(16,16,true);data.setUint16(20,3,true);data.setUint16(22,count,true);
    data.setUint32(24,sampleRate,true);data.setUint32(28,sampleRate*count*4,true);
    data.setUint16(32,count*4,true);data.setUint16(34,32,true);text(36,'data');
    data.setUint32(40,length*count*4,true);
    for(let i=0;i<length;i++)for(let c=0;c<count;c++)data.setFloat32(44+(i*count+c)*4,channels[c][i],true);
    return new Blob([bytes],{type:'audio/wav'});
  }
  async function playAudition() {
    if(!auditionBox||!editor.source)return;
    stopAudition(); const token=auditionToken,box={...auditionBox},item=versions.current();
    const rate=editor.source.stft.sample_rate,n=editor.source.stft.fft_size;
    const status=$('[data-role="audition-status"]');
    status.textContent='Готовлю прослушивание выделенной области…';
    try {
      if(!auditionCache||auditionCache.url!==item.preview_url){
        const url=item.preview_url;
        auditionCache={url,promise:fetch(url).then(r=>{if(!r.ok)throw Error('Не удалось прочитать WAV');return r.arrayBuffer();})
          .then(bytes=>new OfflineAudioContext(1,1,rate).decodeAudioData(bytes))};
      }
      const buffer=await auditionCache.promise;
      if(token!==auditionToken)return;
      const start=Math.floor(box.t0*rate),end=Math.min(buffer.length,Math.ceil(box.t1*rate));
      const left=Math.max(0,start-n),right=Math.min(buffer.length,end+n);
      const indices=box.channel==='both'?Array.from({length:buffer.numberOfChannels},(_,i)=>i):[Math.min(+box.channel,buffer.numberOfChannels-1)];
      const channels=indices.map(i=>buffer.getChannelData(i).slice(left,right));
      const workerURL=URL.createObjectURL(new Blob([`const render=${spectralAuditionJob.toString()};onmessage=e=>{try{const a=render(e.data);postMessage({channels:a},a.map(c=>c.buffer));}catch(error){postMessage({error:error.message});}}`],{type:'text/javascript'}));
      const worker=new Worker(workerURL);URL.revokeObjectURL(workerURL);auditionWorker=worker;
      const output=await new Promise((resolve,reject)=>{
        worker.onmessage=e=>e.data.error?reject(Error(e.data.error)):resolve(e.data.channels);
        worker.onerror=e=>reject(Error(e.message));
        worker.postMessage({channels,sampleRate:rate,fftSize:n,low:box.f0,high:box.f1,from:start-left,length:end-start},channels.map(c=>c.buffer));
      });
      worker.terminate(); if(auditionWorker===worker)auditionWorker=null;
      if(token!==auditionToken)return;
      if(auditionURL)URL.revokeObjectURL(auditionURL);
      auditionURL=URL.createObjectURL(auditionWav(output,rate));auditionAudio.src=auditionURL;
      auditionActive=true;
      status.textContent=`${box.t0.toFixed(2)}–${box.t1.toFixed(2)} с · ${Math.round(box.f0)}–${Math.round(box.f1)} Гц · ${versions.label(item)}`;
      await auditionAudio.play();
    }catch(error){
      if(token===auditionToken){auditionCache=null;status.textContent=`Прослушивание: ${error.message}`;}
    }
  }
  function selectAudition(a,b) {
    if(Math.abs(a.x-b.x)<3||Math.abs(a.y-b.y)<3)return;
    auditionBox={t0:Math.min(a.t,b.t),t1:Math.max(a.t,b.t),f0:Math.min(a.f,b.f),f1:Math.max(a.f,b.f),channel:$('[data-role="channel"]').value};
    invalidate();playAudition();
  }
  $('[data-role="audition-repeat"]').addEventListener('click',playAudition);
  $('[data-role="audition-stop"]').addEventListener('click',()=>stopAudition());
  auditionAudio.addEventListener('play',()=>{
    if(!auditionBox)return; auditionActive=true;
    allAudio().forEach(audio=>audio.pause()); refreshPlayhead();
  });
  ['pause','timeupdate','seeking','seeked','ended','loadedmetadata'].forEach(name=>auditionAudio.addEventListener(name,refreshPlayhead));
  function drawAudition(plot) {
    if(!auditionBox)return;
    const b=auditionBox,x=plot.x+(b.t0-editor.view.t0)/(editor.view.t1-editor.view.t0)*plot.w;
    const y=plot.y+(editor.view.f1-b.f1)/(editor.view.f1-editor.view.f0)*plot.h;
    const w=(b.t1-b.t0)/(editor.view.t1-editor.view.t0)*plot.w,h=(b.f1-b.f0)/(editor.view.f1-editor.view.f0)*plot.h;
    ctx.strokeStyle='#91ffab';ctx.fillStyle='#91ffab18';ctx.lineWidth=1.5;ctx.setLineDash([6,3]);
    ctx.fillRect(x,y,w,h);ctx.strokeRect(x,y,w,h);
  }

  return {stop:stopAudition, play:playAudition, select:selectAudition, draw:drawAudition, audio:auditionAudio,
    get box(){return auditionBox;}, get active(){return auditionActive;}};
}
