/* Full wiring regressions through UI events and the unchanged public mask bridge. */
const assert=require('node:assert/strict');
const {harness}=require('./controller_harness.cjs');
function trace(directory,modern=true) {
  const h=harness(directory,modern),snapshots=[];
  const save=()=>snapshots.push(h.snapshot());
  h.deliver(h.source);h.preview.onloadedmetadata();save();
  h.host.niiSpectralEditor.rectangle(.1,.3,500,1500,.025);h.flush();save();
  h.tools.find(t=>t.dataset.tool==='brush').click();
  h.canvas.emit('pointerdown',h.event(350,110));h.canvas.emit('pointerup',h.event(600,110));h.flush();save();
  h.actions.find(t=>t.dataset.action==='undo').click();h.flush();save();
  h.actions.find(t=>t.dataset.action==='redo').click();h.flush();save();
  h.tools.find(t=>t.dataset.tool==='eraser').click();
  h.canvas.emit('pointerdown',h.event(400,110));h.canvas.emit('pointerup',h.event(450,110));h.flush();save();
  const mask=h.host.niiSpectralEditor.serialize();
  const item={id:'one',method:'ACE',images:['repaint-left.png','repaint-right.png'],preview_url:'one.wav',details:'ACE'};
  h.preview.currentTime=.4;
  h.deliver({...h.source,revision:'history',history_only:true,history:[item],history_select:'one'});
  const oldMetadata=h.preview.onloadedmetadata;
  const ruler=h.field('[data-role="seek-ruler"]');
  ruler.emit('keydown',{key:'Home',shiftKey:false,preventDefault(){}});
  h.preview.onloadedmetadata();assert.equal(h.preview.currentTime,0);save();
  h.field('[data-role="versions"]').children[0].click();
  oldMetadata();h.preview.onloadedmetadata();h.flush();
  assert.equal(h.host.niiSpectralEditor.serialize(),mask);save();
  h.deliver({...h.source,revision:'stale',signature:'old-source',history_only:true,history:[item],history_select:'one'});
  assert.equal(h.preview.src,'source.wav');save();
  ruler.emit('pointerdown',h.event(475,0));ruler.emit('pointerup',h.event(475,0));h.flush();
  assert.equal(h.other.currentTime,h.preview.currentTime);assert.equal(h.host.niiSpectralEditor.serialize(),mask);save();
  h.deliver({...h.source,signature:'replacement',revision:'new-source',images:['new-left.png','new-right.png']});
  h.preview.onloadedmetadata();h.flush();assert.equal(h.snapshot().tabs.length,1);save();
  h.deliver({});assert.deepEqual(JSON.parse(h.host.niiSpectralEditor.serialize()),{});save();
  return snapshots;
}
const snapshots=trace(__dirname);
assert.equal(snapshots.length,12);
console.log('controllers: source, rectangle, brush, eraser, undo/redo, history, stale completion, metadata seek, sync and clear passed');
module.exports={trace};

// Damage previews change only the displayed WAV/images and hide the cyan selection.
{
  const h=harness(__dirname);h.deliver(h.source);h.preview.onloadedmetadata();
  h.host.niiSpectralEditor.rectangle(.1,.3,500,1500,.025);h.flush();
  const mask=h.host.niiSpectralEditor.serialize();
  h.deliver({...h.source,revision:'damage',history_only:true,history_select:'hole',history:[
    {id:'hole',kind:'damage-preview',method:'DSP',images:['hole-left.png','hole-right.png'],
     preview_url:'damaged.wav',details:'Тестовое удаление'}]});
  assert.equal(h.field('[data-role="overlay"]').checked,false);
  assert.equal(h.preview.src,'damaged.wav');
  assert.equal(h.host.niiSpectralEditor.serialize(),mask);
  assert.equal(h.snapshot().tabs[1].label,'Провал 1');
  h.field('[data-role="versions"]').children[0].click();h.flush();
  assert.equal(h.preview.src,'source.wav');
  assert.equal(h.host.niiSpectralEditor.serialize(),mask);
  console.log('damage preview: independent tab/audio, hidden overlay and source/mask preservation passed');
}
