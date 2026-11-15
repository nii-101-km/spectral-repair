/* Accordion preferences contain only open flags, never generation values or prompts. */
function createLayoutController(ids, host, store, frame) {
  const bound = new WeakSet(), current = new Map(); let pending = false;
  const key = id => `nii.spectral.layout.v2.${id}`;
  function read(id) {
    if (current.has(id)) return current.get(id);
    try { return store.getItem(key(id)); } catch (_) { return null; }
  }
  function write(id, open) {
    current.set(id, open ? '1' : '0');
    try { store.setItem(key(id), open ? '1' : '0'); } catch (_) {}
  }
  function opened(header) { return header.nextElementSibling?.style.display !== 'none'; }
  function schedule() {
    if (!pending) { pending = true; frame(() => { pending = false; sync(); }); }
  }
  function sync() {
    ids.forEach(id => {
      const panel = host.getElementById(id), header = panel?.querySelector('button.label-wrap');
      if (!header) return;
      if (!bound.has(header)) {
        bound.add(header);
        // Capture the intent before Gradio toggles/mounts its content.
        header.addEventListener('click', () => { write(id, !opened(header)); schedule(); }, {capture:true});
      }
      const saved = read(id);
      if (saved !== null && (saved === '1') !== opened(header)) header.click();
    });
    [['nii-expand-settings',true],['nii-collapse-settings',false]].forEach(([id, open]) => {
      const button = host.getElementById(id);
      if (button && !bound.has(button)) {
        bound.add(button); button.addEventListener('click', () => setAll(open));
      }
    });
  }
  function setAll(open) { ids.forEach(id => write(id, open)); sync(); }
  return {sync, schedule, setAll};
}
const layoutPanelIds = ['source','stft','rectangle','ace','generation','expert','sampler',
  'test','mask','import','diagnostics','repaint-boundaries'].map(id => `nii-settings-${id}`);
const layoutController = createLayoutController(layoutPanelIds, document, {
  getItem:key=>localStorage.getItem(key),setItem:(key,value)=>localStorage.setItem(key,value)
}, requestAnimationFrame);
new MutationObserver(records => {
  if (records.some(record => [...record.addedNodes].some(node => node.nodeType === 1)))
    layoutController.schedule();
}).observe(document.body, {childList:true,subtree:true});
layoutController.sync();
