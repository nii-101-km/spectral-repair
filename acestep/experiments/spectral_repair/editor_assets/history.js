/* Versions change only the displayed spectrum/audio. The editable source mask stays intact. */
class SpectralVersions {
  constructor() { this.signature = null; this.items = []; this.selected = 'source'; }
  reset(data) {
    this.signature = data.signature; this.selected = 'source';
    this.items = [{id:'source', images:data.images, preview_url:data.preview_url,
      details:'Исходный фрагмент · Маска общая для вкладок; обработка использует исходный WAV.'}];
    this.merge(data.signature, data.history || []);
  }
  merge(signature, entries) {
    if (signature !== this.signature) return false;
    for (const entry of entries) {
      if (entry.id === 'source' || this.items.some(v => v.id === entry.id)) continue;
      this.items.push(entry);
    }
    return true;
  }
  select(id) {
    if (!this.items.some(v => v.id === id)) return false;
    this.selected = id; return true;
  }
  current() { return this.items.find(v => v.id === this.selected); }
  label(item) {
    if (item.id === 'source') return 'Исходник';
    const damage = item.kind === 'damage-preview';
    const peers = this.items.filter(v => v.id !== 'source' && (v.kind === 'damage-preview') === damage);
    const number = peers.indexOf(item) + 1;
    return damage ? `Провал ${number}` : `Репеинт ${number} · ${item.method}`;
  }
}

/* createVersionController: explicit dependencies, state owned by this editor instance. */
function createVersionController({editor, versions, preview, $, playback, cancelPlaybackSeek, stopAudition, finishGesture, invalidate}) {
  function renderVersions() {
    const list = $('[data-role="versions"]'); list.replaceChildren();
    versions.items.forEach(item => {
      const button = document.createElement('button'); button.type = 'button';
      button.role = 'tab'; button.textContent = versions.label(item);
      button.setAttribute('aria-selected', String(item.id === versions.selected));
      button.tabIndex = item.id === versions.selected ? 0 : -1;
      button.addEventListener('click', () => activateVersion(item.id));
      button.addEventListener('keydown', e => {
        const i = versions.items.indexOf(item), n = versions.items.length;
        const next = {ArrowLeft:(i+n-1)%n, ArrowRight:(i+1)%n, Home:0, End:n-1}[e.key];
        if (next === undefined) return;
        e.preventDefault(); activateVersion(versions.items[next].id);
        list.children[next].focus({preventScroll:true});
      });
      list.appendChild(button);
    });
    const item = versions.current();
    $('[data-role="version-details"]').textContent = item ? item.details : '';
    if (item) preview.setAttribute('aria-label', 'Звук: ' + versions.label(item));
    const active = list.querySelector('[aria-selected="true"]');
    if (active && versions.selected === 'source') list.scrollLeft = 0;
    else if (active) {
      const tab = active.getBoundingClientRect(), area = list.getBoundingClientRect();
      const left = area.left + list.children[0].getBoundingClientRect().width + 6;
      if (tab.right > area.right) list.scrollLeft += tab.right - area.right;
      else if (tab.left < left) list.scrollLeft -= left - tab.left;
    }
  }
  function activateVersion(id) {
    if (!versions.select(id)) return;
    cancelPlaybackSeek();
    stopAudition(true);
    if (editor.drag) finishGesture(editor.drag.last);
    const item = versions.current();
    if (item.kind === 'damage-preview') $('[data-role="overlay"]').checked = false;
    editor.images = item.images.map(source => {
      const image = new Image(); image.onload = invalidate; image.src = source; return image;
    });
    playback.switchVersion(item, () => versions.current() === item);
    renderVersions(); invalidate();
  }

  return {render:renderVersions, activate:activateVersion};
}
