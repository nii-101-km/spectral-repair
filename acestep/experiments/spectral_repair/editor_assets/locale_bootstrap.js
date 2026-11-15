/* Keep language changes local: no Gradio callback, source reload or mask replacement. */
if (!window.niiLocale) window.niiLocale = createSpectralLocale({host:document,
  store:{getItem:key=>localStorage.getItem(key),setItem:(key,value)=>localStorage.setItem(key,value)},
  Observer:MutationObserver,schedule:queueMicrotask,translate:translateSpectralText});
window.niiLocale.refresh();
document.addEventListener('nii-language-change', () => {
  if (!root.isConnected) return;
  if (editor.grid) drawing.selectionChanged(); else drawing.invalidate();
});
