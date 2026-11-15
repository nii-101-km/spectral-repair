/* Discover full-track players, including Gradio shadow roots, independently of playback policy. */
function createAudioRegistry({host, exclude, onChange, frame, Observer}) {
  const observedRoots = new WeakSet();
  let pending = 0;
  const observer = new Observer(records => {
    if (pending || !records.some(record => record.type === 'attributes' ||
        [...record.addedNodes].some(node => node.nodeType === 1))) return;
    pending = frame(() => { pending = 0; onChange(); });
  });
  function all(parent = host) {
    if (!observedRoots.has(parent)) {
      observedRoots.add(parent);
      observer.observe(parent, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    }
    const result = [...parent.querySelectorAll('audio')].filter(audio =>
      !exclude(audio) && (audio.src || audio.currentSrc));
    parent.querySelectorAll('*').forEach(element => {
      if (element.shadowRoot) result.push(...all(element.shadowRoot));
    });
    return result;
  }
  return {all};
}
