/* Reversible display localization at the Gradio/Canvas boundary, independent of model state. */
function createSpectralLocale({host, store, Observer, translate}) {
  const key = 'nii.spectral.language.v1', records = new WeakMap(), bound = new WeakSet();
  const excluded = 'script,style,pre,code,td,[contenteditable="true"],'
    + '[data-role="payload"],#nii-run-metrics > :not(:first-child),#nii-raw-settings';
  const attributes = ['aria-label','aria-valuetext','title','placeholder','alt'];
  let language = 'en';
  try { if (store.getItem(key) === 'ru') language = 'ru'; } catch (_) {}
  function value(owner, field, raw, write) {
    let fields = records.get(owner);
    if (!fields) { fields = new Map(); records.set(owner, fields); }
    let entry = fields.get(field);
    if (!entry || raw !== entry.output) entry = {source:raw, output:raw};
    const next = translate(entry.source, language);
    entry.output = next; fields.set(field, entry);
    if (next !== raw) write(next);
  }
  function blocked(element, text = false) {
    return element?.closest?.(excluded + (text ? ",textarea,input" : ""));
  }
  function node(item) {
    if (item.nodeType === 3) {
      if (!blocked(item.parentElement, true)) value(item, 'text', item.data, text => {item.data=text;});
      return;
    }
    if (item.nodeType !== 1 || blocked(item)) return;
    for (const name of attributes) if (item.hasAttribute(name))
      value(item, name, item.getAttribute(name), text => item.setAttribute(name,text));
    if (!["INPUT","TEXTAREA"].includes(item.tagName)) item.childNodes.forEach(node);
  }
  function bind() {
    host.querySelectorAll('[data-nii-language]').forEach(select => {
      select.value = language;
      if (bound.has(select)) return;
      bound.add(select); select.addEventListener('change', () => set(select.value));
    });
  }
  function status() {
    // Translate display values only: non-editable status and closed dropdown captions.
    const field = host.querySelector('#nii-run-status textarea');
    host.querySelectorAll('input[role="listbox"]').forEach(input => {
      if (input.getAttribute('aria-expanded') !== 'true')
        value(input, 'value', input.value, text => {input.value=text;});
    });
    if (field && (field.disabled || field.readOnly))
      value(field, 'value', field.value, text => {field.value=text;});
  }
  function refresh() {
    bind(); status();
    host.documentElement.lang = language;
    value(host, 'title', host.title, text => {host.title=text;});
    node(host.body);
  }
  function set(next) {
    if (!['en','ru'].includes(next)) return;
    language=next;
    try { store.setItem(key,language); } catch (_) {}
    refresh();
    host.dispatchEvent(new Event('nii-language-change'));
  }
  // Translate only changed branches during drawing/playback, not the full page per frame.
  const observer = new Observer(changes => {
    for (const change of changes) {
      if (change.type === 'characterData') node(change.target);
      else if (change.type === 'attributes') {
        const element=change.target, name=change.attributeName;
        if (!blocked(element) && element.hasAttribute(name))
          value(element,name,element.getAttribute(name),text=>element.setAttribute(name,text));
      } else change.addedNodes.forEach(node);
    }
    bind(); status();
  });
  refresh(); observer.observe(host.body, {subtree:true,childList:true,characterData:true,
    attributes:true,attributeFilter:attributes});
  return {get language(){return language;}, text:value=>translate(value,language), set, refresh,
    destroy:()=>observer.disconnect()};
}
