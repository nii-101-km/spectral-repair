/* Verify actual localization code with inert UI fixtures, without translating user data. */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const assert=require('node:assert/strict');
const scope=vm.createContext({Event:class{constructor(type){this.type=type;}}});
vm.runInContext(['locale_catalog','locale_text','locale'].map(n=>
  fs.readFileSync(path.join(__dirname,n+'.js'),'utf8')).join('\n'),scope);
const text=(s,lang)=>vm.runInContext(`translateSpectralText(${JSON.stringify(s)},${JSON.stringify(lang)})`,scope);
assert.equal(text('Канал 2','en'),'Channel 2');
assert.equal(text('Исходник и фрагмент · Исходник.wav · 45–105 с','en'),
  'Source and excerpt · Исходник.wav · 45–105 s');
assert.equal(text('Маска .npy','en'),'Mask .npy');
assert.equal(text('32.8 с · 500 Гц · бин 123:4','en'),'32.8 s · 500 Hz · bin 123:4');
assert.equal(text('Восстановить выделение','ru'),'Восстановить выделение');
assert.equal(text('Source.WAV · seed 3917560322','en'),'Source.WAV · seed 3917560322');
class Element {
  constructor(tag,children=[],attrs={}) {
    this.tagName=tag.toUpperCase();this.nodeType=1;this.childNodes=children;this.attrs=attrs;
    this.listeners={};this.value='';children.forEach(n=>n.parentElement=this);
  }
  closest(selector) {
    if(selector.includes(this.tagName.toLowerCase()) || this.protected) return this;
    return this.parentElement?.closest(selector)||null;
  }
  hasAttribute(name){return name in this.attrs;}
  getAttribute(name){return this.attrs[name];}
  setAttribute(name,value){this.attrs[name]=value;}
  addEventListener(name,fn){this.listeners[name]=fn;}
}
const node=s=>({nodeType:3,data:s});
function fixture(saved,denied=false) {
  const label=node('Восстановить выделение'),later=node('Провал 4');
  const prompt=new Element('textarea',[node('Исходник')]);prompt.value='Русский текст песни';
  const number=new Element('input',[],{'aria-label':'Начало, с'});number.value='32.8';
  const payload=new Element('script',[node('{"lyrics":"Исходник","mask_f32":"unchanged"}')]);
  const raw=node('Исходник');const protectedOutput=new Element('pre',[raw]);
  const picker=new Element('select'),status=new Element('textarea');status.disabled=true;
  status.value='Метод выбран. Нажмите «Восстановить выделение»; вкладка над полотном пока показывает предыдущий результат.';
  const body=new Element('main',[new Element('button',[label]),prompt,number,payload,protectedOutput]);
  const events=[],storage=new Map(saved?[['nii.spectral.language.v1',saved]]:[]);let observer;
  const host={body,documentElement:{lang:''},title:'НИИ-101 · Редактор спектральных масок',
    querySelectorAll:()=>[picker],querySelector:()=>status,dispatchEvent:e=>events.push(e.type)};
  const store={getItem:k=>{if(denied)throw Error('denied');return storage.get(k);},
    setItem:(k,v)=>{if(denied)throw Error('denied');storage.set(k,v);}};
  class Observer {constructor(fn){observer=fn;}observe(){}disconnect(){}}
  Object.assign(scope,{host,store,Observer,schedule:fn=>fn()});
  const locale=vm.runInContext('createSpectralLocale({host,store,Observer,schedule,translate:translateSpectralText})',scope);
  return {locale,label,prompt,number,payload,raw,picker,status,host,events,storage,later,
    deliver:changes=>observer(changes)};
}
assert.equal(text('Готово. Шаги: 32; Время: 1.5 с. Файлы: output','en'),
  'Done. Steps: 32; Time: 1.5 s. Files: output');
const f=fixture();
assert.equal(f.locale.language,'en');assert.equal(f.label.data,'Repair selection');
assert.equal(f.host.documentElement.lang,'en');assert.equal(f.number.attrs['aria-label'],'Start, s');
assert.equal(f.prompt.value,'Русский текст песни');assert.equal(f.prompt.childNodes[0].data,'Исходник');
assert.equal(f.number.value,'32.8');assert.equal(f.payload.childNodes[0].data,'{"lyrics":"Исходник","mask_f32":"unchanged"}');
assert.equal(f.raw.data,'Исходник');assert.match(f.status.value,/Method selected/);
f.locale.set('ru');assert.equal(f.label.data,'Восстановить выделение');
assert.equal(f.number.attrs['aria-label'],'Начало, с');assert.match(f.status.value,/Метод выбран/);
f.locale.set('en');assert.equal(f.label.data,'Repair selection');assert.equal(f.picker.value,'en');
assert.equal(typeof f.picker.listeners.change,'function');
f.later.parentElement=f.host.body;f.deliver([{type:'childList',addedNodes:[f.later]}]);
assert.equal(f.later.data,'Damage 4');
f.label.data='Сохранить маску';f.deliver([{type:'characterData',target:f.label}]);
assert.equal(f.label.data,'Save mask');f.locale.set('ru');assert.equal(f.label.data,'Сохранить маску');
assert.equal(fixture('ru').label.data,'Восстановить выделение');
assert.equal(fixture(undefined,true).label.data,'Repair selection');
assert.equal(f.prompt.value,'Русский текст песни');assert.equal(f.number.value,'32.8');
console.log('PASS: English default, Russian persistence, reversible labels/status, dynamic updates, protected model data');
