/* Pure display translation; machine values, prompts and serialized masks stay untouched. */
function translateSpectralText(value, language = 'en') {
  if (language !== 'en' || !/[А-Яа-яЁё]/.test(value)) return value;
  const filenames = [];
  const protectedText = value.replace(/[^·\n]*\S\.(?:wav|npy|json)\b/giu, name => {
    filenames.push(name); return `\u0000${filenames.length-1}\u0000`;
  });
  const translated = protectedText.replace(spectralTranslationPattern, key => spectralEnglish[key]);
  return translated.replace(/(\d)\s+с(?=$|[^а-яё])/giu, '$1 s')
    .replace(/\u0000(\d+)\u0000/g, (_,index) => filenames[Number(index)]);
}
const spectralTranslationPattern = new RegExp(Object.keys(spectralEnglish)
  .sort((a,b) => b.length-a.length)
  .map(key => `(?<![\\p{L}])${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\p{L}])`)
  .join('|'), 'gu');
