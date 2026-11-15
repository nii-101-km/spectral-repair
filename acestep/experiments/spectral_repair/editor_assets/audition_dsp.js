/* Pure overlap-add band isolation; runs in a worker and never changes source samples. */
function spectralAuditionJob(job) {
  const {channels, sampleRate, fftSize:n, low, high, from, length} = job, hop = n/4;
  function fft(re, im, inverse) {
    for (let i=1,j=0;i<n;i++) {
      let bit=n>>1; for (;j&bit;bit>>=1) j^=bit; j^=bit;
      if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}
    }
    for(let size=2;size<=n;size*=2){
      const angle=(inverse?2:-2)*Math.PI/size, wr=Math.cos(angle), wi=Math.sin(angle);
      for(let base=0;base<n;base+=size){
        let ar=1,ai=0;
        for(let j=0;j<size/2;j++){
          const a=base+j,b=a+size/2,br=re[b]*ar-im[b]*ai,bi=re[b]*ai+im[b]*ar;
          re[b]=re[a]-br;im[b]=im[a]-bi;re[a]+=br;im[a]+=bi;
          const next=ar*wr-ai*wi;ai=ar*wi+ai*wr;ar=next;
        }
      }
    }
    if(inverse)for(let i=0;i<n;i++){re[i]/=n;im[i]/=n;}
  }
  const window=Float64Array.from({length:n},(_,i)=>.5-.5*Math.cos(2*Math.PI*i/n));
  const keep=Uint8Array.from({length:n},(_,i)=>{
    const hz=Math.min(i,n-i)*sampleRate/n;return hz>=low&&hz<=high?1:0;
  });
  return channels.map(input=>{
    const out=new Float64Array(input.length), norm=new Float64Array(input.length);
    const re=new Float64Array(n),im=new Float64Array(n);
    for(let start=-n+hop;start<input.length;start+=hop){
      for(let i=0;i<n;i++){re[i]=(input[start+i]||0)*window[i];im[i]=0;}
      fft(re,im,false);
      for(let i=0;i<n;i++)if(!keep[i]){re[i]=0;im[i]=0;}
      fft(re,im,true);
      for(let i=0;i<n;i++){
        const at=start+i;if(at<0||at>=out.length)continue;
        out[at]+=re[i]*window[i];norm[at]+=window[i]*window[i];
      }
    }
    const fade=Math.min(Math.round(sampleRate*.005),Math.floor(length/2));
    return Float32Array.from({length},(_,i)=>{
      const weight=fade?Math.min(1,i/fade,(length-1-i)/fade):1;
      return out[from+i]/(norm[from+i]||1)*weight;
    });
  });
}
