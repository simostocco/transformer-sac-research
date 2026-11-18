#include "core.hpp"
#include <numbers>
namespace shadow {
namespace {
// NumPy's contiguous float64 reduction order (eight accumulators, 128 block).
double sum(const double* a,size_t n){
 if(n<8){double s=-0.0;for(size_t i=0;i<n;++i)s+=a[i];return s;}
 if(n<=128){double r[8];for(size_t i=0;i<8;++i)r[i]=a[i];size_t i=8;
  for(;i+8<=n;i+=8)for(size_t j=0;j<8;++j)r[j]+=a[i+j];
  double s=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));for(;i<n;++i)s+=a[i];return s;}
 size_t half=n/2;half-=half%8;return sum(a,half)+sum(a+half,n-half);
}
template<class R> double rolling_sum(const R& r,size_t w){if(r.size<w)return NaN;std::array<double,304>a{};for(size_t i=0;i<w;++i){double x=r.back(w-1-i);a[i]=std::isfinite(x)?x:0;}return sum(a.data(),w);}
template<class R> double z(const R& r,size_t w){if(r.size<w)return NaN;std::array<double,304>a{};for(size_t i=0;i<w;++i){double x=r.back(w-1-i);a[i]=std::isfinite(x)?x*x:0;}double mean=rolling_sum(r,w)/w;double sd=std::sqrt(std::max(sum(a.data(),w)/w-mean*mean,0.0)+1e-12);return (r.back()-mean)/(sd+1e-12);}
double entropy(double b,double s){constexpr double eps=1e-12;if(!std::isfinite(b+s)||b+s<=eps)return NaN;double p=std::clamp(b/(b+s),eps,1-eps);return -(p*std::log(p)+(1-p)*std::log(1-p));}
double finite(double x){return std::isfinite(x)?x:0;}
}
FeatureRow FeatureState::append(const CanonicalBar& b,int ny_second){
 const double eps=config.eps;const auto W1=config.entropy,W2=config.persist,W3=config.slope,Wz=config.z;
 require(W1>0&&W1<=300&&W2>0&&W2<=300&&W3>1&&W3<=300&&Wz>0&&Wz<=300&&eps==1e-12,"unsupported feature config");
 auto ema=[&](double p,double x,double a){return initialized?a*finite(x)+(1-a)*p:finite(x);};
 double lp=b.close>1e-12?std::log(b.close):NaN;
 double ret=initialized?finite(lp-prev_log):0,r3=logp.size<3?0:finite(lp-logp.back(2)),r10=logp.size<10?0:finite(lp-logp.back(9));
 double vol=finite(b.volume),trades=double(b.trades),ofi=finite(b.ofi),bn=double(b.buy_n),sn=double(b.sell_n),bv=finite(b.buy_v),sv=finite(b.sell_v);
 double ratio_v=ofi/(vol+eps),vl=std::log1p(std::max(vol,0.0));
 double rf=ema(ret_fast,ret,.30),rs=ema(ret_slow,ret,.05),of=ema(ofi_fast,ratio_v,.30),os=ema(ofi_slow,ratio_v,.05),jerk=ret-prev_ret;
 buy_n.push(bn);sell_n.push(sn);buy_v.push(bv);sell_v.push(sv);ratio.push(ratio_v);abs_ratio.push(std::abs(ratio_v));vol_log.push(vl);logp.push(lp);
 sum_ratio+=finite(ratio_v);sumsq_ratio+=finite(ratio_v*ratio_v);c1.push(sum_ratio);c2.push(sumsq_ratio);
 double ec=entropy(rolling_sum(buy_n,W1),rolling_sum(sell_n,W1)),ev=entropy(rolling_sum(buy_v,W1),rolling_sum(sell_v,W1));
 double pc=std::isfinite(ec)?1-ec:0,pv=std::isfinite(ev)?1-ev:0;ec=finite(ec);ev=finite(ev);
 double mu=NaN,sd=NaN,sharpe=NaN;if(c1.size>=W2){mu=(c1.back()-(c1.size>W2?c1.back(W2):0))/double(W2);double m2=(c2.back()-(c2.size>W2?c2.back(W2):0))/double(W2);sd=std::sqrt(std::max(m2-mu*mu,0.0)+eps);sharpe=mu/(sd+eps);}
 cum_ofi+=ofi;cum_vol+=vol;cum_norm.push(cum_ofi/(cum_vol+eps));
 double slope=NaN,r2=NaN;if(cum_norm.size>=W3){std::array<double,300> y{},t{},yy{},xy{},xx{},err{};
  for(size_t i=0;i<W3;++i){y[i]=cum_norm.back(W3-1-i);t[i]=double(i)-double(W3-1)/2;xx[i]=t[i]*t[i];}
  double ym=sum(y.data(),W3)/double(W3);for(size_t i=0;i<W3;++i){yy[i]=(y[i]-ym)*(y[i]-ym);xy[i]=t[i]*(y[i]-ym);}
  slope=sum(xy.data(),W3)/(sum(xx.data(),W3)+eps);for(size_t i=0;i<W3;++i){double d=y[i]-(ym+slope*t[i]);err[i]=d*d;}
  r2=1-sum(err.data(),W3)/(sum(yy.data(),W3)+eps);
 }
 double iv=(bv-sv)/(bv+sv+eps),it=(bn-sn)/(bn+sn+eps),lv=std::log(bv+eps)-std::log(sv+eps),ln=std::log(bn+eps)-std::log(sn+eps);
 double imf=ema(imb_fast,iv,.30),ims=ema(imb_slow,iv,.05),vema=ema(ema_vol,vol,.03),tema=ema(ema_trades,trades,.03);
 double ang=2*std::numbers::pi*(double(ny_second)/86400.0);
 std::array<double,FeatureCount> a={std::sin(ang),std::cos(ang),ret,vl,ofi,ratio_v,r3,r10,rf,rs,rf-rs,jerk,of,os,of-os,ec,pc,ev,pv,mu,sd,sharpe,slope,r2,z(vol_log,Wz),z(abs_ratio,Wz),std::abs(mu),std::abs(sharpe),std::abs(slope),iv,it,lv,ln,imf-ims,vol/(vema+eps),trades/(tema+eps)};
 FeatureRow out;for(size_t i=0;i<FeatureCount;++i)out.values[i]=float(a[i]);warm.push(out);
 prev_log=lp;prev_ret=ret;ret_fast=rf;ret_slow=rs;ofi_fast=of;ofi_slow=os;imb_fast=imf;imb_slow=ims;ema_vol=vema;ema_trades=tema;initialized=true;return out;
}
}
