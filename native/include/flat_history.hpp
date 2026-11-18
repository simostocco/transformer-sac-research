#pragma once
#include "pure_engine.hpp"
#include <mkl_cblas.h>
namespace shadow {
inline constexpr size_t FlatPackedRows=184,FlatHistoryRows=65;
// Extra predecessor row is private state, not an ABI change. B's qualified
// 183-row retained snapshots and the 320-symbol stress layout stay unchanged.
struct NativeFlatHistory {
 FeatureState features;
 ModelFacingState model;
 Ring<InputRow,FlatPackedRows> packed;
 Ring<uint8_t,FlatPackedRows> finite_rows;size_t invalid_rows=0;
 NativeFlatHistory(){features.config={10,10,10,180};}
 void append(const CanonicalBar& b,int ny,const ClusterAssets& main,const ClusterAssets& ret,const InputAssets& assets){auto f=features.append(b,ny);auto input=model.append(b,f,main,ret,assets);bool finite=std::isfinite(input.log_close)&&input.k1>=0&&input.k2>=0;for(float value:input.values)finite=finite&&std::isfinite(value);if(finite_rows.size==FlatPackedRows&&!finite_rows.back(FlatPackedRows-1))--invalid_rows;finite_rows.push(uint8_t(finite));if(!finite)++invalid_rows;packed.push(input);}
 bool ready()const{return model.labeled.size==192&&packed.size==FlatPackedRows&&invalid_rows==0;}
 pure::v2::PreparedInput context(size_t index)const{
  require(ready()&&index<FlatHistoryRows,"flat exact context not ready");pure::v2::PreparedInput out;
  float origin=packed.back(FlatPackedRows-1-index).log_close;
  for(size_t i=0;i<120;++i){const auto& row=packed.back(FlatPackedRows-1-index-i);std::copy(row.values.begin(),row.values.end(),out.x.begin()+i*31);float relative=row.log_close-origin;out.x[i*31+30]=relative;out.relative[i]=relative;out.k1[i]=row.k1;out.k2[i]=row.k2;}
  return out;
 }
 // Physical slots repeat only the final logical row. Ownership stays on the
 // original symbol/snapshot; padding cannot acquire a PureJob or a result.
 void physical_batch(size_t start,std::array<pure::v2::PreparedInput,20>& out,size_t& logical)const{
  require(start<FlatHistoryRows,"flat B20 range");logical=std::min(size_t(20),FlatHistoryRows-start);
  for(size_t i=0;i<20;++i)out[i]=context(start+std::min(i,logical-1));
 }
};
// NumPy 2.x pairwise reduction for contiguous float32 vectors <=128 elements.
// Preserve eight accumulator lanes and the exact final addition tree.
inline float numpy_sum(const float* a,size_t n){
 if(n<8){float sum=-0.0f;for(size_t i=0;i<n;++i)sum+=a[i];return sum;}
 require(n<=128,"bounded float reduction");std::array<float,8> r;std::copy_n(a,8,r.begin());size_t i=8;
 for(;i+8<=n;i+=8)for(size_t j=0;j<8;++j)r[j]+=a[i+j];
 float sum=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));for(;i<n;++i)sum+=a[i];return sum;
}
inline float numpy_mean(const float* a,size_t n){return numpy_sum(a,n)/float(n);}
inline float numpy_std(const float* a,size_t n){float mean=numpy_mean(a,n);std::array<float,65> sq{};require(n<=sq.size(),"bounded float std");for(size_t i=0;i<n;++i){float d=a[i]-mean;sq[i]=d*d;}return std::sqrt(numpy_sum(sq.data(),n)/float(n));}
// Frozen pinned-flat observation builder. Quantiles are checkpoint ordered on
// input, lexically ordered in SAC. Every engineered scalar follows the actual
// IncrementalSACArrayBuilder formula and float32 reduction/mutation order.
struct NativeFlatObservation {
 using TF=std::array<float,439>;
 std::array<float,60> w06{},w04{};
 NativeFlatObservation(){for(auto spec:{std::pair{.6,&w06},std::pair{.4,&w04}}){double w=1;(*spec.second)[0]=1;for(size_t k=1;k<60;++k){w=w*((double(k)-1)-spec.first)/double(k);(*spec.second)[k]=float(w);}}}
 std::array<float,490> build(const std::array<TF,65>& tf,const std::array<double,65>& price,double max_notional=10000)const{
  require(max_notional==10000,"frozen flat max notional");std::array<std::array<float,65>,3> edge{};std::array<float,65> prices{},returns{};
  for(size_t t=0;t<65;++t){for(auto v:tf[t])require(std::isfinite(v),"nonfinite TF history");require(std::isfinite(price[t])&&price[t]>0,"invalid history price");edge[0][t]=tf[t][4]-std::abs(tf[t][5]);edge[1][t]=tf[t][34]-std::abs(tf[t][35]);edge[2][t]=edge[1][t]-edge[0][t];prices[t]=float(price[t]);returns[t]=t?prices[t]-prices[t-1]:0;}
  std::array<float,490> out{};for(size_t i=0;i<55;++i)out[i]=tf.back()[pure::v2::SacQuantileOrder[i]];
  std::array<float,3> d{},dd{},vol{},roll{};std::array<std::array<float,2>,3> fd{},roll_fd{};
  for(size_t e=0;e<3;++e){const auto& x=edge[e];d[e]=x[64]-x[63];dd[e]=(x[64]-x[63])-(x[63]-x[62]);std::array<float,60> diffs{};for(size_t i=0;i<60;++i)diffs[i]=x[i+5]-x[i+4];vol[e]=std::max(numpy_std(diffs.data(),60),1e-4f);
   std::array<float,5> deltas{};for(size_t i=0;i<5;++i)deltas[i]=x[i+60]-x[i+59];roll[e]=numpy_mean(deltas.data(),5);
   for(size_t w=0;w<2;++w){const auto& weights=w?w04:w06;std::array<float,5> endpoints{};for(size_t j=0;j<5;++j){std::array<float,60> reversed{};for(size_t k=0;k<60;++k)reversed[k]=x[60+j-k];endpoints[j]=cblas_sdot(60,weights.data(),1,reversed.data(),1);}fd[e][w]=endpoints[4];roll_fd[e][w]=numpy_mean(endpoints.data(),5);}
  }
  size_t i=55;for(size_t e=0;e<3;++e)out[i++]=edge[e][64];for(size_t e=0;e<3;++e)out[i++]=d[e]/vol[e];for(size_t e=0;e<3;++e)out[i++]=dd[e];for(size_t e=0;e<3;++e)for(size_t w=0;w<2;++w)out[i++]=fd[e][w]/vol[e];for(size_t e=0;e<3;++e)out[i++]=roll[e]/vol[e];for(size_t e=0;e<3;++e)for(size_t w=0;w<2;++w)out[i++]=roll_fd[e][w]/vol[e];
  std::array<float,3> drift{},rv{},up{},eff{};size_t wi=0;for(size_t width:{10,30,60}){auto first=returns.data()+65-width;drift[wi]=numpy_mean(first,width);rv[wi]=numpy_std(first,width);std::array<float,60> abs{},positive{};for(size_t j=0;j<width;++j){abs[j]=std::abs(first[j]);positive[j]=float(first[j]>0);}up[wi]=numpy_mean(positive.data(),width);eff[wi]=std::abs(prices[64]-prices[64-width])/std::max(numpy_sum(abs.data(),width),1e-6f);++wi;}
  for(auto v:drift){out[i++]=v;}
  for(size_t w=0;w<3;++w){out[i++]=drift[w]/std::max(rv[w],1e-6f);}
  for(auto v:up){out[i++]=v;}
  for(auto v:eff){out[i++]=v;}
  for(auto v:rv){out[i++]=v;}
  out[i++]=rv[0]/std::max(rv[2],1e-6f);require(i==95,"SAC engineered shape");
  // Frozen production flat-owner extras, including pending-first-real-bar zeros.
  out[99]=float(price[64]/max_notional);for(size_t j=0;j<384;++j)out[106+j]=tf[64][55+j];for(auto v:out)require(std::isfinite(v),"nonfinite SAC observation");return out;
 }
};
}
