#pragma once
#include "core.hpp"
namespace shadow {
// Match NumPy 2.2.6's float32 log reduction/polynomial, including explicit FMA.
// Source references and coefficient attribution: docs/NUMERICS.md.
inline float input_log(float value){
 if(std::isnan(value)){return value;}
 if(value==0){return -INFINITY;}
 if(value<0){return float(NaN);}
 if(std::isinf(value)){return value;}
 int exponent;float x=std::frexp(value,&exponent);
 if(x<=0.7071067811865475244f){x=x+x;--exponent;}x-=1.0f;
 constexpr std::array<float,6> p={0.0f,9.999999999999998702752e-01f,2.112677543073053063722f,1.480000633576506585156f,3.808837741388407920751e-01f,2.589979117907922693523e-02f};
 constexpr std::array<float,6> q={1.0f,2.612677543073109236779f,2.453006071784736363091f,9.864942958519418960339e-01f,1.546476374983906719538e-01f,5.875095403124574342950e-03f};
 float numerator=p[5],denominator=q[5];for(int j=4;j>=0;--j){numerator=std::fma(numerator,x,p[size_t(j)]);denominator=std::fma(denominator,x,q[size_t(j)]);}
 return std::fma(float(exponent),0.6931471805599453094f,numerator/denominator);
}

constexpr size_t InputWidth=30,ContextLength=120,ReadyRows=183;
struct ClusterAssets {
 std::array<uint32_t,5> columns{};std::array<float,5> mean{},std{};
 std::array<std::array<float,5>,4> centers{};uint32_t count=0;
 int64_t predict(const FeatureRow& row)const{
  require(count>0&&count<=4,"cluster count");std::array<float,5> x{};
  for(size_t j=0;j<5;++j){require(columns[j]<FeatureCount&&std[j]>0,"cluster assets");if(!std::isfinite(row.values[columns[j]]))return -1;x[j]=(row.values[columns[j]]-mean[j])/std[j];}
  float best=std::numeric_limits<float>::infinity();int64_t label=0;
  // Only discrete labels are exposed by the authoritative MiniBatchKMeans
  // prediction. Frozen centers never update during warm append.
  for(size_t c=0;c<count;++c){float distance=0;for(size_t j=0;j<5;++j){float d=x[j]-centers[c][j];distance+=d*d;}if(distance<best){best=distance;label=int64_t(c);}}
  return label;
 }
};
struct InputAssets {std::array<int32_t,InputWidth> columns{};std::array<float,InputWidth> mean{},std{};int64_t neutral=4,neutral_ret=3;};
using InputRow=ModelInputRow;
struct LabeledRow {CanonicalBar bar{};FeatureRow feature{};int64_t k1=-1,k2=-1;};
struct TransformerContext {std::array<std::array<float,InputWidth+1>,ContextLength> x{};std::array<float,ContextLength> relative{};std::array<int64_t,ContextLength> k1{},k2{};};
struct ModelFacingState {
 Ring<LabeledRow,192> labeled;
 Ring<InputRow,ContextLength> inputs;
 Ring<InputRow,ReadyRows> ready;
 InputRow append(const CanonicalBar& b,const FeatureRow& row,const ClusterAssets& main,const ClusterAssets& ret,const InputAssets& assets){
  LabeledRow label{b,row,main.predict(row),ret.predict(row)};labeled.push(label);InputRow input;
  for(size_t j=0;j<InputWidth;++j){int32_t c=assets.columns[j];float raw;
   if(c>=0){require(size_t(c)<FeatureCount,"input column");raw=row.values[size_t(c)];}
   else{switch(c){case -1:raw=float(b.buy_n);break;case -2:raw=float(b.sell_n);break;case -3:raw=float(b.buy_v);break;case -4:raw=float(b.sell_v);break;default:throw std::logic_error("raw input column");}}
   require(assets.std[j]>0,"input scaler");input.values[j]=(raw-assets.mean[j])/assets.std[j];
  }
  input.log_close=input_log(std::max(float(b.close),1e-12f));input.k1=label.k1<0?assets.neutral:label.k1;input.k2=label.k2<0?assets.neutral_ret:label.k2;
  inputs.push(input);ready.push(input);return input;
 }
 TransformerContext context_at(size_t stop)const{
  require(stop>=ContextLength&&stop<=ready.size,"future or missing exact context");TransformerContext out;size_t first_lag=ready.size-stop+ContextLength-1;float first=ready.back(first_lag).log_close;
  for(size_t i=0;i<ContextLength;++i){auto& row=ready.back(first_lag-i);std::copy(row.values.begin(),row.values.end(),out.x[i].begin());float relative=row.log_close-first;out.x[i][InputWidth]=relative;out.relative[i]=relative;out.k1[i]=row.k1;out.k2[i]=row.k2;}return out;
 }
 TransformerContext context()const{
  require(inputs.size==ContextLength,"context not ready");TransformerContext out;float first=inputs.back(ContextLength-1).log_close;
  for(size_t i=0;i<ContextLength;++i){auto& row=inputs.back(ContextLength-1-i);std::copy(row.values.begin(),row.values.end(),out.x[i].begin());float relative=row.log_close-first;out.x[i][InputWidth]=relative;out.relative[i]=relative;out.k1[i]=row.k1;out.k2[i]=row.k2;}
  return out;
 }
};
}
