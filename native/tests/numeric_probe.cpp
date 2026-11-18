#include "model_state.hpp"
#include <iostream>
#include <iomanip>
int main(){
 shadow::require(std::isinf(shadow::input_log(0.0f)),"log zero");
 shadow::require(std::isnan(shadow::input_log(-1.0f)),"log negative");
 std::cout<<std::setprecision(9);
 for(float value;std::cin>>value;)std::cout<<shadow::input_log(value)<<'\n';
 return std::cin.eof()?0:2;
}
