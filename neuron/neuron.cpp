#include "../utils/vector.h"
#include "../utils/randomNumberGenerator.h"
#include <functional>
#include <stdexcept>
#include "../utils/json.hpp"
using json = nlohmann::json;
enum ActivationTypes {unipolarBinary,bipolarBinary,unipolarSigmoidal,bipolarSigmoidal,relu,parametricLeakyRelu,exponential,swish,nothing};
struct ActivationParameters {
  float unipolarBinaryThreshold=0.0;
  float bipolarBinaryThreshold=0.0;
  float unipolarSigmoidalLambda=1.0;
  float bipolarSigmoidalLambda=1.0;
  float reluThreshold=0.0;
  float parametricLeakyReluParameter=0.001;
  float parametricLeakyReluThreshold=0.0;
  float exponentialParameter=1.0;
  float exponentialThreshold=0.0;
};
class Neuron{
  public:
  
  float output;
  float bias;
  float neti;
  ActivationParameters activationParameters;
  std::function<float(float,ActivationParameters)>activationFunc;
  Vector weight;
  float fNeti;
  float delE;
  float delB;
  Vector delW;
  int weightDimension;
  Neuron(ActivationTypes activationType,Vector weight,float bias=0,ActivationParameters parameters=ActivationParameters());
  ActivationTypes activationType;
  Neuron(ActivationTypes activationType,int weightVectorSize,float bias=0, ActivationParameters parameters=ActivationParameters());
  Neuron(){}
  void setActivationFunction();
  float calcNeti(Vector input);
  float calcFNeti();
  float calcFDashNeti();
  void update();
  void print();
  json parseNeuron();
};
Neuron::Neuron(ActivationTypes activationType, Vector weight, float bias, ActivationParameters parameters)
    : activationType(activationType), weight(weight), bias(bias), activationParameters(parameters) {
      // cout<<activationType<<endl;
      weightDimension=weight.size;
    setActivationFunction();
    }
Neuron::Neuron(ActivationTypes activationType,int weightVectorSize,float bias,ActivationParameters parameters):activationType(activationType),activationParameters(parameters),bias(bias){
      // cout<<activationType<<endl;
      weightDimension=weightVectorSize;
      
    this->weight=Vector(weightVectorSize);
    for(int i=0;i<weight.size;i++){
      weight[i]=generateRandomFloat(-0.25,0.25);
    }
    setActivationFunction();
};
void Neuron::setActivationFunction(){
  switch (activationType)
  {
  case unipolarBinary:
    activationFunc= [this](float neti,ActivationParameters activationParameters ){
      return (neti>=activationParameters.unipolarBinaryThreshold)?1.0:0.0;};
      break;
  case bipolarBinary:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return (neti>=activationParameters.bipolarBinaryThreshold)?1.0:-1.0;
    };
    break;
  case unipolarSigmoidal:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return 1/(1+exp(-activationParameters.unipolarSigmoidalLambda*neti));
    };
    break;
  case bipolarSigmoidal:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return 2/(1+exp(-activationParameters.bipolarSigmoidalLambda*neti))-1.0;
    };
    break;
      case relu:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return max(activationParameters.reluThreshold,neti);
    };
    break;
      case parametricLeakyRelu:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return neti>=activationParameters.parametricLeakyReluThreshold?neti:activationParameters.parametricLeakyReluParameter*neti;
    };
    break;
      case exponential:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return neti>=activationParameters.exponentialThreshold?neti:activationParameters.exponentialParameter*(exp(neti)-1);
    };
    break;
      case swish:
    activationFunc=[this](float neti,ActivationParameters activationParameters){
      return neti/(1+exp(-neti));
    };
    break;
     default:
     activationFunc=[this](float neti,ActivationParameters activationParameters){
      // cout<<"Nothinf"<<neti<<endl;
      return neti;
     };
    break;
  }
}
void Neuron::print(){
  // cout<<"Weight ";
  // for(int i=0;i<weight.size;i++){
  //   cout<<weight[i]<<" ";
  // }
  // cout<<bias;
  cout<<"Neti="<<neti<<" FnetI="<<fNeti<<"Activation Type="<<activationType;
  // <<" fDash="<<calcFDashNeti()<<" dele=" <<delE<<" bias= "<<bias<<" delB="<<delB<<" weights=";
  // for(int i=0;i<delW.size;i++){
  //  cout<<delW[i]<<" ";
  // }
  cout<<endl<<endl;
}
float Neuron:: calcNeti(Vector input){
  return neti=weight*input+bias;
}
float Neuron::calcFNeti(){
   if (!activationFunc) {
      throw std::runtime_error("Activation function not set!");
   }
  //  if(activationType==nothing)
  //  cout<<"calculating the "<<neti<<endl;
  return fNeti= activationFunc(neti,activationParameters);
}
float Neuron::calcFDashNeti(){
  if(activationType==unipolarSigmoidal)
  return activationParameters.unipolarSigmoidalLambda*fNeti*(1-fNeti);
  else if(activationType==bipolarSigmoidal)
  return activationParameters.bipolarSigmoidalLambda*(1-fNeti*fNeti);
else return neti;
}
void Neuron::update(){
  weight=weight+delW;
  bias=bias+delB;
}
json Neuron::parseNeuron(){
  json neuronJson;
    for(int l=0;l<weight.size;l++){
              neuronJson["weight"].push_back(weight[l]);
            }
            neuronJson["bias"] = bias;
  return neuronJson;  
}