#include "./utils/vector.h"
#include "./utils/randomNumberGenerator.h"
#include <functional>
#include <stdexcept>
enum ActivationFunctions {unipolarBinary,bipolarBinary,unipolarSigmoidal,bipolarSigmoidal,relu,parametricLeakyRelu,exponential,swish};
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
  private:
  Vector weight;
  float output;
  float bias;
  float neti;
  ActivationFunctions activationFunction;
  ActivationParameters activationParameters;
  std::function<float()>activationFunc;
  public:
  Neuron(ActivationFunctions activationFunction,Vector weight,float bias=0,ActivationParameters parameters=ActivationParameters());
  Neuron(ActivationFunctions activationFunction,int weightVectorSize,ActivationParameters parameters=ActivationParameters());
  void setActivationFunction();
  float calcNeti(Vector input);
  float calcFNeti();
};
Neuron::Neuron(ActivationFunctions activationFunction, Vector weight, float bias, ActivationParameters parameters)
    : activationFunction(activationFunction), weight(weight), bias(bias), activationParameters(parameters) {
    setActivationFunction();
    }
Neuron::Neuron(ActivationFunctions activationFunction,int weightVectorSize,ActivationParameters parameters=ActivationParameters()):activationFunction(activationFunction),activationParameters(parameters){
    this->weight=Vector(weightVectorSize);
    for(int i=0;i<weight.size;i++){
      weight[i]=generateRandomFloat(0.0,0.25);
    }
    setActivationFunction();
};
void Neuron::setActivationFunction(){
  switch (activationFunction)
  {
  case unipolarBinary:
    activationFunc=[this](){
      return (neti>=activationParameters.unipolarBinaryThreshold)?1.0:0.0;};
      break;
  case bipolarBinary:
    activationFunc=[this](){
      return (neti>=activationParameters.bipolarBinaryThreshold)?1.0:-1.0;
    };
    break;
  case unipolarSigmoidal:
    activationFunc=[this](){
      return 1/(1+exp(-activationParameters.unipolarSigmoidalLambda*neti));
    };
    break;
  case bipolarSigmoidal:
    activationFunc=[this](){
      return 2/(1+exp(-activationParameters.bipolarSigmoidalLambda*neti))-1.0;
    };
    break;
      case relu:
    activationFunc=[this](){
      return max(activationParameters.reluThreshold,neti);
    };
    break;
      case parametricLeakyRelu:
    activationFunc=[this](){
      return neti>=activationParameters.parametricLeakyReluThreshold?neti:activationParameters.parametricLeakyReluParameter*neti;
    };
    break;
      case exponential:
    activationFunc=[this](){
      return neti>=activationParameters.exponentialThreshold?neti:activationParameters.exponentialParameter*(exp(neti)-1);
    };
    break;
      case swish:
    activationFunc=[this](){
      return neti/(1+exp(-neti));
    };
    break;
     default:
            throw std::invalid_argument("Unsupported activation function");
    break;
  }
}
float Neuron:: calcNeti(Vector input){
  neti=(weight*input);
  return neti;
}
float Neuron::calcFNeti(){
  return activationFunc();
}