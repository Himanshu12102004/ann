#ifndef layerImported
#define layerImported
#include "./layer.cpp"
#endif
#include "../utils/json.hpp"
using json = nlohmann::json;
class OutputLayer:public Layer{
  public:
   OutputLayer(int neuronCount,int layerNumber):Layer(neuronCount,layerNumber){}
  void buildLayer(int lastNeuronCount,ActivationTypes activationType=unipolarSigmoidal);
  void computeDelE(Vector desiredOutput);
};
void OutputLayer:: buildLayer(int lastNeuronCount,ActivationTypes activationType){
  for(int i=0;i<neuronCount;i++){
  layerNeurons[i]=(Neuron(activationType,lastNeuronCount));
  }
}
void OutputLayer::computeDelE (Vector desiredOutput){
  for(int i=0;i<neuronCount;i++){
    layerNeurons[i].delE=(desiredOutput[i]-layerNeurons[i].fNeti)*layerNeurons[i].calcFDashNeti();
  layerNeurons[i].delB=learningConstant*layerNeurons[i].delE;
  }
}