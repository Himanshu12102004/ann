
#ifndef layerImported
#define layerImported
#include "./layer.cpp"
#endif
class InputLayer:public Layer{
  public:
  InputLayer(int neuronCount,int layerNumber):Layer(neuronCount,layerNumber){};
  void buildLayer(int lastNeuronCount,ActivationTypes activationType=nothing);
};
void InputLayer:: buildLayer(int lastNeuronCount=1,ActivationTypes activationType){
  for(int i=0;i<neuronCount;i++){
  layerNeurons[i]=(Neuron(activationType,vector<float>(1,1)));
  }
}
