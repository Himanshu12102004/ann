#ifndef layerImported
#define layerImported
#include "./layer.cpp"
#endif
class HiddenLayer:public Layer{
  public:
   HiddenLayer(int neuronCount,int layerNumber):Layer(neuronCount,layerNumber){}
  void buildLayer(int lastNeuronCount,ActivationTypes activationType=unipolarSigmoidal);
};
void HiddenLayer:: buildLayer(int lastNeuronCount,ActivationTypes activationType){
  for(int i=0;i<neuronCount;i++){
  layerNeurons[i]=(Neuron(activationType,lastNeuronCount));
  }
}
