#include "../neuron/neuron.cpp"
class Layer{
  public:
  vector<Neuron> layerNeurons;
  Layer(int neuronCount,int layerNumber);
  const int neuronCount;
  int layerNumber;
  float learningConstant=0.2;
  Vector passBackError();
  virtual  void computeDelE(Vector passedBackError);
  virtual void computeDelW(Vector previousLayerOutput);
  virtual void buildLayer(int lastLayerNeuronCount,ActivationTypes activationType=unipolarSigmoidal  )=0;
};
Layer::Layer(int neuronCount,int layerNumber):neuronCount(neuronCount),layerNumber(layerNumber){
  this->layerNeurons=vector<Neuron>(neuronCount);
}
Vector Layer::passBackError(){
  int lengthOfPassBack=layerNeurons[0].weightDimension;
  Vector passback(lengthOfPassBack);
  for(int i=0;i<lengthOfPassBack;i++){
    for(int j=0;j<neuronCount;j++){
      passback[i]+=layerNeurons[j].delE*layerNeurons[j].weight[i];
    }
  }
  return passback;
}
void Layer::computeDelE(Vector passedBackError){
  for(int i=0;i<neuronCount;i++){
  layerNeurons[i].delE=passedBackError[i]*layerNeurons[i].calcFDashNeti();
  layerNeurons[i].delB=learningConstant*layerNeurons[i].delE;
  }
}
void Layer::computeDelW(Vector previousLayerOutput){
for(int i=0;i<neuronCount;i++){
  layerNeurons[i].delW=previousLayerOutput*layerNeurons[i].delE*learningConstant;
}
}