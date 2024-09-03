#include "./ann/ann.cpp"
int main(){
  vector<LayerInfo> vec;
  LayerInfo lyr;
  for(int i=0;i<3;i++){
    lyr.activationType=unipolarSigmoidal;
    lyr.neuronCount=3;
    vec.push_back(lyr);
  }
  int inputVectorDimensions=2;
  /* This ANN constructor has a series of inputs the first input being the 
  input file for the training data the next one if for the input vector dimension 
  then is the output vector dimension then the number of training data sets you 
  want to use for training the next one is the mode of the ann then is the info of the layers
  then the output for the trained weights and biases 
    */
  Ann a("inputData/input.json",inputVectorDimensions,1,300,training,vec,"trainedModel/ann.json",0.05);
  // a.normalizeData();
  // a.train();
  while(true){
  vector<float> input(inputVectorDimensions);
    for(int i = 0; i < inputVectorDimensions; i++){
        cin>>input[i];
    }
    a.normalizeSingleInput(input);
    a.calcOutput();

a.printOutput();

}}