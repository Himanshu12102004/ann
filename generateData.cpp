#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include "utils/json.hpp"
using json = nlohmann::json;
using namespace  std;
struct DataPoint {
    vector<double> inputs;
    vector<double> outputs;
};

int main() {
    vector<DataPoint> data;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 100.0);

    for (int i = 0; i < 300; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        double z = dis(gen) * 0.7;

        double p = pow(x, 1) + sqrt(5 * y) + sqrt(z);
        double q = x + y + z;

        DataPoint point;
        point.inputs = {x, y, z};
        point.outputs = {p, q};

        data.push_back(point);
    }

    json j_data = json::array();
    for (const auto& point : data) {
        j_data.push_back({{"inputs", point.inputs}, {"outputs", point.outputs}});
    }

    ofstream file("inputData/input.json");
    file << j_data.dump(4); 
     cout<<"Data Generated Successfully\n";
    return 0;
}
