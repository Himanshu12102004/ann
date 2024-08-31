#include<vector>
#include<iostream>
#include<numeric>
#include <initializer_list>
using namespace std;
class Vector{
  private:
   vector<float>v;
  public:
   int size;
   Vector(int n);
   Vector(vector<float> vec);
   Vector();
   Vector(std::initializer_list<float> initList) : v(initList) {}
   Vector operator + (const Vector & v);
   Vector operator - (const Vector & v);
   float operator * (const Vector & v);
   Vector operator * (const float & s);
   Vector operator / (const float & v);
   float& operator [] (const int &i);
   const float& operator[](const int &i) const;
};
Vector::Vector(int n):size(n){
   this->v=vector<float>(n,0.0);
}
Vector::Vector(vector<float> v ):v(v){
  size=v.size();
}
Vector::Vector(){
  size=1;
  v=vector<float>(1);
}
Vector Vector:: operator +( const Vector & v){
  if(size!=v.size)
  throw "Vector size mismatch for addition operation";
Vector ans(size);
  for(int i=0;i<size;i++){
    ans[i]=(*this)[i]+v[i];
  }
  return ans;
}
float Vector:: operator *(const Vector  & v){
  if(size!=v.size)
  throw "Vector size mismatch for addition operation";
Vector ans(size);
  for(int i=0;i<size;i++){
    ans[i]=(*this)[i]*v[i];
  }

  return accumulate(ans.v.begin(),ans.v.end(),0.0f);
}
Vector Vector:: operator -(const Vector & v){
  if(size!=v.size)
  throw "Vector size mismatch for addition operation";
Vector ans(size);
  for(int i=0;i<size;i++){
    ans[i]=(*this)[i]-v[i];
  }
  return ans;
}
Vector Vector:: operator *( const float & multiplier){
    Vector ans(size);
  for(int i=0;i<size;i++){
  ans[i]=(*this)[i]*multiplier;
  }
  return ans;
}
Vector Vector:: operator /( const float & divider){
    Vector ans(size);
  for(int i=0;i<size;i++){
    ans[i]=(*this)[i]/divider;
  }
  return ans;
}



// Modifing index i
float& Vector::operator [](const int& i){
     if (i < 0 || i >= size) {
        throw out_of_range("Index out of bounds");
    }
    return v[i];
}
// Retrieving index i
const float& Vector::operator [] (const int& i) const{
  if (i < 0 || i >= size) {
        throw out_of_range("Index out of bounds");
    }
    return v[i]; 
}
