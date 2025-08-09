#include<iostream>
using namespace std;

// encaps and inheritance
// class student{
//     //encapsulation
// private:
//   string name ;
//   int age ;
//   int height;
// public:
// int getAge(){
//     return this->age;
// }
// };     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
int main(){
    int i = 0;
    for(i=0; i<=5 ;i++){
        int j = -1;
    for(j=-1;j<i;j++){
        cout << j << "\n";
    }
    }
    
}

// class human{
// public:
// int height;
// int weight ;
// int age;

// public:
// int getWight(){
//     return this->weight;
// }
// void setWeight(int w){
//     this->weight = w;
// }
// };
// // inheritance(single)

// class child:public human{
//  public:
//  string color;

//  void sleep(){
//     cout << "hello" << endl;
//  }
// };

// class maleChild:public child{ // multilevel
// };

// class child2: public child, public maleChild { // multiple
// };


// class Male:protected human{
//  public:
//  string color;

//  void sleep(){
//     cout << "hello" << endl;
//  }
//  int getHeight(){
//     return this->height;
//  }
// };

// class Female:private human{
//  public:
//  string color;

//  void sleep(){
//     cout << "hello" << endl;
//  }
//  int getAge(){
//     return this->age;
//  }
// };


// int main(){
// student first;
// child obj0;
// maleChild obj3;
// obj3.setWeight(45);
// Male obj1;
// Female obj2;
// child2  obj4;
// cout << obj0.age << endl;
// cout << obj1.getHeight() << endl;
// cout << obj2.getAge() << endl;


// return 0;
// }

// class A{
//  public:
//  void func1(){
//  cout << "inside 1" << endl;
//  }
//  void func2(){
//     cout << "inside 2" << endl;
//  }
// };

// class D{
// public:
// void func2(){
//  cout << "inside 2" << endl;
// }
// };

// class B: public A{
//     public:
//     void func2(){
//         cout << "inside 2" << endl;
//  }
// };

// class C: public A{
//     public:
//     void func3(){
//         cout << "inside 3" << endl;
//     }
// }; 
// class E: public A, public D{//inheritance ambguity
// };


// int main(){
//     A obj1;
//     obj1.func1();

//     B obj2;
//     obj2.func1();
//     obj2.func2();

//     C obj3;
//     obj3.func1();
//     obj3.func3();

//     E obj4;
//     obj4.A::func2();
//     obj4.D::func2();
 

// }

//Polymorphism

// class A{//func overloading(void<->int X)
//  public:
//  void sayHello(){
//     cout << "Hello world" << endl;
//  }

//  void sayHello(string name){ // return type change 
//  //karne se farak nai padta
//     cout << "Hello World" << endl;
//  }
// };


// class B{
//     public:
// int a,b;

// int add() {
//     return a+b;
// }
// void operator+ (B &obj){
//     int value1 = this -> a ;
//     int value2 = obj.a;
//     cout << value2-value1 << endl;
// }

// void operator() () {
//     cout << " brackett" << endl;
// }

// };

// class animal{ // runtime polymorphism 
//     public:
//     void speak(){
//     cout << "speak" << endl;
// }
// };

// class dog:public animal{
//     public:
//     void speak(){
//         cout << "bark" << endl; // if not written 
//         // anything then speak will be printed 
//         // this is function overwriting
//     }
// };

// int main() {
//     // A obj3;
//     // obj3.sayHello();

//     B obj1,obj2;
//     obj1.a = 4;
//     obj2.a = 7;
//     cout << obj1.add() + obj2.add() << endl;
//     obj1 + obj2;
//     obj1();

//     dog obj;
//     obj.speak();
//     return 0;
// }

// abstraction

