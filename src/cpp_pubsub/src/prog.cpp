#include<iostream>
#include<cstring>
using namespace std;
 
 class hero{
   private:
    int health;
   public :
   char *name;
    char  level;
    static int timeToComplete;

   hero(){
      cout << "constructor called " << endl;
      name  = new char[100];
   }
// dont get cnfused 
// will put  health into h3(this->) health
// this is parameterisd constructor
   hero (int health, char level){
      cout << "this->" << this << endl;
      this->health = health; 
      this->level  = level;
   }
// copy constructor(this is default constrcutor 
//which already exists)
   hero (hero& temp){
      char *ch  = new char[strlen(temp.name)+1];// this is deep copy
      strcpy(ch,temp.name);
      this->name = ch;
      this-> health  = temp.health;
   }
// destructor (memory de allocate)
      
   int getHealth(){
      return health;
   }
   
   int getLevel(){
      return level;
   }

   void setHealth(int h){
       health = h;
   }
   
   void setLevel(char l){
       level = l;
   }
   ~hero(){
cout << "destructor called" << endl;
   }

 };

 class base{
   char x;
public: 
   base(int a) {x = a;}
   void print(){ cout << (int)x << endl;}

 };
 
 int hero :: timeToComplete = 5;
 
 int main() {
   
   // static 
    hero h1;
   // for parameterised constructor
   hero h3(10,'a');
   hero h4(10,'a');
      // or 
  // hero h4 = h3;
   cout << "address of ramesh " << &h3 << endl;
   // if
   h4 = h3;// values of h4 will be changed
   //to values of h3
    // dynamic 
    hero *h2 = new hero; 
    cout << "health is " << h1.getHealth() << endl;
    (*h2).setHealth(70);
    (*h2).setLevel(69);

      
    cout << "health is :" << (*h2).getHealth() << '\n';
    //or 
    cout << "health is :" << h2->getHealth() << '\n';

    cout << "health is :" << h1.getHealth() << '\n';
    cout << "level is :" << h1.level << '\n';
   
    cout << hero::timeToComplete << endl;

    base b(44);
    b.print();
    return 0;
 }