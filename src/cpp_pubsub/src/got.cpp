#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include<bits/stdc++.h>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "turtlesim/msg/pose.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;
using namespace std;

class TurtlePublisher : public rclcpp::Node
{
    public:
    TurtlePublisher() 
    : Node("minimal_publisher"), count_(0) {
        subscription_ = this->create_subscription<turtlesim::msg::Pose>(
      "/turtle1/pose", 10, std::bind(&TurtlePublisher::topic_callback, this, _1));
        publisher_= this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
        timer_ = this->create_wall_timer(5ms, std::bind(&TurtlePublisher::timer_callback, this));
        TurtlePublisher::set_goal();    
    }  

    private:
    float x_self,y_self,x_goal,y_goal,theta_self;
    void topic_callback(turtlesim::msg::Pose msg){
        x_self = msg.x;
        y_self = msg.y;
        theta_self = msg.theta; 
    }

    void set_goal(){
    cout << "Enter x coord:" << endl;
        cin >> x_goal;
    cout << "Enter y coord:" << endl;
        cin >> y_goal;
    }

    void timer_callback(){
        double g_dis,angle;
        double y_dash,x_dash;
        double lin_vel=0;
        double ang_vel=0;
        auto message = geometry_msgs::msg::Twist();
        g_dis = sqrt(pow(x_goal-x_self,2) + pow(y_goal-y_self,2));
        if (g_dis>0.01)
        {    
            lin_vel=.5*g_dis;
            y_dash=(y_goal-y_self);
            x_dash=(x_goal-x_self);
            angle = atan2(y_dash,x_dash)-theta_self;
            ang_vel=2.4*angle;
            message.linear.x = lin_vel;
            message.angular.z=ang_vel;
            RCLCPP_INFO(this->get_logger(), "The g_distance to goal is %f", g_dis);   
            publisher_->publish(message);
        }
        else{
            set_goal();
        }
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<   turtlesim::msg::Pose>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TurtlePublisher>());
    rclcpp::shutdown();
    return 0;
}