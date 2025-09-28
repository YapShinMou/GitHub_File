#include "ros/ros.h" // /opt/ros/noetic/include/ros/ros.h

int main(int argc,char*argv[])
{
	ros::init(argc,argv,"hello_world_node"); //rosnode initialize
	
	ros::NodeHandle handler;  // create Node Handler - handler
	ROS_INFO("hello world"); //Output，hello world
	
	ROS_INFO("\n");
	ROS_INFO("2025/5/3 leb edit");
	
	return 0;
}

//test test test
