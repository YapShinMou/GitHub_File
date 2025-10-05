// 訂閱發布 多執行緒 範例程式
#include "common/motor_crc_hg.h" // ~/unitree_ros2/example/src/include
#include "rclcpp/rclcpp.hpp" // /opt/ros/humble/include/rclcpp
#include "unitree_hg/msg/low_cmd.hpp"   // ~/unitree_ros2/cyclonedds_ws/install/unitree_hg/include
#include "unitree_hg/msg/low_state.hpp" // ~/unitree_ros2/cyclonedds_ws/install/unitree_hg/include
#include "unitree_hg/msg/motor_cmd.hpp" // ~/unitree_ros2/cyclonedds_ws/install/unitree_hg/include
#include <yaml-cpp/yaml.h>
#include <string>
#include <thread> //
#include <torch/torch.h>
#include <random>

class LowLevelCmdSender : public rclcpp::Node
{
public:
	LowLevelCmdSender() : Node("yap_g1_ros_rl")
	{
		lowstate_subscriber_ = this->create_subscription<unitree_hg::msg::LowState>("lowstate", 10,
			[this](const unitree_hg::msg::LowState::SharedPtr message) {LowStateHandler(message);});
		
		lowcmd_publisher_ = this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", 10);
		
		timer_ = this->create_wall_timer(std::chrono::milliseconds(2), [this] {
			Control();
		});
		
		input_thread_ = std::thread([this]() { this->main_loop(); }); //無頻率限制 不影響其他函式執行頻率
	}
	
	~LowLevelCmdSender()
	{
		// 結束前要確保 thread 收尾
		if (input_thread_.joinable())
		{
			input_thread_.join();
		}
	}
	
private:
	void Control()
	{
		low_command_.mode_pr = 0;
		low_command_.mode_machine = mode_machine_;
		for (int i = 0; i < 29; ++i)
		{
			low_command_.motor_cmd[i].mode = 1;
			low_command_.motor_cmd[i].tau = 0;
			low_command_.motor_cmd[i].q = 0;
			low_command_.motor_cmd[i].dq = 0.0;
			low_command_.motor_cmd[i].kp = 200;
			low_command_.motor_cmd[i].kd = 1.5;
		}
		
		get_crc(low_command_); // ~/unitree_ros2/example/src/src/common/motor_crc_hg.cpp
		lowcmd_publisher_->publish(low_command_);  // Publish lowcmd message
	}
	
	void LowStateHandler(const unitree_hg::msg::LowState::SharedPtr &message)
	{
		mode_machine_ = static_cast<int>(message->mode_machine);
		imu_ = message->imu_state;
		for (int i = 0; i < 29; i++)
		{
			motor_[i] = message->motor_state[i];
		}
	}
	
	int main_loop()
	{
		return 0;
	}
	
	//類別初始化只會執行一次
	rclcpp::TimerBase::SharedPtr timer_;  // ROS2 timer
	rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;         // ROS2 Publisher
	rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_subscriber_; // ROS2 Subscriber
	unitree_hg::msg::LowCmd low_command_;                // Unitree hg lowcmd message
	unitree_hg::msg::IMUState imu_;                      // Unitree hg IMU message
	std::array<unitree_hg::msg::MotorState, 29> motor_;  // Unitree hg motor state message
	int mode_machine_{};
	
	std::thread input_thread_;
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<LowLevelCmdSender>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
