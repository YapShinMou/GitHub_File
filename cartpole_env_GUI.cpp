//範例程式
#include "rclcpp/rclcpp.hpp"
#include "my_package/msg/cartpole1.hpp"
#include "my_package/msg/cartpole2.hpp"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>

class CartPoleEnv : public rclcpp::Node
{
public:
	CartPoleEnv()
	: Node("cartpole_env_GUI")
	{
		publisher_ = this->create_publisher<my_package::msg::Cartpole1>(
			"cartpole_state", 10);
		
		subscriber_ = this->create_subscription<my_package::msg::Cartpole2>(
			"cartpole_motor", 10, 
			std::bind(&CartPoleEnv::State_subscribe, this, std::placeholders::_1));
	}
	
	bool init_mujoco()
	{
		// -----------------------------------
		// 載入模型與資料
		// -----------------------------------
		char error[1000];
		m_ = mj_loadXML("/home/yap/my_ws/src/mujoco_pkg/models/cart_pole.xml", 
			nullptr, error, 1000);
		if (!m_) {
			std::cerr << "Failed to load XML: " << error << std::endl;
			return false;
		}
		d_ = mj_makeData(m_);
		
		// -----------------------------------
		// 初始化 GLFW
		// -----------------------------------
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW" << std::endl;
			return false;
		}
		
		window_ = glfwCreateWindow(1200, 700, "MuJoCo GUI", NULL, NULL);
		if (!window_) {
			std::cerr << "Failed to create GLFW window" << std::endl;
			return false;
		}
		
		glfwMakeContextCurrent(window_);
		glfwSwapInterval(1);  // 垂直同步
		
		// -----------------------------------
		// 初始化 MuJoCo 可視化
		// -----------------------------------
		mjv_defaultCamera(&cam_);
		mjv_defaultOption(&opt_);
		mjv_defaultScene(&scn_);
		mjr_defaultContext(&con_);
		
		mjv_makeScene(m_, &scn_, 1000);
		mjr_makeContext(m_, &con_, mjFONTSCALE_150);
		
		cam_.type = mjCAMERA_FREE;  // 使用自由相機模式（避免跟模型綁定）
		cam_.lookat[0] = 0.0;
		cam_.lookat[1] = 0.0;
		cam_.lookat[2] = 2.0;   // 高度視角，看起來會稍微往下看
		cam_.distance = 15.0;
		cam_.azimuth = 90;   // 左右旋轉角
		cam_.elevation = -10; // 上下旋轉角
		
		// -----------------------------------
		// 控制與感測器 ID
		// -----------------------------------
		cart_motor_id = mj_name2id(m_, mjOBJ_ACTUATOR, "cart_motor");
		cart_pos_id   = mj_name2id(m_, mjOBJ_SENSOR, "cart_pos");
		cart_vel_id   = mj_name2id(m_, mjOBJ_SENSOR, "cart_vel");
		pole_pos_id   = mj_name2id(m_, mjOBJ_SENSOR, "pole_pos");
		pole_vel_id   = mj_name2id(m_, mjOBJ_SENSOR, "pole_vel");
		
		return true;
	}
	
	void State_subscribe(const my_package::msg::Cartpole2::SharedPtr msg)
	{
		cart_cmd = msg->cart_motor;
	}
	
	void main_loop(std::shared_ptr<CartPoleEnv> node_ptr)
	{
		// -----------------------------------
		// 主迴圈
		// -----------------------------------
		while (!glfwWindowShouldClose(window_))
		{
			// 處理 ROS callback（單執行緒）
			rclcpp::spin_some(node_ptr);
			
			mjtNum simstart = d_->time;
			while (d_->time - simstart < 1.0/60.0) {
				d_->ctrl[cart_motor_id] = cart_cmd;
				// 模擬一步
				mj_step(m_, d_);
			}
			
			// 發布感測器狀態
			my_package::msg::Cartpole1 state_msg;
			state_msg.cart_pos = d_->sensordata[m_->sensor_adr[cart_pos_id]];
			state_msg.cart_vel = d_->sensordata[m_->sensor_adr[cart_vel_id]];
			state_msg.pole_pos = d_->sensordata[m_->sensor_adr[pole_pos_id]];
			state_msg.pole_vel = d_->sensordata[m_->sensor_adr[pole_vel_id]];
			
			publisher_->publish(state_msg);
			
			mjrRect viewport = {0,0,0,0};
			glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
			// 更新畫面
			mjv_updateScene(m_, d_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
			mjr_render(viewport, &scn_, &con_);
			glfwSwapBuffers(window_);
			glfwPollEvents();
		}
		
		cleanup();
	}
	
	void cleanup()
	{
		// -----------------------------------
		// 清理
		// -----------------------------------
		mj_deleteData(d_);
		mj_deleteModel(m_);
		mjv_freeScene(&scn_);
		mjr_freeContext(&con_);
		if (window_) glfwDestroyWindow(window_);
		glfwTerminate();
	}

private:
	mjModel* m_ = nullptr;
	mjData* d_ = nullptr;
	GLFWwindow* window_ = nullptr;
	mjvCamera cam_;
	mjvOption opt_;
	mjvScene scn_;
	mjrContext con_;
	
	int cart_motor_id, cart_pos_id, cart_vel_id, pole_pos_id, pole_vel_id;
	
	double cart_cmd = 0.0;
	
	rclcpp::Publisher<my_package::msg::Cartpole1>::SharedPtr publisher_;
	rclcpp::Subscription<my_package::msg::Cartpole2>::SharedPtr subscriber_;
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<CartPoleEnv>();
	
	// 初始化 MuJoCo
	if (!node->init_mujoco()) {
		return 1;
	}
	
	node->main_loop(node);
	rclcpp::shutdown();
	
	return 0;
}

