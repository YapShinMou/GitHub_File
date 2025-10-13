#include "common/motor_crc_hg.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "unitree_hg/msg/motor_cmd.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>

// libtorch
#include <torch/torch.h>

// ==================== 超參數 ====================
constexpr int EPISODES = 300;
constexpr float GAMMA = 0.9;
constexpr float LR = 1e-3f; //learning rate
constexpr int BATCH_SIZE = 64;
constexpr int MEMORY_SIZE = 1000;
constexpr float EPS_START = 1.0f;
constexpr float EPS_END = 0.01f;
constexpr float EPS_DECAY = 10000.0f;
constexpr int TARGET_UPDATE = 10;

// 環境維度（你在程式中用到 motor[18].q, dq）
const int state_dim = 2;
const int action_dim = 2;

// ==================== DQN Implement ====================
struct DQNImpl : torch::nn::Module {
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
	DQNImpl(int64_t sdim, int64_t adim) {
		fc1 = register_module("fc1", torch::nn::Linear(sdim, 128));
		fc2 = register_module("fc2", torch::nn::Linear(128, 128));
		fc3 = register_module("fc3", torch::nn::Linear(128, adim));
	}
	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);
		return x;
	}
};
TORCH_MODULE(DQN); // defines DQN as module holder (shared_ptr<DQNImpl>)

// ==================== Replay Buffer ========================
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		std::vector<float> state;
		int action;
		float reward;
		std::vector<float> next_state;
		bool done;
	};
	
	void push(const std::vector<float>& s, int a, float r, const std::vector<float>& s2, bool d) {
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({s, a, r, s2, d});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::uniform_int_distribution<int> dist(0, buffer_.size() - 1); //dist(rng) 為0到buffer_.size() - 1隨機整數
		for (int i=0; i<BATCH_SIZE; i++) {
			batch.push_back(buffer_[dist(rng)]);
		}
		return batch;
	}
	
	int size() const { return buffer_.size(); }
	
private:
	std::deque<Experience> buffer_; //宣告一個名稱叫做 buffer_ 的變數，型態為 std::deque<Experience>，頭尾兩端插入及刪除十分快速的陣列，元素型態為 Experience
	std::mt19937 rng{std::random_device{}()}; //建立一個亂數引擎
};

// ==================== Node with RL loop ====================
class LowLevelCmdSender : public rclcpp::Node {
public:
	LowLevelCmdSender() : Node("yap_g1_rl_node")
	{
		// device
		device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
		RCLCPP_INFO(this->get_logger(), "Using device: %s", device.is_cuda() ? "CUDA" : "CPU");
		
		// create nets
		policy_net = DQN(state_dim, action_dim);
		policy_net->to(device);
		target_net = DQN(state_dim, action_dim);
		target_net->to(device);
		torch::save(policy_net, "tmp.pt");
		torch::load(target_net, "tmp.pt");
		target_net->eval(); //設定成推論模式
		
		optimizer = std::make_unique<torch::optim::Adam>(policy_net->parameters(), torch::optim::AdamOptions(LR));
		
		// init low command defaults
		for (int i=0;i<29;i++) {
			low_command_.motor_cmd[i].mode = 1;
			low_command_.motor_cmd[i].tau = 0;
			low_command_.motor_cmd[i].q = 0;
			low_command_.motor_cmd[i].dq = 0;
			low_command_.motor_cmd[i].kp = 200;
			low_command_.motor_cmd[i].kd = 1.5;
		}
		low_command_.motor_cmd[15].q = -1;
		low_command_.mode_pr = 0;
		
		// ROS2 pub/sub/timer
		lowstate_subscriber_ = this->create_subscription<unitree_hg::msg::LowState>("lowstate", rclcpp::QoS(10), std::bind(&LowLevelCmdSender::LowStateHandler, this,
		 std::placeholders::_1));
		
		lowcmd_publisher_ = this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", rclcpp::QoS(10));
		timer_ = this->create_wall_timer(std::chrono::milliseconds(2), std::bind(&LowLevelCmdSender::Control, this));
		
		// start background thread
		rl_thread_ = std::thread(&LowLevelCmdSender::MainLoop, this);
	}
	
	~LowLevelCmdSender() override {
		shutdown_flag_ = true;
		if (rl_thread_.joinable()) rl_thread_.join();
	}
	
private:
	// ---------- ROS callbacks ----------
	void Control() {
		std::lock_guard<std::mutex> lk(mtx_);
		low_command_.mode_machine = mode_machine_;
		get_crc(low_command_);
		lowcmd_publisher_->publish(low_command_);
	}
	
	void LowStateHandler(const unitree_hg::msg::LowState::SharedPtr message) {
		std::lock_guard<std::mutex> lk(mtx_);
		mode_machine_ = static_cast<int>(message->mode_machine);
		imu_ = message->imu_state;
		for (int i=0;i<29;i++) motor_[i] = message->motor_state[i];
	}
	
	std::vector<float> get_state() {
		return {motor_[18].q, motor_[18].dq};
	}
	
	float get_reward(const std::vector<float>& state, const std::vector<float>& next_state, bool done) {
		float q = next_state[0];
		if (q > -0.15 && q < 0.15) return 1;
		return 0;
	}
	
	int select_action(const std::vector<float>& state, float epsilon) {
		std::uniform_real_distribution<float> maxQ_or_random(0.0f,1.0f);
		if (maxQ_or_random(rng) < epsilon) {
			std::uniform_int_distribution<int> random_action(0, action_dim-1);
			return random_action(rng);
		} else {
			// forward
			torch::Tensor select_action_state = torch::from_blob(const_cast<float*>(state.data()), {1, (long)state.size()}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone();
			auto select_action_qvalue = policy_net->forward(select_action_state);
			auto indx = select_action_qvalue.argmax(1).item<int64_t>();
			return static_cast<int>(indx);
		}
	}
	
	// ---------- tensor helper: convert vector<vector<float>> -> tensor ----------
	torch::Tensor batch_to_tensor(const std::vector<std::vector<float>>& vv) {
		if (vv.empty()) return torch::empty({0});
		size_t N = vv.size();
		size_t D = vv[0].size();
		std::vector<float> flat;
		flat.reserve(N*D);
		for (const auto &r : vv) {
			flat.insert(flat.end(), r.begin(), r.end());
		}
		auto t = torch::from_blob(flat.data(), {(long)N, (long)D}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device);
		return t;
	}
	
	// ---------- Main RL loop (runs in background thread) ----------
	void MainLoop() {
		for (int episode = 0; episode < EPISODES && !shutdown_flag_; ++episode) {
			// reset / init for episode
			{
				std::lock_guard<std::mutex> lk(mtx_); //讀寫變數避免資料競爭
				low_command_.motor_cmd[18].tau = 0;
				low_command_.motor_cmd[18].q = 0;
				low_command_.motor_cmd[18].kp = 10;
				low_command_.motor_cmd[18].kd = 0.2;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
			{
				std::lock_guard<std::mutex> lk(mtx_);
				low_command_.motor_cmd[18].kp = 0;
				low_command_.motor_cmd[18].kd = 0;
			}
			
			float total_reward = 0.0f;
			bool done = false;
			int step_count = 0;
			
			std::vector<float> state;
			{
				std::lock_guard<std::mutex> lk(mtx_);
				state = get_state();
			}
			
			// episode loop
			while (!done && !shutdown_flag_) {
				// compute epsilon
				float epsilon = EPS_END + (EPS_START - EPS_END) * std::exp(-1.0f * float(steps_done) / EPS_DECAY);
				++steps_done;
				
				// choose action
				int action = select_action(state, epsilon);
				
				// apply action: here set motor 18 torque as simple discrete actions
				{
					std::lock_guard<std::mutex> lk(mtx_);
					if (action == 0) low_command_.motor_cmd[18].tau = -3;
					else low_command_.motor_cmd[18].tau = 3;
				}
				
				// wait small time for environment to step
				std::this_thread::sleep_for(std::chrono::milliseconds(4));
				
				// read next_state and reward
				std::vector<float> next_state;
				{
					std::lock_guard<std::mutex> lk(mtx_);
					next_state = get_state();
				}
				
				float reward = get_reward(state, next_state, done);
				total_reward += reward;
				
				step_count++;
				// simple termination condition for demo (you can replace)
				if (step_count > 200) done = true;
				
				// store to memory
				memory.push(state, action, reward, next_state, done);
				
				// update state
				state = next_state;
				
				// train if enough samples
				if (memory.size() >= BATCH_SIZE) {
					auto batch = memory.sample();
					// prepare minibatch
					std::vector<std::vector<float>> states, next_states;
					std::vector<int64_t> actions;
					std::vector<float> rewards;
					std::vector<float> dones;
					states.reserve(batch.size()); //預留位置
					next_states.reserve(batch.size());
					actions.reserve(batch.size());
					rewards.reserve(batch.size());
					dones.reserve(batch.size());
					
					for (auto &e : batch) { //對 batch 裡面的每一個元素，逐一取出來命名為 e
						states.push_back(e.state);
						next_states.push_back(e.next_state);
						actions.push_back(e.action);
						rewards.push_back(e.reward);
						dones.push_back(e.done ? 0.0f : 1.0f);
					}
					
					auto state_tensor = batch_to_tensor(states); // [B, state_dim]
					auto next_state_tensor = batch_to_tensor(next_states); // [B, state_dim]
					auto action_tensor = torch::from_blob(actions.data(), {(long)actions.size()}, torch::kInt64).clone().to(device);
					auto reward_tensor = torch::from_blob(rewards.data(), {(long)rewards.size()}, torch::kFloat32).clone().to(device);
					auto done_tensor = torch::from_blob(dones.data(), {(long)dones.size()}, torch::kFloat32).clone().to(device);
					
					// Q values for taken actions
					auto q_values_all = policy_net->forward(state_tensor); // [B, A]
					auto q_values = q_values_all.gather(1, action_tensor.unsqueeze(1)).squeeze(1); // [B]
					
					// next Q values (target)
					auto next_q_all = target_net->forward(next_state_tensor);
					auto next_q_values = std::get<0>(next_q_all.max(1)); // [B]
					
					auto expected = reward_tensor + GAMMA * next_q_values * done_tensor;
					
					auto loss = torch::mse_loss(q_values, expected.detach());
					
					optimizer->zero_grad();
					loss.backward();
					optimizer->step();
				} // end update neural network
			} // end episode loop
			
			// update target net
			if (episode % TARGET_UPDATE == 0) {
				torch::save(policy_net, "tmp.pt");
				torch::load(target_net, "tmp.pt");
				std::cout << "Episode " << episode << " update target net" << std::endl;
			}
			RCLCPP_INFO(this->get_logger(), "Episode %d finished, total reward: %.3f", episode, total_reward);
		} // end training loop
		
		// save policy
		try { //嘗試執行可能會失敗的操作
			torch::save(policy_net, "policy_net.pt");
			RCLCPP_INFO(this->get_logger(), "Saved policy_net.pt");
		} catch (const std::exception &e) { //如果失敗，讀取錯誤訊息
			RCLCPP_WARN(this->get_logger(), "Failed to save model: %s", e.what());
		}
		
		rclcpp::shutdown();
	}
	
	
	// ---------- members ----------
	DQN policy_net{nullptr}; //宣告一個空的模型
	DQN target_net{nullptr};
	std::unique_ptr<torch::optim::Adam> optimizer;
	
	ReplayBuffer memory;
	std::mt19937 rng{std::random_device{}()};
	
	// ROS
	rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr lowstate_subscriber_;
	rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;
	rclcpp::TimerBase::SharedPtr timer_;
	
	unitree_hg::msg::LowCmd low_command_;
	unitree_hg::msg::IMUState imu_;
	std::array<unitree_hg::msg::MotorState, 29> motor_;
	int mode_machine_ = 0;
	
	// RL internals
	torch::Device device{torch::kCPU};
	int steps_done = 0;
	std::thread rl_thread_; //建立新執行緒
	std::atomic<bool> shutdown_flag_ = false; //讀寫變數避免資料競爭
	std::mutex mtx_; //上鎖程式區塊
};

// ---------- main ----------
int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<LowLevelCmdSender>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}

