//#include <string>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// ==================== 超參數 ====================
constexpr int EPISODES = 1000;
constexpr float GAMMA = 0.9;
constexpr float LR = 1e-3f; //learning rate
constexpr int BATCH_SIZE = 64;
constexpr int MEMORY_SIZE = 1000;
constexpr float EPS_START = 1.0f;
constexpr float EPS_END = 0.01f;
constexpr float EPS_DECAY = 200000.0f;
constexpr int TARGET_UPDATE = 10;

// 全域變數（參考官方）
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

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

// ==================== Replay Buffer ======================== 沒問題+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		std::vector<double> state;
		int action;
		float reward;
		std::vector<double> next_state;
		bool done;
	};
	
	void push(const std::vector<double>& s, int a, float r, const std::vector<double>& s2, bool d) {
		if (buffer_.size() >= MEMORY_SIZE) buffer_.pop_front();
		buffer_.push_back({s, a, r, s2, d});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::uniform_int_distribution<int> dist(0, buffer_.size() - 1); //dist(rng) 為0到buffer_.size() - 1隨機整數
		for (int i=0; i<BATCH_SIZE; i++) {
			batch.push_back(buffer_[dist(rng)]); // 有可能重複
		}
		return batch;
	}
	
	int size() const { return buffer_.size(); }
	
private:
	std::deque<Experience> buffer_; //宣告一個名稱叫做 buffer_ 的變數，型態為 std::deque<Experience>，頭尾兩端插入及刪除十分快速的陣列，元素型態為 Experience
	std::mt19937 rng{std::random_device{}()}; //建立一個亂數引擎
};

// -------------------------------------------
class ClassDQN {
public:
	DQN policy_net{nullptr}; //宣告一個空的模型
	DQN target_net{nullptr};
	std::unique_ptr<torch::optim::Adam> optimizer;
	torch::Device device{torch::kCPU};
	
	void initclass() {
		// device
		if (torch::cuda::is_available()) {
			device = torch::Device(torch::kCUDA);
		} else {
			device = torch::Device(torch::kCPU);
		}
		
		if (device.is_cuda()) {
			std::cout << "Using device: CUDA" << std::endl;
		} else {
			std::cout << "Using device: CPU" << std::endl;
		}
		
		// create nets
		policy_net = DQN(state_dim, action_dim);
		policy_net->to(device);
		target_net = DQN(state_dim, action_dim);
		target_net->to(device);
		torch::save(policy_net, "tmp.pt");
		torch::load(target_net, "tmp.pt");
		target_net->eval(); //設定成推論模式
		
		optimizer = std::make_unique<torch::optim::Adam>(policy_net->parameters(), torch::optim::AdamOptions(LR));
	}
	
	float get_reward(const std::vector<double>& state, const std::vector<double>& next_state, bool done) {
		if (next_state[2] > -0.5 && next_state[2] < 0.5) {
			return 0.2 - 0.4*abs(next_state[2]);
		} else {
			return 0;
		}
	}
	
	int select_action(const std::vector<double>& state, float epsilon) { ///////////////////////<double>&
		std::uniform_real_distribution<float> maxQ_or_random(0.0f,1.0f);
		if (maxQ_or_random(rng) < epsilon) {
			std::uniform_int_distribution<int> random_action(0, action_dim-1);
			return random_action(rng);
		} else {
			// forward
			torch::Tensor select_action_state = torch::from_blob(const_cast<double*>(state.data()), {1, (long)state.size()},  torch::TensorOptions().dtype(torch::kFloat32)).to(device).clone(); //////////////////////////////////////////////////////////////////////////////////////double*
			auto select_action_qvalue = policy_net->forward(select_action_state);
			auto indx = select_action_qvalue.argmax(1).item<int64_t>();
			return static_cast<int>(indx);
		}
	}
	
	// ---------- tensor helper: convert vector<vector<float>> -> tensor ----------
	torch::Tensor batch_to_tensor(const std::vector<std::vector<double>>& vv) {
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
	
	// ----------------------------------DL
	void deep_learning(auto batch)
	{
		// prepare minibatch
		std::vector<std::vector<double>> states, next_states;
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
	
private:
	const int state_dim = 4;
	const int action_dim = 7;
	
	std::mt19937 rng{std::random_device{}()};
};

// ---------- main ----------
int main()
{
	// -----------------------------------
	// 載入模型與資料
	// -----------------------------------
	char error[1000];
	std::string xml_file = std::string(MUJOCO_MODEL_DIR) + "/cart_pole.xml";
	mjModel* m = mj_loadXML(xml_file.c_str(), nullptr, error, 1000);
	if (!m) {
		std::cerr << "Failed to load XML: " << error << std::endl;
		return 1;
	}
	mjData* d = mj_makeData(m);
	
	// -----------------------------------
	// 初始化 GLFW
	// -----------------------------------
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << "Failed to initialize GLFW" << std::endl;
		return 1;
	}
	
	GLFWwindow* window = glfwCreateWindow(1200, 700, "MuJoCo GUI", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		return 1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);  // 垂直同步
	
	// -----------------------------------
	// 初始化 MuJoCo 可視化
	// -----------------------------------
	mjv_defaultCamera(&cam);
	mjv_defaultOption(&opt);
	mjv_defaultScene(&scn);
	mjr_defaultContext(&con);
	
	mjv_makeScene(m, &scn, 1000);
	mjr_makeContext(m, &con, mjFONTSCALE_150);
	
	cam.type = mjCAMERA_FREE;  // 使用自由相機模式（避免跟模型綁定）
	cam.lookat[0] = 0.0;
	cam.lookat[1] = 0.0;
	cam.lookat[2] = 2.0;   // 高度視角，看起來會稍微往下看
	cam.distance = 15.0;
	cam.azimuth = 90;   // 左右旋轉角
	cam.elevation = -10; // 上下旋轉角
	
	// -----------------------------------
	// 控制與感測器 ID
	// -----------------------------------
	int cart_motor_id = mj_name2id(m, mjOBJ_ACTUATOR, "cart_motor");
	int cart_pos_id = mj_name2id(m, mjOBJ_SENSOR, "cart_pos");
	int cart_vel_id = mj_name2id(m, mjOBJ_SENSOR, "cart_vel");
	int pole_pos_id = mj_name2id(m, mjOBJ_SENSOR, "pole_pos");
	int pole_vel_id = mj_name2id(m, mjOBJ_SENSOR, "pole_vel");
	
	double cart_pos;
	double cart_vel;
	double pole_pos;
	double pole_vel;
	
	int steps_done = 0;
	ClassDQN DQNClass;
	DQNClass.initclass();
	ReplayBuffer memory;
	// -----------------------------------
	// 主迴圈
	// -----------------------------------
	for (int episode = 0; episode < EPISODES ; ++episode)
	{
		mj_resetData(m, d);
		mj_forward(m, d); 
		
		float epsilon = EPS_START;
		float total_reward = 0.0f;
		bool done = false;
		int step_count = 0;
		
		cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
		cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
		pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
		pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
		std::vector<double> state = {cart_pos, cart_vel, pole_pos, pole_vel};
		d->ctrl[cart_motor_id] = 0;
		
		// episode loop
		while (!done) {
			mjtNum simstart = d->time;
			while (d->time - simstart < 1.0/30) {
//				std::cout << "time: " << d->time << std::endl;
				
				// compute epsilon
				epsilon = EPS_END + (EPS_START - EPS_END) * std::exp(-1.0f * float(steps_done) / EPS_DECAY);
				steps_done = steps_done + 1;
				
				// choose action
				int action = DQNClass.select_action(state, epsilon);
				if (action == 0) {
					d->ctrl[cart_motor_id] = -900;
				} else if (action == 1) {
					d->ctrl[cart_motor_id] = -600;
				} else if (action == 2) {
					d->ctrl[cart_motor_id] = -300;
				} else if (action == 3) {
					d->ctrl[cart_motor_id] = 0;
				} else if (action == 4) {
					d->ctrl[cart_motor_id] = 300;
				} else if (action == 5) {
					d->ctrl[cart_motor_id] = 600;
				} else { 
					d->ctrl[cart_motor_id] = 900;
				}
				
				// 模擬一步
				mj_step(m, d);
				cart_pos = d->sensordata[m->sensor_adr[cart_pos_id]];
				cart_vel = d->sensordata[m->sensor_adr[cart_vel_id]];
				pole_pos = d->sensordata[m->sensor_adr[pole_pos_id]];
				pole_vel = d->sensordata[m->sensor_adr[pole_vel_id]];
				std::vector<double> next_state = {cart_pos, cart_vel, pole_pos, pole_vel};
				
				float reward = DQNClass.get_reward(state, next_state, done);
				total_reward += reward;
				
				step_count = step_count + 1;
				if (step_count>2000 || cart_pos>4.5 || cart_pos<-4.5 || pole_pos<-0.5 || pole_pos>0.5) done = true;
				
				// store to memory
				memory.push(state, action, reward, next_state, done);
				
				// update state
				state = next_state;
				
				// train if enough samples
				if (memory.size() >= BATCH_SIZE) {
					auto batch = memory.sample();
					DQNClass.deep_learning(batch);
				}
			
			}
			
			mjrRect viewport = {0, 0, 0, 0};
			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
			mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
			mjr_render(viewport, &scn, &con);
			glfwSwapBuffers(window);
			glfwPollEvents();
		} // end episode loop
		
		if (episode % TARGET_UPDATE == 0) {
			torch::save(DQNClass.policy_net, "tmp.pt");
			torch::load(DQNClass.target_net, "tmp.pt");
			std::cout << "Episode " << episode << " update target net" << std::endl;
			std::cout << "epsilon: " << epsilon << std::endl;
		}
		std::cout << "total time-steps: " << step_count << std::endl;
	} // end training loop
	
	// save policy
	try { //嘗試執行可能會失敗的操作
		torch::save(DQNClass.policy_net, "policy_net.pt");
		std::cout << "Saved policy_net.pt " << std::endl;
	} catch (const std::exception &e) { //如果失敗，讀取錯誤訊息
		std::cout << "Failed to save model" << e.what() << std::endl;
	}
	
	// -----------------------------------
	// 清理
	// -----------------------------------
	mj_deleteData(d);
	mj_deleteModel(m);
	
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);
	return 0;
}
	
	
	
	
	


