
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import math
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
#from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv

class MuJoCoCPUEnv(VecEnv):
   def __init__(self, xml_path, num_envs=1, device='cuda', render=False): #render 渲染
      super().__init__()

      self.model = mujoco.MjModel.from_xml_path(xml_path)
      self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

      self.num_envs = num_envs
      self.num_actions = 12
      self.max_episode_length = 1000 # int 或 torch.Tensor(num_envs)
      self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long) # Buffer for current episode lengths
      self.device = device
      self.render = render
      self.cfg = {} # 沒在使用，dict 或 object，Configuration object.
      self.extras = {} # time_outs, log

      self.unwrapped = self # rsl_rl/modules/rnd.py 會用到 env.unwrapped.step_dt
      self.step_dt = 0.02 # 控制頻率 rsl_rl/modules/rnd.py 會用到 env.unwrapped.step_dt
      
      # 2. 定義維度 (需依照你的 G1 模型修改)
      self.num_obs = self.model.nq + self.model.nv # 範例：位置 + 速度
      
      # 3. 預先分配記憶體 (Buffer)
      self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device) # 狀態，好像
      self.rewards = torch.zeros(num_envs, device=device) # 獎勵，好像
      self.dones = torch.zeros(num_envs, device=device, dtype=torch.bool)
      self.falls = torch.zeros(num_envs, device=device, dtype=torch.bool)
      
      self.viewer = None
      if self.render:
         self.viewer = mujoco.viewer.launch_passive(self.model, self.datas[0]) # 只顯示第 0 個環境

   def step(self, actions): # actions (num_envs, num_actions)
      actions_np = actions.detach().cpu().numpy()
      
      for i in range(self.num_envs):
         self.datas[i].ctrl[0:12] = actions_np[i] # 1. 設定控制訊號
         
         mujoco.mj_step(self.model, self.datas[i]) # 2. 物理步進
         
         self.episode_length_buf[i] += 1
         
         if self.episode_length_buf[i] > self.max_episode_length:
            self.dones[i] = True
         #else:
         #   self.dones[i] = False

         gravity_orientation = self.get_gravity_orientation(self.datas[i].qpos[3:7])
         yaw = self.get_euler_angle(self.datas[i].qpos[3:7])

         if gravity_orientation[2]<-0.75 or self.datas[i].qpos[2]>0.6 or abs(yaw)<1.57:
            self.rewards[i] = self.get_reward(gravity_orientation[2], self.datas[i].qpos[0:3], self.datas[i].qvel[0:3], yaw)
         else:
            self.rewards[i] = 0
            self.dones[i] = True
            self.falls[i] = True

      # 執行實際的 Reset 操作
      env_ids = self.dones.nonzero(as_tuple=False).flatten()
      if len(env_ids) > 0:
         self.reset_idx(env_ids)

      # 更新畫面 (Sync Viewer)
      if self.viewer is not None:
         # 檢查視窗是否還開著
         if self.viewer.is_running():
            self.viewer.sync()
         else:
            # 如果使用者按了 ESC 關閉視窗，我們可以選擇關閉渲染或結束程式
            pass

      # --- 更新觀察值 --- 取得符合 TensorDict 格式的 obs
      observations = self.get_observations()

      # 準備 extras (rsl_rl 會紀錄 time_outs)
      self.extras["time_outs"] = self.falls.clone() # 簡單假設 reset 就是 timeout self.falls.clone()
      self.extras["log"] = {}

      # 注意：不需要回傳 privileged_obs (那個 None)，因為它已經包在 observations 裡了
      return observations, self.rewards, self.dones, self.extras

   def get_gravity_orientation(self, quaternion):
      qw = quaternion[0]
      qx = quaternion[1]
      qy = quaternion[2]
      qz = quaternion[3]

      gravity_orientation = np.zeros(3)
      gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
      gravity_orientation[1] = -2 * (qz * qy + qw * qx)
      gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
      return gravity_orientation

   def get_euler_angle(self, quaternion):
      qw = quaternion[0]
      qx = quaternion[1]
      qy = quaternion[2]
      qz = quaternion[3]

      yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
      return yaw

   def get_reward(self, z_gravity_orientation, pos, vel, yaw):
      orientation_reward = (- z_gravity_orientation - 0.75)/0.25 # (angle - 0.75)/(1 - 0.75)
      hight_reward = max(1, (pos[2] - 0.6)/0.15) # (hight - 0.6)/(0.75 - 0.6)
      xvel_reward = math.exp(-abs(vel[0] - 2))
      yvel_reward = math.exp(-abs(vel[1]))
      yaw_reward = math.exp(-abs(yaw))
      rewards = orientation_reward * hight_reward * xvel_reward * yvel_reward * yaw_reward
      return rewards

   def reset_idx(self, env_ids):
      # 將指定的環境重置回初始狀態
      ids_np = env_ids.cpu().numpy()
      for i in ids_np:
         mujoco.mj_resetData(self.model, self.datas[i])
         mujoco.mj_forward(self.model, self.datas[i])
         self.episode_length_buf[i] = 0
         self.dones[i] = False
         self.falls[i] = False
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")
         print(f"=============================================")

   # rsl_rl 需要這個函數來取得觀察值
   def get_observations(self) -> TensorDict:
      # 收集所有環境的數據
      obs_list = []
      for i in range(self.num_envs):
         # 簡單範例：觀察值 = [位置, 速度]
         qpos = self.datas[i].qpos.flat[:]
         qvel = self.datas[i].qvel.flat[:]
         obs_list.append(np.concatenate([qpos, qvel]))
      self.obs_buf = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device) # 一次性轉成 GPU Tensor

      # 假設 self.obs_buf 是你算出來的 (num_envs, num_obs) Tensor
      return TensorDict({
         "policy": self.obs_buf,  # Actor 看的
         "critic": self.obs_buf   # Critic 看的 (通常一樣，除非有致盲)
      }, batch_size=self.num_envs)
   
   #def reset(self):
   #   self.reset_idx(torch.arange(self.num_envs, device=self.device))
   #   #self._update_obs()
   #   return self.get_observations(), None

   # rsl_rl 需要這個函數來取得特權觀察值 (若無則回傳 obs)
   def get_privileged_observations(self):
      return None

   # <--- 5. 記得解構時關閉視窗
   def __del__(self):
      if self.viewer is not None:
         self.viewer.close()


if __name__ == '__main__':
   # 1. 設定你的 G1 XML 路徑
   xml_path = "../unitree_rl_gym/resources/robots/g1_description/scene.xml" # 請確保你有這個檔案
   
   # 2. 載入 YAML 設定檔
   yaml_path = "config/example_config.yaml"  # <--- 指向你剛剛建立的檔案
   try:
      with open(yaml_path, 'r') as f:
         # 先讀取完整的 YAML
         full_config = yaml.safe_load(f)
         ppo_config = full_config['runner']
      print(f"成功載入設定檔: {yaml_path}")
   except:
      print(f"錯誤: 找不到 {yaml_path}")
      exit()

   # 3. 建立環境
   env = MuJoCoCPUEnv(xml_path, num_envs=1, device='cuda', render=True)

   # 4. 動態調整參數 (Optional)
   # 如果你的環境數量很少 (例如測試時 num_envs=1)，必須強制修改 batch size
   # 不然 PPO 會因為 batch 太小而報錯
   if env.num_envs < ppo_config["algorithm"]["num_mini_batches"]:
      print(f"警告: 環境數量 ({env.num_envs}) 小於 mini_batches，自動修正為 1")
      ppo_config["algorithm"]["num_mini_batches"] = 1

   # 4. 初始化 rsl_rl Runner
   runner = OnPolicyRunner(env, ppo_config, log_dir='./logs', device='cuda')
   
   # 5. 開始訓練！
   print("Start Training...")
   runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)
