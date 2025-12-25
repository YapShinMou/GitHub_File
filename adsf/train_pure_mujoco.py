
import torch
import numpy as np
import mujoco
import mujoco.viewer
import yaml  # <--- 1. 新增這個 import
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
#from rsl_rl.algorithms import PPO
#from rsl_rl.env import VecEnv

class MuJoCoCPUEnv:
   def __init__(self, xml_path, num_envs=1, device='cuda:0', render=False):# <--- 2. 新增 render 參數
      """
      這就是你要的單純環境：
      內部用 Numpy 運算 MuJoCo，外部轉成 Tensor 給 rsl_rl
      """
      self.num_envs = num_envs
      self.device = device
      self.render_mode = render
        
      # 1. 載入 MuJoCo 模型
      self.model = mujoco.MjModel.from_xml_path(xml_path)
      # 為了平行化，我們建立 N 個 data 實例 (這是 CPU 模擬的瓶頸)
      self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

      # --- [關鍵修正] 新增 rsl_rl 需要的屬性 ---
      self.unwrapped = self  # 讓 env.unwrapped 指向自己
      # 設定控制頻率 (Control DT)。
      # 假設 MuJoCo 物理步長是 0.002s，如果你每 10 次物理步進做一次動作，這裡就是 0.02s
      # 為了先讓程式跑起來，我們先設一個標準值 0.02 (50Hz)
      self.step_dt = 0.02  
      # -------------------------------------
        
      # 2. 定義維度 (需依照你的 G1 模型修改)
      self.num_obs = self.model.nq + self.model.nv # 範例：位置 + 速度
      self.num_actions = self.model.nu             # 馬達數量 (G1 12dof = 12)
      self.max_episode_length = 1000               # 如果沒有特權觀察，設為 None
        
      # 3. 預先分配記憶體 (Buffer)
      self.obs_buf = torch.zeros(num_envs, self.num_obs, device=device)
      self.rew_buf = torch.zeros(num_envs, device=device)
      self.reset_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)
      self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
      self.extras = {} # 用來存額外資訊

      # 4. 必須定義 cfg，雖然這裡是空的，但父類別可能會檢查
      self.cfg = {}

      # <--- 3. 初始化 Viewer (只顯示第 0 個環境，避免視窗混亂)
      self.viewer = None
      if self.render_mode:
         # 這裡我們只將第一個環境 (datas[0]) 綁定到視窗
         self.viewer = mujoco.viewer.launch_passive(self.model, self.datas[0])

   def step(self, actions):
      """
      rsl_rl 傳入的是 GPU Tensor 的 actions
      """
      # --- [關鍵步驟] Tensor (GPU) -> Numpy (CPU) ---
      # 為了單純，這裡要把 actions 搬回 CPU
      actions_np = actions.detach().cpu().numpy()
        
      # --- 執行 MuJoCo 物理模擬 (純 CPU) ---
      # 這裡用最暴力的 For Loop，之後可以用 multiprocess 優化
      for i in range(self.num_envs):
         # 1. 設定控制訊號
         self.datas[i].ctrl[:] = actions_np[i]
         # 2. 物理步進
         mujoco.mj_step(self.model, self.datas[i])
         # 3. 計算 Reward (範例：簡單的活著就有分)
         self.rew_buf[i] = 1.0 
            
         # 4. 檢查 Reset (範例：倒地就重置)
         # 這裡需要你寫判斷 G1 是否跌倒的邏輯
         # if is_fallen: self.reset_buf[i] = 1
            
         self.episode_length_buf[i] += 1
            
         # 5. 處理重置
         if self.episode_length_buf[i] > self.max_episode_length: # 假設 1000 步結束
            self.reset_buf[i] = True
         else:
            self.reset_buf[i] = False

      # 執行實際的 Reset 操作
      env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
      if len(env_ids) > 0:
         self.reset_idx(env_ids)

      # <--- 4. 更新畫面 (Sync Viewer)
      if self.viewer is not None:
         # 檢查視窗是否還開著
         if self.viewer.is_running():
            self.viewer.sync()
         else:
            # 如果使用者按了 ESC 關閉視窗，我們可以選擇關閉渲染或結束程式
            pass

      # --- 更新觀察值 ---
      self._update_obs()

      # 取得符合 TensorDict 格式的 obs
      obs_dict = self.get_observations()

      # 準備 extras (rsl_rl 會紀錄 time_outs)
      self.extras["time_outs"] = self.reset_buf.clone() # 簡單假設 reset 就是 timeout
      self.extras["log"] = {}

      # <--- 2. 修改這裡：Step 只能回傳 4 個值 (Obs, Rew, Done, Extras)
      # 注意：不需要回傳 privileged_obs (那個 None)，因為它已經包在 obs_dict 裡了
      return obs_dict, self.rew_buf, self.reset_buf, self.extras

   def reset(self):
      self.reset_idx(torch.arange(self.num_envs, device=self.device))
      self._update_obs()
      return self.get_observations(), None

   def reset_idx(self, env_ids):
      # 將指定的環境重置回初始狀態
      ids_np = env_ids.cpu().numpy()
      for i in ids_np:
         mujoco.mj_resetData(self.model, self.datas[i])
         self.episode_length_buf[i] = 0
         self.reset_buf[i] = False

   def _update_obs(self):
      # --- [關鍵步驟] Numpy (CPU) -> Tensor (GPU) ---
      # 收集所有環境的數據
      obs_list = []
      for i in range(self.num_envs):
         # 簡單範例：觀察值 = [位置, 速度]
         qpos = self.datas[i].qpos.flat[:]
         qvel = self.datas[i].qvel.flat[:]
         obs_list.append(np.concatenate([qpos, qvel]))
        
      # 一次性轉成 GPU Tensor
      self.obs_buf = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)

   # rsl_rl 需要這個函數來取得觀察值
   def get_observations(self)-> TensorDict:
      # 假設 self.obs_buf 是你算出來的 (N, obs_dim) Tensor
      return TensorDict({
         "policy": self.obs_buf,  # Actor 看的
         "critic": self.obs_buf   # Critic 看的 (通常一樣，除非有致盲)
      }, batch_size=self.num_envs)
        
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
            
         # [關鍵修正] 檢查最外層是否有 'runner' key
         # 如果有，我們就只取 runner 裡面的內容，這樣結構才會變平的
         if 'runner' in full_config:
            ppo_config = full_config['runner']
         else:
            ppo_config = full_config

      print(f"成功載入設定檔: {yaml_path}")

   except FileNotFoundError:
      print(f"錯誤: 找不到 {yaml_path}，請確認檔案在同一目錄下")
      exit()

   # 3. 建立環境
   # 從 YAML 讀取參數來決定環境設定 (選用，讓你的 yaml 更強大)
   # 這裡我們手動指定，因為 env 的參數通常不寫在 PPO config 裡
   env = MuJoCoCPUEnv(xml_path, num_envs=1, device='cuda:0', render=False)

   # 4. 動態調整參數 (Optional)
   # 如果你的環境數量很少 (例如測試時 num_envs=1)，必須強制修改 batch size
   # 不然 PPO 會因為 batch 太小而報錯
   if env.num_envs < ppo_config["algorithm"]["num_mini_batches"]:
      print(f"警告: 環境數量 ({env.num_envs}) 小於 mini_batches，自動修正為 1")
      ppo_config["algorithm"]["num_mini_batches"] = 1

   # 4. 初始化 rsl_rl Runner
   runner = OnPolicyRunner(env, ppo_config, log_dir='./logs', device='cuda:0')
    
   # 5. 開始訓練！
   print("Start Training...")
   runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)
