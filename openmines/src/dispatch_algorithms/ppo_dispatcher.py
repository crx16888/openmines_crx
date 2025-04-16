from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional

import torch
from openmines.src.dispatcher import BaseDispatcher
from openmines.src.mine import Mine
from openmines.src.truck import Truck

# 导入 rl_dispatch.py 中的 preprocess_observation 函数
from openmines.src.dispatch_algorithms.rl_dispatch import RLDispatcher

class PPODispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "PPODispatcher"
        self.model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "mine", "Mine-v1__ppo_single_net__s1__lr2.28e-03__e1.43e-02__g0.997__c0.20__l0.993__ep4__gr0.36__hs256__ns1400__ne50__mb4__rmreward_norm__t1740559741", "best_model_step19670000_tons10812.1_reward0.00.pt")
        self.device = self._get_device()  # 获取可用设备
        self.load_rl_model(self.model_path)
        self.rl_dispatcher_helper = RLDispatcher("NaiveDispatcher", reward_mode="dense")        
        self.max_sim_time = 240
        
    def _get_device(self):
        """
        确定使用的设备（CUDA/CPU）
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def load_rl_model(self, model_path: str):
        """
        Load an model for inference.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        from openmines.test.cleanrl.ppo_single_net import Agent, Args
        
        self.args = Args()
        self.agent = Agent(envs=-1, args=self.args, 
                         norm_path=os.path.join(os.path.dirname(__file__), "ppo_norm_params_dense.json"))
        # - envs=-1 表示是推理模式，不需要与环境交互
        # - args 包含了 PPO 算法的各种超参数设置
        # - norm_path 是对于原始的矿山数据预处理后的数据，保存在这个位置
        
        # 加载模型时指定设备映射
        state_dict = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(state_dict)
        self.agent.to(self.device)  # 确保模型在正确的设备上
        self.agent.eval()

    def give_init_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (initial loading).
        """
        return self._dispatch_action(truck, mine)

    def give_haul_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (hauling).
        """
        return self._dispatch_action(truck, mine)

    def give_back_order(self, truck: Truck, mine: Mine) -> int:
        """
        Given the current truck state and mine, choose an action (returning to charging or loading site).
        """
        return self._dispatch_action(truck, mine)

    def _dispatch_action(self, truck: Truck, mine: Mine) -> int:
        """
        Dispatch the truck to the next action based on model inference.
        """
        from openmines.src.utils.feature_processing import preprocess_observation 

        current_observation_raw = self._get_raw_observation(truck, mine) # 获取原始矿卡对象和矿场对象数据作为强化学习算法的输入
        processed_obs = torch.FloatTensor(
            preprocess_observation(current_observation_raw, self.max_sim_time) # 对数据预处理，将原始观察数据处理成强化学习算法可以使用的标准化特征向量
        ).to(self.device)  # 确保输入数据在正确的设备上
        
        with torch.no_grad():  # 推理时不需要梯度
            action, logprob, _, value, _ = self.agent.get_action_and_value( # 调用该方法进行决策
                processed_obs, sug_action=None
            )        

        return action # 返回决策结果动作序列

    def _get_raw_observation(self, truck: Truck, mine: Mine):
        """
        获取原始的、未经预处理的观察值，直接复用 RLDispatcher 中的 _get_observation 方法
        """
        # 调用 RLDispatcher 中的 _get_observation 方法
        # 输入是当前需要调度的矿卡和矿场对象，返回的是一个字典，包括： 具体见函数怎么写的
        # - truck_name : 卡车名称
        # - event_name : 当前事件类型（"init"/"haul"/"unhaul"）
        # - info : 矿场基本信息
        # - the_truck_status : 当前卡车状态
        # - target_status : 目标地点状态
        # - cur_road_status : 道路网络状态
        # - mine_status : 矿场 KPI 状态
        return self.rl_dispatcher_helper._get_observation(truck, mine) 

# Example usage (for testing - you'd integrate this into your simulation):
if __name__ == "__main__":
    # This is a placeholder for a Mine and Truck object - you need to create
    # actual instances of Mine and Truck as defined in your openmines simulation.
# - 不需要完整矿场模拟系统的情况下测试调度器
# - 提供最小化的数据结构来验证 PPODispatcher 的基本功能
# - 帮助开发人员快速测试代码而不需要设置完整的运行环境
    class MockLocation: # 模拟位置类
        def __init__(self, name):
            self.name = name
    class MockTruck: # 模拟矿卡类
        def __init__(self, name="Truck1", current_location_name="charging_site", truck_load=0, truck_capacity=40, truck_speed=40):
            self.name = name
            self.current_location = MockLocation(current_location_name)
            self.truck_load = truck_load
            self.truck_capacity = truck_capacity
            self.truck_speed = truck_speed
            self.truck_cycle_time = 0

        def get_status(self):
            return {} # Placeholder

    class MockMine: # 模拟矿场类：包括装载点、卸载点和环境对象
        def __init__(self):
            self.env = MockEnv()
            self.load_sites = [MockLocation("load_site_1"), MockLocation("load_site_2")]
            self.dump_sites = [MockLocation("dump_site_1"), MockLocation("dump_site_2")]

        def get_status(self):
            return {} # Placeholder
    class MockEnv: # 模拟环境类
        def __init__(self):
            self.now = 10.0


    dispatcher = PPODispatcher()
    mock_mine = MockMine()
    mock_truck = MockTruck()

    # Example of getting orders:
    init_order = dispatcher.give_init_order(mock_truck, mock_mine) # 输入数据结构需要是Truck类，需要给前面类继承Truck类来修复报错
    haul_order = dispatcher.give_haul_order(mock_truck, mock_mine)
    back_order = dispatcher.give_back_order(mock_truck, mock_mine)

    print(f"Init Order: {init_order}")
    print(f"Haul Order: {haul_order}")
    print(f"Back Order: {back_order}")