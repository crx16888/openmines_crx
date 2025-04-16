# 系统给的
## 运行仿真
openmines run -f C:\Users\95718\Desktop\vscode\Mine_Dispatcher\openmines\openmines\src\conf\north_pit_mine_short.json
## 可视化运行结果
openmines visualize -f "C:\Users\95718\Desktop\vscode\Mine_Dispatcher\openmines\results\MINE-NorthOpenPitMine-ALGO-FixedGroupDispatcher-TIME-2025-04-15 10-27-16.json"
图表有各种算法可视化对比

# 我自己写的
openmines run -f C:\Users\95718\Desktop\vscode\Mine_Dispatcher\openmines\openmines\src\conf\north_pit_mine_test1.json

openmines visualize -f "C:\Users\95718\Desktop\vscode\Mine_Dispatcher\openmines\results\MINE-NorthOpenPitMine-ALGO-OptimizeDispatcher-TIME-2025-04-15 11-24-27.json"

# 强化学习方法
1. python test_ppo.py --env_config ../../openmines/src/conf/north_pit_mine.json --num_updates 1000 --num_processes 7 --max_steps 1000
2. openmines run -f C:/path/to/your/config.json
3. openmines visualize -f "path/to/result/MINE-xxx-ALGO-PPODispatcher-TIME-xxx.json"

1. python -m openmines.test.cleanrl.ppo_single_net --mine_config openmines/src/conf/north_pit_mine.json --total_timesteps 5000000 --checkpoint_dir ./checkpoints --exp_name my_ppo_model
2. python -m openmines.src.cli.run --config openmines/src/conf/north_pit_mine.json --dispatcher ppo --model_path ./checkpoints/my_ppo_model/best_model_*.pt
