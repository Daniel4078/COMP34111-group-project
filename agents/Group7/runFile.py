import subprocess
import time

# 训练脚本的路径
training_script_path = r"model_11 with_other_agent_update.py"

# 迭代次数
num_iterations = 6*2

# 每次迭代之间的等待时间（秒）
wait_time = 0.1

for i in range(num_iterations):
    # 运行训练脚本
    print("Starting training iteration", i + 1)
    subprocess.run(['python', training_script_path]) #about 5 mins

    # 等待一段时间
    print(f"Waiting for {wait_time} seconds before next iteration...")
    time.sleep(wait_time)

print("Training completed.")