import subprocess
import time

# 训练脚本的路径
training_script_path = r"D:\COMP34111-group-project\agents\Group7\model_update_11.py"

# 迭代次数
num_iterations = 17

# 每次迭代之间的等待时间（秒）
wait_time = 2

for i in range(num_iterations):
    # 运行训练脚本
    print("Starting training iteration", i + 1)
    subprocess.run(['python', training_script_path])

    # 等待一段时间
    print(f"Waiting for {wait_time} seconds before next iteration...")
    time.sleep(wait_time)

print("Training completed.")