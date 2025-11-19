import os
import re

def rename_images_in_subfolders(root_folder):
    """
    遍历指定根文件夹下的所有子文件夹，
    并将子文件夹中的图片名称（如 'image_5.png'）重命名为4位补零的格式（'image_0005.png'）。
    """
    print(f"开始处理根文件夹: {root_folder}\n")

    # 检查根文件夹是否存在
    if not os.path.isdir(root_folder):
        print(f"错误: 文件夹 '{root_folder}' 不存在。请检查路径是否正确。")
        return

    # 遍历根文件夹下的所有项目（文件和子文件夹）
    for subdir_name in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir_name)

        # 只处理子文件夹
        if os.path.isdir(subdir_path):
            print(f"--- 正在进入子文件夹: {subdir_path} ---")
            files_renamed_count = 0
            
            # 遍历子文件夹中的所有文件
            for filename in os.listdir(subdir_path):
                # 使用正则表达式匹配 'image_数字.png' 格式的文件名
                # \d+ 匹配一个或多个数字
                match = re.match(r'image_(\d+)\.png$', filename)

                if match:
                    # 提取括号中匹配到的数字部分
                    image_idx_str = match.group(1)
                    image_idx_int = int(image_idx_str)

                    # 格式化为4位整数，不足的前面补零 (e.g., 5 -> '0005')
                    new_idx_str = f"{image_idx_int:04d}"
                    
                    # 构建新的文件名
                    new_filename = f"image_{new_idx_str}.png"

                    # 如果新旧文件名不同，则执行重命名
                    if new_filename != filename:
                        old_filepath = os.path.join(subdir_path, filename)
                        new_filepath = os.path.join(subdir_path, new_filename)
                        
                        try:
                            os.rename(old_filepath, new_filepath)
                            print(f"  ✅ 已重命名: {filename} -> {new_filename}")
                            files_renamed_count += 1
                        except OSError as e:
                            print(f"  ❌ 重命名失败: {filename}。错误: {e}")
                    else:
                        # 如果文件名已经符合格式，则跳过
                        print(f"  👌 已跳过 (格式正确): {filename}")

            if files_renamed_count == 0:
                print("  该文件夹中没有需要重命名的文件。")
            print("-" * (len(subdir_path) + 18))
            print("\n")

    print("🎉 所有操作完成！")


if __name__ == '__main__':
    # --- 请在这里设置你的目标文件夹路径 ---
    target_directory = 'rollouter/results/test_1115'
    
    # 运行主函数
    rename_images_in_subfolders(target_directory)

