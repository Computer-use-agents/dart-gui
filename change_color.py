def replace_colors_in_vue(input_file, output_file):
    """
    将Vue文件中的蓝色系配色替换为紫色系
    """
    
    # 定义所有需要替换的颜色映射
    color_replacements = [
        # 主要颜色替换
        ('#1a237e', '#4a148c'),  # 深紫色，用于标题
        
        # 链接和强调色
        ('#1976d2', '#7b1fa2'),  # 主紫色
        ('#1565c0', '#6a1b9a'),  # 深一点的紫色，hover状态
        ('#0d47a1', '#4a148c'),  # 更深的紫色，hover状态
        
        # 浅色背景和边框
        ('#90caf9', '#ce93d8'),  # 浅紫色，用于边框
        ('#64b5f6', '#ba68c8'),  # 渐变色的浅端
        ('#42a5f5', '#ab47bc'),  # hover渐变的浅端
        
        # 背景色调整
        ('#e3f0fc', '#f3e5f5'),  # 很浅的紫色背景
        ('#e3f2fd', '#f3e5f5'),  # 图标颜色
        ('#e3eaf2', '#e1bee7'),  # 边框色
        
        # 阴影颜色调整
        ('rgba(25, 118, 210, 0.04)', 'rgba(123, 31, 162, 0.04)'),
        ('rgba(25, 118, 210, 0.06)', 'rgba(123, 31, 162, 0.06)'),
        ('rgba(25, 118, 210, 0.08)', 'rgba(123, 31, 162, 0.08)'),
        ('rgba(25, 118, 210, 0.13)', 'rgba(123, 31, 162, 0.13)'),
        ('rgba(25, 118, 210, 0.18)', 'rgba(123, 31, 162, 0.18)'),
        
        # 渐变调整
        ('linear-gradient(90deg, #1976d2 0%, #64b5f6 100%)', 
         'linear-gradient(90deg, #7b1fa2 0%, #ba68c8 100%)'),
        ('linear-gradient(90deg, #1565c0 0%, #42a5f5 100%)', 
         'linear-gradient(90deg, #6a1b9a 0%, #ab47bc 100%)'),
    ]
    
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return
    
    # 执行所有替换
    replaced_count = 0
    for old_color, new_color in color_replacements:
        count = content.count(old_color)
        if count > 0:
            content = content.replace(old_color, new_color)
            replaced_count += count
            print(f"替换 {old_color} -> {new_color} ({count} 处)")
    
    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ 成功！总共替换了 {replaced_count} 处颜色")
        print(f"新文件已保存到：{output_file}")
    except Exception as e:
        print(f"写入文件时出错：{e}")
        return

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_vue_file = "D:\科研\ICLR2026\dart\src\components\Main.vue"  # 替换为你的原始Vue文件路径
    output_vue_file = "D:\科研\ICLR2026\dart\src\components\Main_alt.vue"  # 输出的紫色主题Vue文件
    
    # 执行颜色替换
    replace_colors_in_vue(input_vue_file, output_vue_file)
    
    print("\n提示：请将你的Vue代码保存为 'original.vue' 文件，")
    print("然后运行这个脚本，会生成 'purple_theme.vue' 文件")