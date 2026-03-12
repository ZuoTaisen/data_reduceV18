#!/usr/bin/env python3
import os
import sys
import subprocess

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_dir = os.path.dirname(current_dir)

# instrument_reader.py文件路径
instrument_file = os.path.join(project_dir, 'modules', 'instrument_reader.py')

# 定义替换模式
patterns = [
    ('self.MaskSwitch = True', 'self.MaskSwitch = False'),
    ('self.MaskSwitch =True', 'self.MaskSwitch =False'),
    ('self.MaskSwitch=True', 'self.MaskSwitch=False')
]

# 读取instrument_reader.py文件内容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 写入instrument_reader.py文件内容
def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# 替换文件内容
def replace_content(content, patterns, reverse=False):
    new_content = content
    for from_pattern, to_pattern in patterns:
        if reverse:
            from_pattern, to_pattern = to_pattern, from_pattern
        new_content = new_content.replace(from_pattern, to_pattern)
    return new_content

# 执行命令
def run_command(cmd, cwd=None):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

# 清理mask.py中的旧PhiMin和PhiMax参数
import re
def clean_old_phi_params():
    mask_file = os.path.join(project_dir, 'masks', 'mask.py')
    print(f"Cleaning old PhiMin/PhiMax parameters in {mask_file}")
    
    with open(mask_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_fan_mask_section = False
    found_new_phi_params = False
    
    for i, line in enumerate(lines):
        # 如果找到# Fan mask，开始检查后面的行
        if "Fan mask" in line:
            in_fan_mask_section = True
            new_lines.append(line)
            continue
            
        # 如果在# Fan mask部分
        if in_fan_mask_section:
            # 如果是非注释的PhiMin或PhiMax行，保留并标记已找到新参数
            if not line.strip().startswith('#') and line.strip().startswith(('PhiMin', 'PhiMax')):
                found_new_phi_params = True
                new_lines.append(line)
                continue
            # 如果已经找到新参数，跳过所有注释行和空行
            elif found_new_phi_params and (line.strip().startswith('#') or line.strip() == ''):
                continue
            # 如果已经找到新参数，遇到非注释行，结束清理
            elif found_new_phi_params and not line.strip().startswith('#'):
                new_lines.append(line)
                in_fan_mask_section = False
                continue
            # 如果还没有找到新参数，跳过所有注释行和空行
            elif not found_new_phi_params and (line.strip().startswith('#') or line.strip() == ''):
                continue
            # 如果还没有找到新参数，遇到非注释行，保留
            elif not found_new_phi_params and not line.strip().startswith('#'):
                new_lines.append(line)
                continue
        
        # 保留其他所有行
        new_lines.append(line)
    
    # 写回文件
    with open(mask_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✓ Old PhiMin/PhiMax parameters cleaned successfully")

def main():
    # 0. 清理mask.py中的旧PhiMin和PhiMax参数
    print("Step 0: Cleaning old PhiMin/PhiMax parameters in mask.py")
    clean_old_phi_params()
    
    # 1. 修改instrument_reader.py中的self.MaskSwitch值为False
    print("Step 1: Modifying instrument_reader.py - setting self.MaskSwitch to False")
    content = read_file(instrument_file)
    modified_content = replace_content(content, patterns)
    write_file(instrument_file, modified_content)
    print("✓ instrument_reader.py modified successfully")
    
    # 2. 运行mask.py
    print("\nStep 2: Running mask.py")
    mask_cmd = f"python {os.path.join('masks', 'mask.py')} ./instrument_info/instrument_info2.2-6.7A_12.75m_8mm.txt ./sample_info/sample_info2.2-6.7A_12.75m_8mm_StandardSample12.75.txt 0 2"
    return_code = run_command(mask_cmd, cwd=project_dir)
    if return_code != 0:
        print(f"✗ mask.py failed with return code {return_code}")
        # 恢复instrument_reader.py
        write_file(instrument_file, content)
        return return_code
    print("✓ mask.py completed successfully")
    
    # 3. 将self.MaskSwitch值恢复为True
    print("\nStep 3: Restoring instrument_reader.py - setting self.MaskSwitch to True")
    content = read_file(instrument_file)
    restored_content = replace_content(content, patterns, reverse=True)
    write_file(instrument_file, restored_content)
    print("✓ instrument_reader.py restored successfully")
    
    # 4. 运行plotD3Mask.py
    print("\nStep 4: Running plotD3Mask.py")
    plot_cmd = "python plotD3Mask.py"
    return_code = run_command(plot_cmd, cwd=current_dir)
    if return_code != 0:
        print(f"✗ plotD3Mask.py failed with return code {return_code}")
        return return_code
    print("✓ plotD3Mask.py completed successfully")
    
    print("\nAll steps completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
