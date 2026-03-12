import numpy as np
import sys
import os
import ast
import re
import copy_path

mod = sys.argv[1]


def replace_last_match(input_string, a, b):
    # 从右到左查找第一个匹配
    pos = input_string.rfind(a)
    
    # 如果找到了匹配
    if pos != -1:
        # 用b替换匹配的部分
        input_string = input_string[:pos] + b + input_string[pos + len(a):]
    
    return input_string

def process_line(line,text,combine):
    # 检查行是否包含 'python' 且不被 '#' 注释
    if 'python' in line and not line.strip().startswith('#'):
        # 使用正则表达式提取单引号内的内容
        match = re.search(r"'([^']*)'", line)
        if match:
            original_content = match.group(1)
            print("OriginalContent:",original_content)
            # 删除 '2D', 'V30deg', 'H60deg' 等字符
            #new_content = re.sub(r'(-2D|-V30deg|-H60deg|-H30deg|-H10deg|-V10deg)', '', original_content)
            new_content = re.sub(combine, '', original_content)
            # 在末尾添加 text
            if len(text)>1:
                new_content += '-' + text
            
            # 用修改后的内容替换原内容
            #line = line.replace(original_content, new_content)
            line = replace_last_match(line, original_content, new_content)
    return line


def change_batchrun(input_file, output_file, text, combine):
    combine = '|'.join(combine)
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            processed_line = process_line(line,text,combine)
            outfile.write(processed_line)
    infile.close()
    outfile.close()
    os.system("cp " + str(output_file) + " " + str(input_file))
    os.system("rm " + str(output_file))

def process_text(text):
    # 找到 "deg" 的位置
    deg_index = text.find('deg')
    
    # 如果找到了 "deg"，开始操作
    if deg_index != -1:
        # 截取 'deg' 之前的部分
        before_deg = text[:deg_index]
        
        # 逐个检查字符，删除数字、"+"、"-"
        i = len(before_deg) - 1
        while i >= 0:
            if before_deg[i].isdigit() or before_deg[i] in ['+', '-']:
                before_deg = before_deg[:i] + before_deg[i+1:]  # 删除字符
            else:
                break  # 一旦遇到字母，停止删除
            i -= 1

        # 删除 "deg"
        text = before_deg + text[deg_index+3:]

    return text

def process_line2(line):
    # 检查行是否包含 'python' 且不被 '#' 注释
    if 'python' in line and not line.strip().startswith('#'):
        # 使用正则表达式提取单引号内的内容
        match = re.search(r"'([^']*)'", line)
        if match:
            original_content = match.group(1)

            # 删除 'deg' 以前的数字 + 和 - 号
            #new_content = re.sub(r'(-2D|-V30deg|-H60deg|-H30deg|-H10deg|-V10deg)', '', original_content)
            new_content = process_text(original_content) #re.sub(combine, '', original_content)
            # 在末尾添加 text
            #if len(text)>1:
            #    new_content += '-' + text

            # 用修改后的内容替换原内容
            line = line.replace(original_content, new_content)

    return line

def get_instrument_file_name(input_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for line in lines:
            if 'python' in line and not line.strip().startswith('#'):
                out_file = line.split()[2]
    return out_file


def change_batchrun2(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            processed_line = process_line2(line)
            outfile.write(processed_line)
    infile.close()
    outfile.close()
    os.system("cp " + str(output_file) + " " + str(input_file))
    os.system("rm " + str(output_file))


def get_data_dict(DataFile):
    with open(DataFile, 'r') as f:
        line = []
        for sline in f.readlines():
            line.append(sline.split())
    output = {}
    for item in line:
        try:
            if item[1] == '=':
                output[item[0]] = item[2]
        except IndexError:
            continue
    return output



def replace_class_params(file_path, param_values):
    """
    替换指定Python文件中的类参数值

    :param file_path: 要修改的文件路径
    :param param_values: 包含要修改的参数及其新值的字典，如：
                         {'QMax2D': 0.15, 'DataReduce2D': False, 'MaskSwitch': False}
    """
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 遍历所有要修改的参数和值
    for param, new_value in param_values.items():
        # 构造正则表达式模式，匹配 self.param = 值（考虑空格和注释）
        pattern = (
            r"(^(\s*)self\."                    # 缩进和self.参数部分
            + re.escape(param) 
            + r"(\s*=\s*))"                     # 等号及周围空格
            r"[^#\n]*"                          # 原值部分（不含注释）
        )

        # 使用repr确保值的正确Python字面量表示
        replacement = r"\g<1>" + repr(new_value)
        
        # 使用多行模式匹配每行内容
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.write(content)

def comment_cd_lines(file_path):
    # 打开文件进行读取
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 打开文件进行写入
    with open(file_path, 'w') as file:
        for line in lines:
            # 检查该行是否包含 "cd .."
            if "cd .." in line:
                # 如果包含，就在该行前添加注释符号 "#"
                file.write("#" + line)
            else:
                # 否则直接写入原内容
                file.write(line)

def uncomment_cd_lines(file_path):
    # 打开文件进行读取
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 打开文件进行写入
    with open(file_path, 'w') as file:
        for line in lines:
            # 如果该行以 "#" 开头并且包含 "cd .."
            if line.strip().startswith("#") and "cd .." in line:
                # 去掉行首的 "#" 和空格
                file.write(line.lstrip("#").lstrip())  # 去掉注释符号并保留其后的内容
            else:
                # 否则直接写入原内容
                file.write(line)

def mask_calc(user_input):
    # 提取方向字符和角度
    direction = user_input[0]  # 第一个字符代表方向
    angle = int(user_input[1:-3])  # 提取角度部分并转换为整数（忽略最后的 "deg"）

    # 判断方向并计算PhiMin和PhiMax
    if direction == "V":
        # 计算 PhiMin 和 PhiMax 对应的值
        phi_min = [0, 90 + angle / 2, 270 + angle / 2]
        phi_max = [90 - angle / 2, 270 - angle / 2, 360]
    elif direction == "H":
        # 计算 PhiMin 和 PhiMax 对应的值
        phi_min = [angle / 2, 180 + angle / 2]
        phi_max = [180 - angle / 2, 360 - angle / 2]
    else:
        print("Invalid direction! Please enter 'V' or 'H'.")
        return
    return phi_min,phi_max

def comment_phi_lines(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 创建一个新的列表存储处理后的行
    updated_lines = []

    # 遍历所有行，检查是否以 "PhiMin" 或 "PhiMax" 开头
    for line in lines:
        if line.strip().startswith("PhiMin") or line.strip().startswith("PhiMax"):
            # 如果是，则在行首添加注释符号 "#"
            updated_lines.append("# " + line)
        else:
            # 否则直接添加原始行
            updated_lines.append(line)

    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)
    file.close()


def insert_phi_info(file_path, angle, phi_min, phi_max):
    # 解析angle字符串
    direction = angle[0]  # 'H' 或 'V'
    degree = int(angle[1:-3])  # 提取角度部分
    
    # 根据方向生成 angle_text
    if direction == 'H':
        angle_text = "# horizontal {} degrees".format(degree)
    elif direction == 'V':
        angle_text = "# vertical {} degrees".format(degree)
    else:
        print("Invalid direction in angle.")
        return

    # 生成 PhiMin 和 PhiMax 的文本
    phi_min_text = "PhiMin = {}".format(str(phi_min))
    phi_max_text = "PhiMax = {}".format(str(phi_max))
    
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 查找包含"Fan mask"的行，并插入新的三行文本
    for i, line in enumerate(lines):
        if "Fan mask" in line:
            # 在"Fan mask"这一行后插入新的三行文本
            lines.insert(i + 1, '\n')
            lines.insert(i + 2, angle_text + '\n')
            lines.insert(i + 3, phi_min_text + '\n')
            lines.insert(i + 4, phi_max_text + '\n')
            break

    # 将更新后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

def insert_phi_info2(file_path, angle, phi_min, phi_max):
    # 生成 angle_text
    angle_text ="# user defined angle {} degrees".format(angle) 
    # 生成 PhiMin 和 PhiMax 的文本
    phi_min_text = "PhiMin = {}".format(str(phi_min))
    phi_max_text = "PhiMax = {}".format(str(phi_max))

    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 查找包含"Fan mask"的行，并插入新的三行文本
    for i, line in enumerate(lines):
        if "Fan mask" in line:
            # 在"Fan mask"这一行后插入新的三行文本
            lines.insert(i + 1, '\n')
            lines.insert(i + 2, angle_text + '\n')
            lines.insert(i + 3, phi_min_text + '\n')
            lines.insert(i + 4, phi_max_text + '\n')
            break

    # 将更新后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)


def extract_phi_values(input_list):
    # 初始化两个空列表，分别用于存储PhiMin和PhiMax
    PhiMin = []
    PhiMax = []

    # 遍历输入列表，提取偶数索引的值作为PhiMin，奇数索引的值作为PhiMax
    for i in range(0, len(input_list), 2):
        PhiMin.append(input_list[i])  # 偶数索引为start值
        if i + 1 < len(input_list):
            PhiMax.append(input_list[i + 1])  # 奇数索引为stop值

    return PhiMin, PhiMax

def string_to_list(input_str):
    # 使用 ast.literal_eval() 将字符串转换为列表
    try:
        result = ast.literal_eval(input_str)
        if isinstance(result, list):
            return result
        else:
            raise ValueError("输入字符串不是有效的列表格式")
    except (ValueError, SyntaxError) as e:
        return "错误: 无法转换为列表 - {}".foramt(str(e))

def format_phi_values(input_list):
    input_list = string_to_list(input_list)
    result = "0"  # 初始化结果字符串
    
    # 遍历输入列表中的start和stop值
    for i in range(0, len(input_list), 2):
        # 按照要求的格式拼接start和stop值
        result += "-{}".format(input_list[i])  # 添加start值
        result += "+{}".format(input_list[i+1])  # 添加stop值
    
    # 添加“-360deg”结尾
    result += "-360deg"
    
    return result



# 示例用法
file_path = './modules/instrument_reader.py'
# 输入和输出文件名
input_file = 'batchrun.sh'  # 请替换为你的输入文件路径
output_file = 'batchrun2.sh'  # 请替换为你的输出文件路径

#replace_class_params(file_path, param_values)


# 示例用法
instrument_file_name = get_instrument_file_name(input_file)
DataDict = get_data_dict(instrument_file_name) #get_data_dict('instrument_info/instrument_info4.5-13.5A_2.49m_8mm.txt')
#print(instrument_file_name,DataDict)
QMax = round(4*np.pi*np.sin(2.5*np.pi/180/2)/float(DataDict["WaveMin"]),2)
file_path = './modules/instrument_reader.py'
param_values_2D = {
    'QMax2D': QMax,
    'DataReduce2D': True,
    'MaskSwitch': False
}
param_values_origin = {
    'QMax2D': 0.15,
    'DataReduce2D': False,
    'MaskSwitch': False
}

HAngles = ['-H' + str(i) + 'deg' for i in range(5,175,5)]
VAngles = ['-V' + str(i) + 'deg' for i in range(5,175,5)]
HAngles.append('-2D')
VAngles.append('-2D')
combine = np.hstack((HAngles,VAngles))
mask_file = './masks/mask.py'
if mod == '2D':
    change_batchrun(input_file, output_file, mod, combine)
#    modify_class_parameters(file_path,QMax2D=0.15,DataReduce2D=False,MaskSwitch=False)  
    replace_class_params(file_path, param_values_2D)

elif mod[0] == 'H':
    mod = mod + 'deg'
    change_batchrun(input_file, output_file, mod, combine) 
    #change_batchrun2(input_file, output_file)
    replace_class_params(file_path, param_values_origin)
    comment_cd_lines('./masks/run.sh')
    PhiMin,PhiMax = mask_calc(mod)
    comment_phi_lines(mask_file)
    insert_phi_info(mask_file,mod,PhiMin,PhiMax)
    os.system("sh " + 'masks/run.sh')
    uncomment_cd_lines('./masks/run.sh')

elif mod[0] == 'V':
    mod = mod + 'deg'
    change_batchrun(input_file, output_file, mod, combine)
    #change_batchrun2(input_file, output_file)
    replace_class_params(file_path, param_values_origin)
    copy_path.main()
    comment_cd_lines('./masks/run.sh')
    PhiMin,PhiMax = mask_calc(mod)
    comment_phi_lines(mask_file)
    insert_phi_info(mask_file,mod,PhiMin,PhiMax)
    os.system("sh " + 'masks/run.sh')
    uncomment_cd_lines('./masks/run.sh')

elif mod[0] == '[':
    mod_txt = format_phi_values(mod)
    change_batchrun(input_file, output_file, mod_txt, combine)
    change_batchrun2(input_file, output_file)
    replace_class_params(file_path, param_values_origin)
    copy_path.main()
    comment_cd_lines('./masks/run.sh')
    mod = string_to_list(mod)
    PhiMin,PhiMax = extract_phi_values(mod)
    comment_phi_lines(mask_file)
    insert_phi_info2(mask_file, mod_txt, PhiMin, PhiMax)
    os.system("sh " + 'masks/run.sh')
    uncomment_cd_lines('./masks/run.sh')

elif mod == 'origin':
    change_batchrun(input_file, output_file, '', combine)
    change_batchrun2(input_file, output_file)
    replace_class_params(file_path, param_values_origin)
else:
    print(mod)
    print("please input the correct values")
    pass
    
print("文件处理完成！")

