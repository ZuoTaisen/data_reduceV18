import numpy as np
import sys
import os
import ast
import re
import subprocess
import copy_path
import path_manager

class FileHandler:
    @staticmethod
    def read_lines(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    @staticmethod
    def write_lines(file_path, lines):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    @staticmethod
    def copy_file(src, dest):
        # 使用Python的shutil模块代替shell命令
        import shutil
        shutil.copy2(src, dest)

    @staticmethod
    def remove_file(file_path):
        # 使用Python的os模块代替shell命令
        os.remove(file_path)

def replace_last_match(input_string, a, b):
    pos = input_string.rfind(a)
    if pos != -1:
        input_string = input_string[:pos] + b + input_string[pos + len(a):]
    return input_string


class BatchProcessor:
    def __init__(self, combine_patterns):
        self.combine = re.compile('|'.join(combine_patterns))
        
    def _process_line(self, line, text):
        if 'python run.py' in line and not line.strip().startswith('#'):
            # 匹配命令行参数中的sample名称
            # 无论是在单引号中还是在f-string中
            
            # 首先尝试匹配单引号中的内容
            matches = re.findall(r"'([^']*)'", line)
            if matches:
                # 取最后一个匹配项（通常是sample名称）
                content = matches[-1]
                new_content = self.combine.sub('', content)
                if text and len(text) > 1:
                    new_content += '-{}'.format(text)
                line = replace_last_match(line, content, new_content)
                return line
            
            # 处理f-string格式的命令行
            # 使用简单直接的方法：找到所有StandardSample开头的字符串，然后替换最后一个
            sample_matches = re.findall(r"StandardSample[\w\d\-\.]+", line)
            if sample_matches:
                # 取最后一个匹配项
                content = sample_matches[-1]
                new_content = self.combine.sub('', content)
                if text and len(text) > 1:
                    new_content += '-{}'.format(text)
                # 使用replace_last_match函数确保只替换最后一个匹配项
                line = replace_last_match(line, content, new_content)
                return line
        return line

    def process_batch(self, input_file, output_file, text):
        lines = FileHandler.read_lines(input_file)
        processed = [self._process_line(line, text) for line in lines]
        FileHandler.write_lines(output_file, processed)
        FileHandler.copy_file(output_file, input_file)
        FileHandler.remove_file(output_file)

def process_text(text):
    # 找到 "deg" 的位置
    deg_index = text.find('deg')

    # 如果找到了 "deg"，只移除 "deg" 后缀，保留前面的所有内容
    if deg_index != -1:
        # 截取 'deg' 之前的部分并直接返回
        text = text[:deg_index]

    return text

class BatchProcessor2:
    def __init__(self, combine_patterns):
        self.combine = re.compile('|'.join(combine_patterns))

    def _process_line(self, line):
        if 'python' in line and not line.strip().startswith('#'):
            match = re.search(r"'([^']*)'", line)
            if match:
                content = match.group(1)
                new_content = process_text(content)
                return line.replace(content, new_content)
        return line

    def process_batch(self, input_file, output_file):
        lines = FileHandler.read_lines(input_file)
        processed = [self._process_line(line) for line in lines]
        FileHandler.write_lines(output_file, processed)
        FileHandler.copy_file(output_file, input_file)
        FileHandler.remove_file(output_file)


class ConfigModifier:
    @staticmethod
    def replace_params(file_path, params):
        content = ''.join(FileHandler.read_lines(file_path))
        for param, value in params.items():
            # 使用明确的组引用语法\g<1>
            pattern = r"(self\.{}\s*=\s*)([^\n#]+)".format(re.escape(param))
            content = re.sub(
                pattern, 
                r"\g<1>{}".format(repr(value)),  # 使用\g<1>明确引用第一个捕获组
                content
            )
        FileHandler.write_lines(file_path, [content])

    @staticmethod
    def toggle_cd_lines(file_path, comment=True):
        lines = FileHandler.read_lines(file_path)
        processed = []
        for line in lines:
            if "cd .." in line:
                line = "#{}".format(line) if comment else line.lstrip("#").lstrip()
            processed.append(line)
        FileHandler.write_lines(file_path, processed)

class MaskHandler:
    @staticmethod
    def process_angle(text):
        deg_pos = text.find('deg')
        if deg_pos == -1:
            return text
            
        before_deg = text[:deg_pos]
        for i in reversed(range(len(before_deg))):
            if before_deg[i] in '+-0123456789':
                before_deg = before_deg[:i]
            else:
                break
        return before_deg + text[deg_pos+3:]

    @staticmethod
    def calculate_phi(angle_input):
        # Handle HV case (no angle specified)
        if angle_input[:-3] == 'HV':
            return ([], [])
        direction = angle_input[0]
        degrees = int(angle_input[1:-3])
        if direction == 'V':
            return (
                [0, 90 + degrees/2, 270 + degrees/2],
                [90 - degrees/2, 270 - degrees/2, 360]
            )
        elif direction == 'H':
            return (
                [degrees/2, 180 + degrees/2],
                [180 - degrees/2, 360 - degrees/2]
            )
        raise ValueError("Invalid direction")

    @staticmethod
    def update_phi_config(file_path, angle, phi_min, phi_max):
        lines = FileHandler.read_lines(file_path)
        new_lines = []
        for line in lines:
            if line.strip().startswith(('PhiMin', 'PhiMax')):
                new_lines.append('# ' + line)
            else:
                new_lines.append(line)
        
        angle_type = 'horizontal' if angle[0] == 'H' else 'vertical'
        insert = [
            '\n',
            '# {} {} degrees\n'.format(angle_type, angle[1:-3]),
            'PhiMin = {}\n'.format(phi_min),
            'PhiMax = {}\n'.format(phi_max)
        ]
        
        for i, line in enumerate(new_lines):
            if "Fan mask" in line:
                new_lines[i+1:i+1] = insert
                break
                
        FileHandler.write_lines(file_path, new_lines)

class DataProcessor:
    @staticmethod
    def load_data(file_path):
        data = {}
        for parts in (line.split() for line in FileHandler.read_lines(file_path)):
            try:
                if len(parts) > 2 and parts[1] == '=':
                    data[parts[0]] = parts[2]
            except IndexError:
                continue
        return data

    @staticmethod
    def format_phi_string(input_str):
        try:
            values = ast.literal_eval(input_str)
            formatted = '0' + ''.join(
                '-{}+{}'.format(values[i], values[i+1]) 
                for i in range(0, len(values), 2)
            ) + '-360deg'
            return formatted
        except (ValueError, SyntaxError):
            raise ValueError("Invalid input format")

def get_instrument_file_name(input_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for line in lines:
            if 'python run.py' in line and not line.strip().startswith('#'):
                # 对于batchrun.sh，使用split()[2]
                if input_file.endswith('.sh'):
                    out_file = line.split()[2]
                # 对于batchrun.py，从f-string中提取
                elif input_file.endswith('.py'):
                    # 查找instrument_info文件路径，同时处理正斜杠和反斜杠
                    match = re.search(r"instrument_info[/\\][^\s']+", line)
                    if match:
                        out_file = match.group(0)
                    else:
                        # 如果没有找到，尝试从完整命令中提取
                        parts = line.split()
                        if len(parts) >= 3:
                            out_file = parts[2]
                        else:
                            raise ValueError("无法从batchrun.py中提取instrument_file_name")
    return out_file


def change_comment(file_path, a, b):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []

    for line in lines:
        # 检查是否包含a列表中的任意一个文本
        if any(text in line for text in a):
            if not line.strip().startswith('#'):  # 如果没有被注释掉
                line = "#        {}".format(line.strip(' '))  # 将该行注释掉

        # 检查是否包含b列表中的任意一个文本
        elif any(text in line for text in b):
            if line.strip().startswith('#'):  # 如果已经注释掉
                line = line.lstrip('#').lstrip()  # 解注释
                line = "        {}".format(line)

        modified_lines.append(line)

    # 将修改后的内容写回文件
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def replace_str(filename, old_str, new_str):
    """适用于大文件的字符串替换"""
    try:
        import tempfile
        import os
        import shutil
        
        replace_count = 0
        
        # 获取文件所在目录
        file_dir = os.path.dirname(os.path.abspath(filename))
        
        # 在相同目录下创建临时文件，确保在同一磁盘驱动器
        with tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            encoding='utf-8',
            dir=file_dir,  # 确保临时文件在同一目录
            suffix='.tmp'  # 添加.tmp后缀以明确是临时文件
        ) as temp_file:
            temp_path = temp_file.name
            
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    # 统计每行的替换次数
                    line_count = line.count(old_str)
                    if line_count > 0:
                        replace_count += line_count
                        line = line.replace(old_str, new_str)
                    temp_file.write(line)
        
        # 关闭文件后替换原文件
        try:
            # 备份原文件（可选）
            # shutil.copy2(filename, filename + '.bak')
            
            # 删除原文件
            os.remove(filename)
            # 将临时文件重命名为原文件名
            shutil.move(temp_path, filename)
            
        except Exception as e:
            # 如果出错，尝试恢复
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
        print(f"成功替换 {replace_count} 处匹配")
        return replace_count > 0
        
    except Exception as e:
        print(f"错误：{e}")
        return False

# 初始化配置
PARAMS_2D = {
    'QMax2D': 0.15,
    'DataReduce2D': True,
    'MaskSwitch': False
}

PARAMS_ORIGIN = {
    'QMax2D': 0.15,
    'DataReduce2D': False,
    'MaskSwitch': False,
    'GISANS_mode': False
}

PARAMS_GISANS = {
    'GISANS_mode': True,
}


PARAMS_LOG = {
    'QBins': 120,
}


PARAMS_LIN = {
    'QBins': 600,
}

PARAMS_LOGLIN = {
    'QBins': 600,
}
def generate_wavelength_points(LambdaSections, WaveBand):
    """
    根据LambdaSections和WaveBand生成StartPointsDiv和StopPointsDiv字符串
    
    参数:
        LambdaSections: 波长分段数
        WaveBand: 波长带宽
    
    返回:
        StartPointsDiv, StopPointsDiv: 字符串形式的波长起始点和终止点列表
    """
    step = WaveBand / LambdaSections
    start_points = []
    stop_points = []
    
    for i in range(LambdaSections):
        start_val = f'self.WaveMin+{step*i:.1f}'
        stop_val = f'self.WaveMin+{step*(i+1):.1f}'
        start_points.append(start_val)
        stop_points.append(stop_val)
    
    StartPointsDiv = '[' + ','.join(start_points) + ']'
    StopPointsDiv = '[' + ','.join(stop_points) + ']'
    
    return StartPointsDiv, StopPointsDiv

# 波长切割参数
LambdaSections = 7  # 波长分段数
WaveBand = 4.5      # 波长带宽
# 生成波长切割点
StartPointsDiv, StopPointsDiv = generate_wavelength_points(LambdaSections, WaveBand)

StartPoints = '[self.WaveMin]'
StopPoints = '[self.WaveMax]'
#StartPoints = '[self.WaveMin]'
#StartPointsDiv = '[self.WaveMin,self.WaveMin+1,self.WaveMin+2,self.WaveMin+3,self.WaveMin+4]'
#StopPoints = '[self.WaveMax]'
#StopPointsDiv = '[self.WaveMin+1,self.WaveMin+2,self.WaveMin+3,self.WaveMin+4,self.WaveMax]'


if __name__ == "__main__":
    #data_reduce_path = path_manager.resource_path('data_reduce')
    #os.chdir(data_reduce_path)
    input_file = 'batchrun.py'  # 从batchrun.sh改为batchrun.py
    mod = sys.argv[1]
    instrument_file_name = get_instrument_file_name(input_file)
    #data = get_data_dict(instrument_file_name)

    data = DataProcessor.load_data(instrument_file_name) #'instrument_info/instrument_info4.5-13.5A_2.49m_8mm.txt')
    qmax = round(4*np.pi*np.sin(2.5*np.pi/180/2)/float(data["WaveMin"]), 2)
    PARAMS_2D['QMax2D'] = qmax
    
    angles = ['-H{}deg'.format(i) for i in range(5,175,5)] + \
            ['-V{}deg'.format(i) for i in range(5,175,5)] + \
            ['-HVdeg'] + ['-2D'] + ['-lambdadiv']
    batch_processor = BatchProcessor(angles)
    batch_processor2 = BatchProcessor2(angles)

    mask_file = './masks/mask.py'
    
    # 处理输入参数，移除"deg"后缀
    processed_mod = process_text(mod)
    
    if mod == '2D':
        #batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', processed_mod)
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_2D)
        
    elif mod[0] in ('H', 'V'):
        mod_with_deg = mod + 'deg'
        processed_mod_with_deg = process_text(mod_with_deg)
       # batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', processed_mod_with_deg)
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_ORIGIN)
        copy_path.main()
       # mod = string_to_list(mod) 
        phi_min, phi_max = MaskHandler.calculate_phi(mod_with_deg)
        MaskHandler.update_phi_config(mask_file, mod_with_deg, phi_min, phi_max)

        # 从batchrun.py中提取仪器文件和样品文件路径，替换masks/run.py中的路径
        with open('batchrun.py', 'r', encoding='utf-8') as f:
            batchrun_content = f.read()

        # 使用正则表达式提取仪器文件和样品文件路径
        match = re.search(r"python run\.py (\./instrument_info/[^\s'\"]+) (\./sample_info/[^\s'\"]+)", batchrun_content)
        if match:
            instrument_path = match.group(1)
            sample_path = match.group(2)  # 直接从batchrun.py中提取样品文件路径

            # 读取masks/run.py
            with open('./masks/run.py', 'r', encoding='utf-8') as f:
                run_content = f.read()

            # 替换mask_cmd中的路径
            old_mask_cmd_pattern = r'(mask_cmd = f"python \{os\.path\.join\(\'masks\', \'mask\.py\'\)\} )(\./instrument_info/[^\s"]+) (\./sample_info/[^\s"]+) (\d \d)'
            new_mask_cmd = r'\g<1>{} {} \4'.format(instrument_path, sample_path)
            run_content = re.sub(old_mask_cmd_pattern, new_mask_cmd, run_content)

            # 写回masks/run.py
            with open('./masks/run.py', 'w', encoding='utf-8') as f:
                f.write(run_content)

        # 调用我们新创建的run.py脚本，它会处理所有mask相关的操作
        # 包括修改self.MaskSwitch、执行mask.py和plotD3Mask.py
        subprocess.run(['python', './masks/run.py'], check=True)
        
    elif mod.startswith('['):
        formatted_mod = DataProcessor.format_phi_string(mod)
        batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', formatted_mod)
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_ORIGIN)
        copy_path.main()
        
        values = ast.literal_eval(mod)
        phi_pairs = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        phi_min, phi_max = zip(*phi_pairs) if phi_pairs else ([], [])
        MaskHandler.update_phi_config(mask_file, formatted_mod, list(phi_min), list(phi_max))
        
        # 调用我们新创建的run.py脚本，它会处理所有mask相关的操作
        # 包括修改self.MaskSwitch、执行mask.py和plotD3Mask.py
        subprocess.run(['python', './masks/run.py'], check=True)
        
    elif mod == 'origin':
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', '')
        batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_ORIGIN)
        replace_str(r'./run.py','LambdaDivided = True','LambdaDivided = False')
        replace_str(r'./modules/instrument_reader.py', 'LambdaDivide = True','LambdaDivide = False')
        replace_str(r'./modules/instrument_reader.py', StartPointsDiv,StartPoints)
        replace_str(r'./modules/instrument_reader.py', StopPointsDiv,StopPoints)
  
    
    elif mod == 'loglin':
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_LOGLIN)
        change_comment('./modules/instrument_reader.py',['self.QX = np.logspace','self.QX = np.linspace'],['self.QX = self.q_generate'])
    elif mod == 'lin':
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_LIN)
        change_comment('./modules/instrument_reader.py',['self.QX = np.logspace','self.QX = self.q_generate'],['self.QX = np.linspace'])
    elif mod == 'log':
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_LOG)
        change_comment('./modules/instrument_reader.py',['self.QX = np.linspace','self.QX = self.q_generate'],['self.QX = np.logspace'])

    elif mod == 'gisans':
        #batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', processed_mod)
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_2D)
        ConfigModifier.replace_params('./modules/instrument_reader.py', PARAMS_GISANS)

    elif mod == 'lambdadiv':
        #batch_processor2.process_batch('batchrun.py', 'batchrun2.sh')
        batch_processor.process_batch('batchrun.py', 'batchrun2.sh', processed_mod)
        replace_str(r'./run.py','LambdaDivided = False','LambdaDivided = True')
        replace_str(r'./modules/instrument_reader.py', 'LambdaDivide = False','LambdaDivide = True')
        replace_str(r'./modules/instrument_reader.py', StartPoints,StartPointsDiv)
        replace_str(r'./modules/instrument_reader.py', StopPoints,StopPointsDiv) 
    else:
        print("Invalid input:", mod)
    
    print("Processing completed!")
