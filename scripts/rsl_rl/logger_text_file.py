import numpy as np
import os

# use example:
# from humanoid.envs import *
# from humanoid.utils import TxtFileWriter
# from humanoid import LEGGED_GYM_ROOT_DIR
# import numpy as np

# mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}'
# tmpTest=TxtFileWriter(file_dir=mujoco_model_path)
# tmpTest.add_item_info(name='step',length=1)
# tmpTest.add_item_info(name='phase',length=1)
# tmpTest.add_item_info(name='yZ',length=2)
# tmpTest.finish_item_adding()
# a=np.array([1,2,3,4])
# b=np.array([1.2,2.2,3.3,4.4])
# c=np.array([1,2,3,4,1,2,3,4])
# tmpTest.rec_item_data('step',a[0])
# tmpTest.rec_item_data('phase',b[0])
# tmpTest.rec_item_data('yZ',c[:2]) 
# tmpTest.finish_line()
# tmpTest.rec_item_data('step',a[1])
# tmpTest.rec_item_data('phase',b[1])
# tmpTest.rec_item_data('yZ',c[2:4])
# tmpTest.finish_line()


class TxtFileWriter:
    def __init__(self, file_dir, delimiter=",", fmt="%.6e"):
        """
        初始化写入类

        Args:
            file_path (str): 保存数据的文件路径。
            delimiter (str): 数据分隔符，默认为空格。
            fmt (str): 写入数据的格式，默认为 '%.6e'
        """
        self.file_dir = file_dir
        self.file_path_with_fileName = os.path.join(file_dir, "dataRec.txt")
        self.delimiter = delimiter
        self.fmt = fmt
        self.data = []  # 存储多个时间步的张量组
        with open(self.file_path_with_fileName, "w") as file:
            pass  # 清空文件内容
        # 以追加模式打开文件
        self.file = open(self.file_path_with_fileName, "a")

                # 数据记录项相关
        self.rec_item_name = []       # 名称列表
        self.rec_item_len = []        # 每个记录项的长度
        self.rec_item_start_col = []  # 每个记录项的起始列
        self.rec_item_end_col = []    # 每个记录项的结束列
        self.col_count = 0            # 当前列计数
        self.is_item_data_in = []
    
    def __del__(self):
        if self.file:
            self.file.close()
            self.file = None  # 确保不能重复关闭
        print(f"文件 {self.file_path_with_fileName} 已关闭。")

    def add_item_info(self, name, length):
        """
        添加一个数据记录项

        Args:
            name (str): 记录项的名称。
            length (int): 记录项的长度。
        """
        if name in self.rec_item_name:
            print(f"{name} has already been used!!!!!")
            raise RuntimeError("Failed to add rec item.")
        
        self.rec_item_name.append(name)
        self.rec_item_len.append(length)
        self.rec_item_start_col.append(self.col_count)
        self.rec_item_end_col.append(self.col_count + length - 1)
        self.col_count += length

    def finish_item_adding(self):
        """
        完成记录项添加，并生成 MATLAB 脚本
        """
        # MATLAB 脚本文件路径
        matlab_script_path = os.path.join(self.file_dir, "matlabReadDataScript.txt")

        try:
            with open(matlab_script_path, "w") as out_file:
                # 写入 MATLAB 脚本内容
                out_file.write("clear variables; close all;\n")
                out_file.write(f"dataRec = load('{self.file_path_with_fileName}');\n")
                for i in range(len(self.rec_item_name)):
                    start_col = self.rec_item_start_col[i] + 1  # MATLAB 列索引从 1 开始
                    end_col = self.rec_item_end_col[i] + 1
                    out_file.write(f"{self.rec_item_name[i]} = dataRec(:, {start_col}:{end_col});\n")

            print(f"MATLAB 脚本已生成: {matlab_script_path}")
        except IOError:
            print("无法创建 MATLAB 脚本文件！")
        
        self.data = np.zeros(self.col_count)
        self.is_item_data_in = np.zeros(len(self.rec_item_name),dtype=bool)

    def rec_item_data(self, name, data_in):
        """
        记录一个条目的数据

        Args:
            name (str): 条目名称，必须是之前通过 add_item 添加的名称。
            data_in (list or numpy.ndarray): 要记录的数据，长度需匹配 add_item 定义的条目长度。
        """
        if name not in self.rec_item_name:
            raise ValueError(f"{name} has not been added!!!!!")
        
        # 获取当前条目的索引
        cur_idx = self.rec_item_name.index(name)

        # 如果 data_in 是标量，则将其转换为列表
        if np.ndim(data_in) == 0:
            data_in = [data_in]

        # 检查数据长度是否匹配
        if len(data_in) != self.rec_item_len[cur_idx]:
            raise ValueError(
                f"Data length mismatch for item '{name}': expected {self.rec_item_len[cur_idx]}, got {len(data_in)}."
            )

        # 填充到相应的 rec_value 范围
        start_idx = self.rec_item_start_col[cur_idx]
        end_idx = self.rec_item_end_col[cur_idx] + 1
        self.data[start_idx:end_idx] = data_in

        # 标记该条目已记录
        self.is_item_data_in[cur_idx] = True

    def finish_line(self):
        try:
            indices = np.where(self.is_item_data_in == False)[0]
            if indices.size > 0:
                missing_idx = indices[0]
                missing_item = self.rec_item_name[missing_idx]
                raise ValueError(f"{missing_item} has not been recorded values!!!!!")
        except ValueError:
            pass  # 如果没有 False，index 会抛出异常，这表示所有条目都已记录

        np.savetxt(self.file, self.data.reshape(1, -1), fmt=self.fmt, delimiter=self.delimiter)
        # 重置记录状态
        self.rec_value = [0.0] * self.col_count
        self.is_item_data_in = [False] * len(self.rec_item_name)

    def close(self):
        """
        关闭文件句柄
        """
        if self.file:
            self.file.close()
            self.file = None  # 确保不能重复关闭
        print(f"文件 {self.file_path_with_fileName} 已关闭。")
