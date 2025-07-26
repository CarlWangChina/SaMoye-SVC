"""
    需要将文件夹中的文件按照16个文件夹一起进行zip打包
    
    1. 读取文件夹 /app/data/4-23
    文件夹结构为：
    4-23
    ├── 0.1-0.1-0.1-0.1-1e-10-1000.0-8
        ├── xxx.mp3
        ├── xxx.mp3
    ├── 0.1-0.1-0.001-0.1-1e-10-1000.0-8
    ...
    2. 每个文件夹生成一个xlsx文件，文件内容为：
    | filename | score |
    | xxx.mp3  |       |
    文件名为：mp3_score.xlsx
    3. 将文件夹中的文件按照16个文件夹一起进行zip打包,不要改变子文件夹结构
    压缩包文件夹结构为:
    4-23
    ├── batch_1.zip
        ├── 1:0.1-0.1-0.1-0.1-1e-10-1000.0-8
            ├── xxx.mp3
            ├── xxx.mp3
            ...
            ├── mp3_score.xlsx
        ├── 2
            ...
    ├── batch_2.zip
"""

import os  
import zipfile  
from pathlib import Path  
import pandas as pd  
import logging  
  
# 设置日志  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
  
# 文件夹路径  
base_folder = Path('/app/data/4-23')  
  
# 定义一个函数来生成xlsx文件，此处的score为占位数据，需要根据实际情况填写音频处理逻辑来生成真实的分数  
def generate_xlsx(folder_path):  
    files = list(folder_path.glob('*.mp3'))  
    data = [{'filename': file.name, 'score': ''} for file in files]  # 假设score为空，需要音频处理逻辑来填充  
    df = pd.DataFrame(data)  
    xlsx_path = folder_path / 'mp3_score.xlsx'  
    df.to_excel(xlsx_path, index=False)  
    return xlsx_path  
  
# 定义一个函数来打包文件夹  
def zip_folders(folders, batch_number, output_dir):  
    zip_filename = output_dir / f'batch_{batch_number}.zip'  
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:  
        for folder in folders:  
            xlsx_path = generate_xlsx(folder)  # 生成xlsx文件并获取路径  
            for file_path in folder.rglob('*'):  
                if file_path.is_file():  
                    arcname = str(file_path.relative_to(base_folder))  
                    zipf.write(file_path, arcname=arcname)  
                    logging.info(f'Added {arcname} to {zip_filename.name}')  
    return zip_filename  
  
# 获取所有子文件夹，并排序  
subfolders = sorted(list(base_folder.glob('*')))  
subfolders = [f for f in subfolders if f.is_dir()]  
  
# 设置输出目录  
output_dir = base_folder / 'output'  
output_dir.mkdir(exist_ok=True)  
  
# 每16个文件夹打包一次  
batch_size = 16  
batches = [subfolders[i:i+batch_size] for i in range(0, len(subfolders), batch_size)]  
  
for batch_number, batch in enumerate(batches, start=1):  
    zip_file = zip_folders(batch, batch_number, output_dir)  
    logging.info(f'Created zip file: {zip_file}')
