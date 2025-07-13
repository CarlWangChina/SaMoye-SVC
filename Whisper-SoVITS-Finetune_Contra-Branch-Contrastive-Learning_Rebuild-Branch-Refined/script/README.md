# 干声、开源歌声语音数据集快速处理脚本（使用前先检查路径）
1. 使用./script/data_process/mv_data.ipynb将数据搬到/home/john/svc/muer-svc-whisper-sovits-svc/data （我将干声和开源歌声语音分开了文件夹，不想分开的话可以合并到一起进行处理；opencpop数据集需要密码手动解压，密码：Mmwjxhn2017）
2. 使用./script/data_process/ln_s.ipynb将data目录内容链接到相应目录下
3. 使用./script/data_process/preprocess_data.ipynb进行切片，然后提取特征
4. 去除音频里某些数据集的内容：./script/data_process/filelist_subset.ipynb

# train_test 训练脚本集合
1. train.ipynb 训练没有修改过的whisper-sovits-svc
2. train_spk.ipynb 训练加入了speaker encoder的模型