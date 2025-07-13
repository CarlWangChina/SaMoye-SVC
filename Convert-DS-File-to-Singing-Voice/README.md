# 环境
pip install -r MuerSinger2/DiffSinger/requriement


# 数据存放

- data

    -midi:新生成的midi，其文件夹名将作为后续所有的中间文件和输出的文件夹名

    -bpm_midi:原midi，用于提取bpm信息

    -json :原时间戳歌词的json

    -ds：经过对齐后的ds文件，提供给唱法模型进行输入

    -var_ds：经过唱法模型生成F0和ph_dur的ds文件，提供给Acoustic模型作为输入

    -origin_mp3:原MP3

    -vocal：最终生成的歌曲歌声

    -song：简单混合后的歌曲

## ds 说明：获取路径/home/john/MuerSinger2/data/ds/test1
[    
    {
        "offset": 16.455000000000002,  # 每一句开始的点

        "word_seq": "AP 我 相 信 有 一 双 手", 
        "word_dur": "0.2 0.441 0.441 0.441 0.441 0.221 0.662 1.323",
        "ph_seq": "AP w o x iang x in y ou y i sh uang sh ou ",

        "note_seq": "rest F♯3 C♯4 B3 C♯4 D♯4 C♯4 F♯3", # 每个note

        "note_dur": "0.2 0.441 0.441 0.441 0.441 0.221 0.662 1.323", # 每个note对应的时长

        要获取note onset 就是 offset + 其前面的音符的时间和，offset ： offset + 其前面的音符的时间和 + 其dur

        "note_slur": "0 0 0 0 0 0 0 0",
        "ph_num": "2 2 2 2 2 2 2 1"
    }