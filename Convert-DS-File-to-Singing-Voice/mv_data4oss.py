import os
import shutil
from src.params import song_ids

def reorganize_folders(data_dir, new_data_dir):
    # 遍历每个文件夹
    origin_midi_dir = os.path.join(data_dir, "bpm_midi", "singer600_2")
    origin_json_dir = os.path.join(data_dir, "json", "singer600_2")
    # 需要对1，到100的test进行错误统计，给我100长度为0的数组，每个位置代表一个test的错误数量
    error_num = [0] * 100

    for song_id in song_ids["singer600_2"]:
        song_id = str(song_id)
        song_id_dir = os.path.join(new_data_dir, song_id)
        if not os.path.exists(song_id_dir):
            os.makedirs(song_id_dir)
            
        origin_midi_path = os.path.join(origin_midi_dir, song_id + "_src.mp3_5b.mid")
        new_midi_path = os.path.join(song_id_dir, song_id + "_origin.mid")
        if os.path.exists(origin_midi_path):
            if not os.path.exists(new_midi_path):
                shutil.copy(origin_midi_path, new_midi_path)
        else:
            print("midi file not exists: ", origin_midi_path)
        
        origin_json_path = os.path.join(origin_json_dir, song_id + "_fixed.json")
        new_json_path = os.path.join(song_id_dir, song_id + "_fixed.json")
        if os.path.exists(origin_json_path):
            if not os.path.exists(new_json_path):
                shutil.copy(origin_json_path, new_json_path)
        else:
            print("json file not exists: ", origin_json_path)
        
        origin_json_path = os.path.join(origin_json_dir, song_id + ".json")
        new_json_path = os.path.join(song_id_dir, song_id + "_origin.json")
        if os.path.exists(origin_json_path):
            if not os.path.exists(new_json_path):
                shutil.copy(origin_json_path, new_json_path)
        else:
            print("json file not exists: ", origin_json_path)

        origin_company_path = os.path.join("/home/john/MuerSinger2/separated/mdx_extra", song_id,  "no_vocals.wav")
        new_company_path = os.path.join(song_id_dir, song_id + "_origin_no_vocals.wav")
        if os.path.exists(origin_company_path):
            if not os.path.exists(new_company_path):
                shutil.copy(origin_company_path, new_company_path)
        else:
            print("company file not exists: ", origin_company_path)

        for i in range(1, 101):
            new_midi_src_path = os.path.join(data_dir, 'midi', f"singer600_2_test{i}", song_id + "_new.mid")
            new_midi_dst_dir = os.path.join(song_id_dir, f"new_{i}")
            if os.path.exists(new_midi_src_path):
                if not os.path.exists(new_midi_dst_dir):
                    os.makedirs(new_midi_dst_dir)
                new_midi_dst_path = os.path.join(new_midi_dst_dir, song_id + "_new_1.mid")
                if not os.path.exists(new_midi_dst_path):
                    shutil.copy(new_midi_src_path, os.path.join(new_midi_dst_path))
            else:
                # print("midi file not exists: ", new_midi_src_path)
                error_num[i-1] += 1
    
    error_test = []
    for i in range(100):
        if error_num[i] > 0 :
            error_test.append(i+1)
    print(error_test)
if __name__ == "__main__":
    data_dir = "/home/john/MuerSinger2/data"  # 指定数据文件夹的路径
    new_data_dir = "/home/john/MuerSinger2/data4oss_2"  # 指定新数据文件夹的路径
    reorganize_folders(data_dir, new_data_dir)
