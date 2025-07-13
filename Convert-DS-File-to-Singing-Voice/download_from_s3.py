import mysql.connector  
import json  
import os  
import subprocess  # 用于执行外部命令
import argparse
from src.params import song_ids

# Database configuration  
config = {  
    'user': 'music_data',  
    'password': 'JR4#BjH675dS21vcOkju',  
    'host': '124.17.0.226',  
    'database': 'mysong',  
    'port': 3301,  
} 

# Function to save lyrics to a JSON file  
def download_from_index_music(test_name,rewrite=False): 
    SongIds = song_ids[test_name]
    # SongIds = [523572, 903269, 1169589, 1686672, 153343, 973273, 847575, 1303121, 888509, 880101, 122844, 1287594]
    bpm_midi_path = f'data/bpm_midi/{test_name}'
    midi_path = f'data/midi/{test_name}'
    json_path = f'data/json/{test_name}'
    origin_mp3_path = f'data/origin_mp3/{test_name}'
    midi_src = f'/export/data/home/john/ds_generator/data/midi_5b'
    
    os.makedirs(bpm_midi_path, exist_ok=True)
    os.makedirs(midi_path, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)
    os.makedirs(origin_mp3_path, exist_ok=True)

    # Establish database connection  
    cnx = mysql.connector.connect(**config)  
    cursor = cnx.cursor(dictionary=True)  
    
    # Prepare SQL query  
    query = "SELECT songId, mfa_lyric, s3_bucket, original_s3_path FROM indexed_music WHERE songId IN (%s)"  
    placeholders = ', '.join(['%s'] * len(SongIds))  
    query = query % placeholders  
    
    try:  
        # Execute the query  
        cursor.execute(query, tuple(SongIds))  
        
        # Fetch all rows and parse lyrics into a list
        for row in cursor.fetchall():
            songId = row['songId']
            s3_bucket = row['s3_bucket']

            mfa_lyric_jsonl_str = row['mfa_lyric']
            original_s3_path = row['original_s3_path']
            
            bpm_midi_file_path = os.path.join(bpm_midi_path, f"{songId}_src.mp3_5b.mid") 
            if rewrite or not os.path.exists(bpm_midi_file_path):
                mv_midi(songId, midi_src, bpm_midi_file_path)
            
            midi_file_path = os.path.join(midi_path, f"{songId}_new.mid") 
            if rewrite or not os.path.exists(midi_file_path):
                mv_midi(songId, midi_src, midi_file_path)

            json_file_path = os.path.join(json_path, f"{songId}.json")
            if rewrite or not os.path.exists(json_file_path):
                save_lyrics_to_json(mfa_lyric_jsonl_str, songId, json_file_path) 

            origin_mp3_file_path = os.path.join(origin_mp3_path, f"{songId}.mp3")
            if rewrite or not os.path.exists(origin_mp3_file_path):
                save_origin_mp3(s3_bucket, songId, original_s3_path, origin_mp3_file_path)
    
    except mysql.connector.Error as err:  
        print(f"Error: {err}")  
    finally:  
        # Close the cursor and database connection  
        cursor.close()  
        cnx.close()


# Function to save lyrics to a JSON file  
def save_lyrics_to_json(mfa_lyric_jsonl_str, songId, file_path):   
    # If mfa_lyric_jsonl_str is not empty, parse and append to list  
    if mfa_lyric_jsonl_str!=None and mfa_lyric_jsonl_str!='':
        lyrics_list = []  
        # Split the JSONL string into separate JSON object strings  
        json_obj_strs = mfa_lyric_jsonl_str.strip().split('\n')  
        
        # Parse each JSON object string and append to lyrics_list  
        for json_obj_str in json_obj_strs:  
            if json_obj_str:  # Skip empty lines  
                lyrics_dict = json.loads(json_obj_str)  
                lyrics_list.append(lyrics_dict)  

        # Save the lyrics_list to a JSON file  
        with open(file_path, 'w', encoding='utf-8') as json_file:  
            json.dump(lyrics_list, json_file, ensure_ascii=False, indent=4)  

        print(f"Lyrics saved to {file_path}")
    else:
        print(f"No lyrics found for song {songId}, skipping")

# Function to save lyrics to a JSON file  
def save_origin_mp3(s3_bucket, songId, original_s3_path, destination_path): 
    # If original_s3_path is not empty, copy file from S3 to local directory
    if original_s3_path:
        # Use awscli to copy from S3 to local directory
        command = f"aws s3 cp s3://{s3_bucket}/{original_s3_path} {destination_path}"
        subprocess.run(command, shell=True)
        print(f"File copied from S3 to {destination_path}")
    else:
        print(f"No original S3 path found for song {songId}, skipping")

def mv_midi(songId,src,dst):
    if os.path.exists(f"{src}/{songId}_src.mp3_5b.mid"):
        os.system(f"cp {src}/{songId}_src.mp3_5b.mid {dst}")
        print(f"File copied from {src}/{songId}_src.mp3_5b.mid to {dst}")
    else:
        print(f"No midi file found for song {songId}, skipping")

if __name__ == '__main__':  
    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--test_name', type=str, default='test1', help='测试名称')
    # 解析命令行参数
    args = parser.parse_args()
    download_from_index_music(args.test_name,rewrite=False)