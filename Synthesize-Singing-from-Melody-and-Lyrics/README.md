# 输出的文件格式
- Vocal
    - SongID
        - songid_origin.mid # 原midi
        - songid_origin_no_vocals.wav # 原伴奏
        - songid_origin.json # 原时间戳
        - songid.ds # 由midi_lyric_align生成
        - songid_fix.mid # 由ds_2_midi生成
        - new_song_id
            - songid_newid.mid # 由书宇的旋律生成模型生成
            - songid_newid_companyid.wav # 新伴奏
            - songid_newid_companyid_mix.mp3/wav #混音后新歌曲

            - songid_newid.ds # 由change_ds_midi 生成
            - songid_newid_var.ds # 由variance model 生成
            - songid_newid_vocal.wav # 由acoustic model 生成
# pipeline

1. 检查数据是否均存在，不存在需从s3下载songID对应的 歌词时间戳、原曲
2. 由midi_lyric_align生成songid.ds
3. 由ds_2_midi生成 songid_fix.mid
4. 输入songid_fix.mid给书宇生成songid_newid.mid
5. 由change_ds_midi 生成songid_newid.ds
6. 由variance model 生成songid_newid_var.ds
7. 由acoustic model 生成songid_newid_vocal.wav
8. 由mix 生成songid_newid_mix.mp3/wav

