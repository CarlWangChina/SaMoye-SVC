import requests
import json

url = "http://127.0.1.1:8000/new_midi_to_vocal/"
data = {
    "new_midi_file_path": "/home/john/pipeline_for_6w/MuerSinger2/data/midi_new/4-23-new-midi/_BALMY3_129191_revision_1.mid",
    "vocal_file_path": "/home/john/CaichongSinger/data/vocal/fast_api_test/_BALMY3_129191_revision_1.mp3",
}

response = requests.post(url, data=json.dumps(data))

print(response.text)
