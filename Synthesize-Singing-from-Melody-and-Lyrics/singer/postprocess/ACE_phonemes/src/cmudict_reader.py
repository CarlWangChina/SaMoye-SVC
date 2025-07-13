import pickle
import os

CMUDICT_PATH = '../resources/acecmudict.rep'

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, CMUDICT_PATH)
CACHE_PATH = os.path.join(current_file_path, '../resources/dict_cache.pickle')


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:          
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split('  ')
                word = word_split[0]

                syllable_split = word_split[1].split(' - ')
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(' ')
                    g2p_dict[word].append(phone_split)

            line_index = line_index +1
            line = f.readline()
        
    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):    
        with open(CACHE_PATH, 'rb') as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict



if __name__ == "__main__":
    g2p_dict = get_dict()
    print(g2p_dict)