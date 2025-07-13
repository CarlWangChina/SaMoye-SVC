""" Change the note in the ds file according to the midi_data """

import logging
from logging.handlers import RotatingFileHandler

from .utils import midi_lyric_align_helper_func as helper
import librosa


# 创建一个logger
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("my_app.log", maxBytes=1024 * 1024, backupCount=5),
        logging.StreamHandler(),
    ],
)


def mq_change_note(song_id: str, origin_ds_data, midi_data):
    """
    Change the note in the ds file according to the midi_data

    Args:
        song_id (str): The song id
        ds (dict): The ds file
        midi_data (pretty_midi.PrettyMIDI): The midi data

    Returns:
        dict: The new ds file
    """
    ds = origin_ds_data
    for sentence in ds:
        note_seq = sentence["note_seq"].split()
        offset = float(sentence["offset"])
        note_dur = helper.word_dur_to_seconds(sentence["note_dur"])
        note_start = offset
        # 遍历note_seq,找对应note
        for i, (note, note_dur) in enumerate(zip(note_seq, note_dur)):
            note_end = note_start + note_dur
            # 找到对应note
            if note != "rest":
                note_avail = []
                for instrument in midi_data.instruments:
                    if instrument.is_drum or instrument.program != 0:
                        continue
                    for note in instrument.notes:
                        if (
                            (note.start >= note_start and note.start <= note_end)
                            or (note.end >= note_start and note.end <= note_end)
                            or (note.start <= note_start and note.end >= note_end)
                            or note.end == instrument.notes[-1].end
                        ):
                            note_avail.append(note)
                        if note.start > note_end:
                            note_avail.append(note)
                            break

                note_sorted = sorted(
                    note_avail,
                    key=lambda x, nt=note_start, ne=note_end: helper.get_note_word_ratio(
                        x.start, x.end, nt, ne
                    ),
                    reverse=True,
                )
                note_closest = note_sorted[0]
                note_seq[i] = librosa.midi_to_note(note_closest.pitch)
                note_start = note_end
        sentence["note_seq"] = " ".join(note_seq)

    return ds


def mq_change_note_v2(song_id: str, origin_ds_data, midi_data):
    """
    Change the note in the ds file according to the midi_data

    Args:
        song_id (str): The song id
        ds (dict): The ds file
        midi_data (pretty_midi.PrettyMIDI): The midi data

    Returns:
        dict: The new ds file
    """
    ds = origin_ds_data
    for sentence in ds:
        note_seq = sentence["note_seq"].split()
        offset = float(sentence["offset"])
        note_dur = helper.word_dur_to_seconds(sentence["note_dur"])
        note_start = offset
        # 遍历note_seq,找对应note
        for i, (note, note_dur) in enumerate(zip(note_seq, note_dur)):
            note_end = note_start + note_dur
            # 找到对应note
            if note != "rest":
                note_avail = []
                for instrument in midi_data.instruments:
                    if instrument.is_drum or instrument.program != 0:
                        continue
                    for note in instrument.notes:
                        if (
                            (note.start >= note_start and note.start <= note_end)
                            or (note.end >= note_start and note.end <= note_end)
                            or (note.start <= note_start and note.end >= note_end)
                            or note.end == instrument.notes[-1].end
                        ):
                            note_avail.append(note)
                        if note.start > note_end:
                            note_avail.append(note)
                            break

                note_sorted = sorted(
                    note_avail,
                    key=lambda x, nt=note_start, ne=note_end: helper.get_note_word_ratio(
                        x.start, x.end, nt, ne
                    ),
                    reverse=True,
                )
                note_closest = note_sorted[0]
                note_seq[i] = librosa.midi_to_note(note_closest.pitch)
                note_start = note_end
        sentence["note_seq"] = " ".join(note_seq)

    return ds
