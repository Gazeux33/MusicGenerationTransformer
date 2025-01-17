import os

import music21

from src.utils import parse_midi_files
from config import *
from src.vectorization import VectorizeLayer


def preprocess():
    create_dirs(DATA_PATH, PARSED_SEQUENCES_DATA_PATH, VOCAB_PATH,PARSED_DATA_PATH)
    music_parser = music21.converter

    sequence_notes, sequence_durations = parse_midi_files(
        get_midi_files(), music_parser, SEQ_LEN + 1, PARSED_SEQUENCES_DATA_PATH
    )
    create_vectorizer(save=True,notes=sequence_notes,durations=sequence_durations)

def create_vectorizer(save=True,**data):
    out = []
    for k,v in data.items():
        vectorizer = VectorizeLayer(special_tokens=["[UNK]", ""])
        vectorizer.adapt(v)
        if save:
            vectorizer.save_vocab(os.path.join(VOCAB_PATH, f"{k}_vocab.json"))
        out.append(vectorizer)
    return out

def get_midi_files():
    return [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".mid")]

def create_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)