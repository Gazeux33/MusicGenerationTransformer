import torch

DATA_PATH = "data/raw/"
PARSED_DATA_PATH = "data/parsed"
PARSED_SEQUENCES_DATA_PATH = "data/parsed_sequences"
VOCAB_PATH = "data/vocab"
PARSE_MIDI_FILES = False
SEQ_LEN = 50
BATCH_SIZE = 256
EMBEDDING_DIM = 128
KEY_DIM = 256
N_HEADS = 4
DROPOUT_RATE = 0.3
FEED_FORWARD_DIM = 256
EPOCHS = 1
LEARNING_RATE=1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOTES_TOKENIZER_FILE = "notes_vocab.json"
DURATIONS_TOKENIZER_FILE = "durations_vocab.json"
CHECKPOINT_DIR = "checkpoints"
TRAINING_SAVING_FREQ = 1
EPOCHS = 5000