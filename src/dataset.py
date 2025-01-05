from torch.utils.data import Dataset



class MusicDataset(Dataset):
    def __init__(self, notes, durations, seq_length=20):
        self.notes = notes
        self.durations = durations
        self.seq_length = seq_length
        if self.notes.shape != self.durations.shape:
            raise ValueError("notes and durations must have the same size")

    def __len__(self):
        return len(self.notes) - self.seq_length

    def __getitem__(self, idx):
        # x_note,x_duration,y_note,y_duration
        return self.notes[idx, :-1], self.durations[idx, :-1] ,self.notes[idx, 1:], self.durations[idx, 1:]