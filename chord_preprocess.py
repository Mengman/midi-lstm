from music21 import converter, note

class NoteInfo:

    def __init__(self, note):
        self.__name = note.name
        self.__octave = note.octave
        self.__offset = note.offset
    
    @property
    def name(self):
        return self.__name
    
    @property
    def octave(self):
        return self.__octave
    
    @property
    def offset(self):
        return self.__offset
    
    def note(self):
        n = note.Note(self.__name + str(self.__octave))
        n.offset = self.__offset
        return n

    def __eq__(self, other):
        return self.__key() == other.__key() 

    def __key(self):
        return (self.__name, self.__octave, self.__offset)
    
    def __hash__(self):
        return hash(self.__key())


def parse_midi(data_fn):
    midi_data = converter.parse(data_fn)

    chord_stream = midi_data[0]
    chord_stream = chord_stream[2:]

    corpus_notes = []
    for chord in chord_stream:
        for n in chord:
            corpus_notes.append(NoteInfo(n))
        
        # use C1 as a pause
        corpus_notes.append(NoteInfo(note.Note('C1')))
    
    return corpus_notes
        
    

    
