########################################################################
# Mantz Wyrick
# Machine Learning Semester Project
#
# Generate classic rock music from midi files using RNN and GAN.
#
########################################################################

# included libraries
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from numpy import expand_dims, zeros, ones, vstack
from numpy.random import randn, randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import LSTM, Bidirectional, Activation, BatchNormalization
from keras.utils import np_utils
from matplotlib import pyplot
from music21 import converter, instrument, note, chord, stream
from google.colab import drive

drive.mount("/content/gdrive")


def get_midi():
    """Parse the rock and roll midi dataset into a list of chords and notes"""
    notes = []
    for file in glob.glob(
        "/content/gdrive/My Drive/My Colab Notebooks/Classic_Rock_Generator/midi_rock/*.mid"
    ):
        # for file in glob.glob("/content/gdrive/My Drive/WSU/CPTS/CPTS 437/Term Project/midi_rock/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


def prep_notes(notes, n_vocab, sequence_length):
    """Format notes to work with GAN and normalize data"""

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    X = []
    y = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        X.append([note_to_int[char] for char in sequence_in])
        y.append(note_to_int[sequence_out])

    n_patterns = len(X)

    # Reshape the input into a format compatible with LSTM layers
    X = np.reshape(X, (n_patterns, sequence_length, 1))

    # Normalize input between -1 and 1
    X = (X - float(n_vocab) / 2) / (float(n_vocab) / 2)
    y = np_utils.to_categorical(y)

    return (X, y)


def init_discriminator(seq_shape):
    """Create the discriminator model with Keras"""
    model = Sequential()
    model.add(LSTM(512, input_shape=seq_shape, return_sequences=True))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def init_generator(latent_dim, seq_shape):
    """Create the generator model with Keras"""
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(seq_shape), activation="tanh"))
    model.add(Reshape(seq_shape))
    return model


def define_gan(g_model, d_model):
    """Create the GAN from the discriminator and generator"""
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


def train(
    d_model,
    g_model,
    gan_model,
    latent_dim,
    disc_loss,
    gen_loss,
    epochs,
    sequence_length=100,
    batch_size=128,
    sample_interval=1,
):  # 50): # keep sample_interval=1 to get as smooth of a graph as possisble
    """Train the GAN with inputed midi data"""
    # Load and convert the data
    notes = get_midi()
    n_vocab = len(set(notes))
    X_train, y_train = prep_notes(notes, n_vocab, sequence_length)

    # Adversarial ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Train the model
    for epoch in range(epochs):
        # Select a random batch of note sequences
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_seqs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_seqs = g_model.predict(noise)

        # Train the discriminator network
        d_loss_real = d_model.train_on_batch(real_seqs, real)
        d_loss_fake = d_model.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the genrator network
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan_model.train_on_batch(noise, real)

        if epoch % sample_interval == 0:
            if epoch % 50 == 0:
                print("Epoch:%d\tD loss: %f\tG loss: %f" % (epoch, d_loss[0], g_loss))
            disc_loss.append(d_loss[0])
            gen_loss.append(g_loss)

    generate(g_model, notes, n_vocab, latent_dim)
    plot_loss(disc_loss, gen_loss)


def create_midi(prediction_output, filename):
    """Translate the generated sequence of notes into readable data for music21 and then create a midi file"""
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for item in prediction_output:
        pattern = item[0]
        # pattern is a chord
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                # new_note.storedInstrument = instrument.Violin() # piano will be the only insturment in the midi file
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(instrument.SteelDrum())
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            # new_note.storedInstrument = instrument.Violin() # tried to change the instrument but no matter what I changed it to it was always a piano playing in the generated midi file
            output_notes.append(instrument.SteelDrum())
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="{}.mid".format(filename))


def generate(g_model, input_notes, n_vocab, latent_dim):
    """generate a sequence of notes using the trained generator"""
    # Get pitch names and store in a dictionary
    notes = input_notes
    pitchnames = sorted(set(item for item in notes))
    int_to_note = dict(
        (number, note) for number, note in enumerate(pitchnames)
    )  # range of numbers from 0 to 201

    # Use random noise to generate sequences
    noise = np.random.normal(0, 1, (1, latent_dim))
    predictions = g_model.predict(noise)
    temp_Pred = []
    pred_notes = []
    for x in predictions[0]:
        if x > 0:
            pred_notes.append(x * 201)
        else:
            pred_notes.append(x * 201 + 201)
    pred_notes = [int_to_note[int(x[0])] for x in pred_notes]

    create_midi(pred_notes, "final")


def plot_loss(disc_loss, gen_loss):
    """Plot a graph of the Loss over time in the GAN"""
    plt.plot(disc_loss, c="red")
    plt.plot(gen_loss, c="blue")
    plt.title("GAN Loss VS Epoch")
    plt.legend(["Discriminator", "Generator"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    """main function running the GAN"""
    seq_length = 200
    seq_shape = (seq_length, 1)
    latent_dim = 1000
    d_loss = []
    g_loss = []

    d_model = init_discriminator(seq_shape)
    g_model = init_generator(latent_dim, seq_shape)
    gan_model = define_gan(g_model, d_model)

    train(
        d_model,
        g_model,
        gan_model,
        latent_dim,
        d_loss,
        g_loss,
        1000,
        sequence_length=seq_length,
    )
