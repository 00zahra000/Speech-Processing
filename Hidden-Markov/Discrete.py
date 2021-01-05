import os
import scipy.io.wavfile as wav
import python_speech_features
from os import listdir, getcwd
from os.path import isfile, join
import pickle
import numpy as np
from VQ import Model as VQModel, M
from hmmlearn import hmm

def build_dataset(sound_path='digits-raw/'):
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = dict()
    i = 0

    for f in files:
        feature = delta_extractor(sound_path=sound_path + f)
        if i % 10 == 0:
            x_test.append(feature)
            y_test.append(int(f[0]))
        else:
            x_train.append(feature)
            y_train.append(f[0])
        i += 1

    for i in range(0, len(x_train), len(x_train) // 10):
        data[y_train[i]] = x_train[i:i + len(x_train) // 10]
    return x_train, y_train, x_test, y_test, data


def delta_extractor(sound_path):
    (sampling_freq, audio) = wav.read(sound_path)
    mfcc_features = python_speech_features.mfcc(audio, sampling_freq, numcep=16, appendEnergy=True)
    delta_features = python_speech_features.delta(mfcc_features, 2)
    delta_features = np.array(delta_features)
    # all_features = [mfcc_features, delta_features]
    # all_features = np.array(all_features)

    return delta_features

# get the Model directory
model_path = join(getcwd(), "Models")


def quantize(codebook, data):
    # this method vector quantizes the data according to the codebook provided.
    quantized_data = []
    for index, datum in enumerate(data):
        distance_vector = (codebook - datum) ** 2
        distance = np.array(map(lambda x: sum(x), distance_vector))
        distance_plus_index = zip(distance, range(len(distance)))
        index_of_min_codeword = min(distance_plus_index)[1]
        quantized_data.append(index_of_min_codeword)
    return list(quantized_data) + range(M)


class Model:
    """
    This class provides the base for tasks like training and testing.
    """
    def __init__(self, id):
        self.id = id
        self.filename = join(model_path, self.id + '.VQHMM')

    def absolute_train(self, data):
        # save the data so that we may use it some other time (e.g. visualization)
        self.data = data
        self.codebook = VQModel.get_codebook(data)
        # quantize the data according to the codebook
        quantized_value = quantize(self.codebook, data)

        self.hmm = hmm.MultinomialHMM(n_components=5)
        self.hmm.fit([quantized_value])
        return

    def train(self, data):
        allfiles = [join(model_path, obj) for obj in listdir(model_path) if isfile(obj)]
        if self.id + '.VQHMM' in allfiles:
            # the model already exists, we update the model instead
            pass
        else:
            self.absolute_train(data)
        # pickle it to save it
        self.save()

    def save(self):
        pickle.dump(self, open(join(model_path, self.id + '.VQHMM'), "w"))
        return

    def calculate_score(self, data):
        quantized_data = quantize(self.codebook, data)
        return -self.hmm.score(quantized_data)

def load_model(filename):
    return pickle.load(open(join(model_path, filename), "r"))


def test(data, probabilistic=False):

    allfiles = [join(model_path, obj) for obj in listdir(model_path) if
                isfile(join(model_path, obj)) and obj[-5:] == "VQHMM"]
    scores = {}
    for files in allfiles:
        model = load_model(files)
        id = files[:-6]
        scores[id] = model.calculate_score(data)

    model_score = zip(scores.keys(), scores.values())
    model_score.sort(key=lambda x: x[1])

    if not probabilistic:
        # just return the highest match
        return model_score[0][0]
    else:
        total_score = sum(i[1] for i in model_score)
        return [(i[0], float(i[1]) / total_score) for i in model_score]
    pass

x_train, y_train, x_test, y_test, data = build_dataset()

learned_hmm = load_model(data.values)
with open("learned.pkl", "wb") as file:
    pickle.dump(learned_hmm, file)

with open("learned.pkl", "rb") as file:
    learned_hmm = pickle.load(file)

y_pred = test(x_test, learned_hmm)


# y_pred = prediction(x_test, learned_hmm)
# report(y_test, y_pred, show_cm=True)


q_data= quantize(20, learned_hmm)
train= Model.absolute_train(q_data)
saving = Model.save(train)
testing = test(x_test)