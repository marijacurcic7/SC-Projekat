from data import *
from pitch import *
from instrument import *


if __name__ == '__main__':

    features_gray = load_data_spec()
    features_cnn = pd.DataFrame(features_gray, columns=["file", "feature", "instrument", "pitch"])
    print(features_cnn.head())

    print("===================================================================")
    print("\n[INSTRUMENT CNN]")
    instrument_training(features_cnn, features_gray)

    print("===================================================================")
    print("\n[PITCH CNN]")
    pitch_training(features_cnn, features_gray)


