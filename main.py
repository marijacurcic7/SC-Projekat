from data import *
from pitch import *
from instrument import *


if __name__ == '__main__':

    features_gray = load_data()
    features_df = pd.DataFrame(features_gray, columns=["file", "feature", "instrument", "pitch"])
    print(features_df.head())

    print("===================================================================")
    print("\n[INSTRUMENT TRAINING]")
    #instrument_training(features_df, features_gray)

    print("===================================================================")
    print("\n[PITCH TRAINING]")
    pitch_training(features_df, features_gray)
