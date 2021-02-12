from data import *
from pitch import *
from instrument import *


if __name__ == '__main__':

    features_df = load_data()
    print(features_df.head())

    print("===================================================================")
    print("\n[INSTRUMENT TRAINING]")
    instrument_training(features_df)

    print("===================================================================")
    print("\n[PITCH TRAINING]")
    pitch_training(features_df)
