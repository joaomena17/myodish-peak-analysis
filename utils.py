import pyabf
import pandas as pd


def extract_channels(abf):
    """
    Function to extract the channels from the abf file into a list of pandas DataFrames
    :param abf: pyabf.ABF object
    :return: list of pandas.DataFrame
    """
    channels = []
    for sweep in range(9):
        abf.setSweep(0, channel=sweep)

        df = pd.DataFrame({
            'time_index': abf.sweepX,
            'signal_value': abf.sweepY
        })

        channels.append(df)

    return channels