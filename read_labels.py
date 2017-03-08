from __future__ import division

import datetime

import pandas as pd


def _time_to_total_seconds_converter(val):
    t = datetime.datetime.strptime(val, '%H.%M.%S.%f').time()
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 10**6
    return total_seconds


def read_labels(path):
    sheets = pd.read_excel(path, sheetname=None, converters={'Time Start': _time_to_total_seconds_converter,
                                                             'Time End': _time_to_total_seconds_converter})
    return sheets