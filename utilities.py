import os
import glob
import csv
import re


import pandas as pd


def read_second_row(file_name: str) -> dict:
    with open(file_name, "r") as f:
        rdr = csv.reader(f)
        next(rdr)
        row_str = " ".join(next(rdr))
    match = re.findall(r'([\w_]+):\s*(-?\d*\.?\d+)([a-zA-Z%/]*)', row_str.strip())
    data = {key: float(value) for key, value, _ in match}
    return data


def parse_file_name(file_name: str, m_type: str):
    ch_num, meas_num, sw_num, conc_num = (0, 0, 0, 0)
    start_str = os.path.split(file_name)[-1].split("_")
    if m_type == "noise":
        ch_num = int(start_str[-1].split(".")[0])
        meas_num = int(start_str[-2])
        conc_num = 0
        sw_num = int(start_str[-3])
    elif m_type == "calibration":
        ch_num = start_str[-3]
        meas_num = int(start_str[-4])
        conc_num = int("".join(start_str[6:8]))
        sw_num = int("".join(start_str[2:6]))
    elif m_type ==  "test":
        ch_num = int(start_str[-1].split(".")[0])
        meas_num = int(start_str[-2])
        conc_num = int(start_str[-3])
        sw_num = 0
    return ch_num, meas_num, sw_num, conc_num


def create_main_dataframe(files_path: str, file_ext: str = "*.csv", measurement_type:str = "noise") -> pd.DataFrame:
    file_list = glob.glob(os.path.join(files_path, file_ext))
    cols = ["Time", "Current"]
    skip = 3
    if measurement_type == "test":
        cols = ["Voltage", "Current", "DataType"]
        skip = 3
    elif measurement_type == "calibration":
        cols = ["Time", "Voltage", "Current", "DataType"]
        skip = 4

    file_frame_list = []

    for file in file_list:
        channel_num, measurement_num, sweep_num, conc_num = parse_file_name(file, m_type=measurement_type)
        info = read_second_row(file)
        file_frame = pd.read_csv(file, delimiter=";", skiprows=skip, header=None)
        file_frame.columns = cols
        file_frame["FileName"] = file
        file_frame["ChNum"] = channel_num
        file_frame["MeasureNum"] = measurement_num
        file_frame["SweepNum"] = sweep_num
        file_frame["ConcNum"] = conc_num
        for key, value in info.items():
            file_frame[key] = value
        file_frame_list.append(file_frame)
    return pd.concat(file_frame_list, axis=0)


