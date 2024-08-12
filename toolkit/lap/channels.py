import numpy as np

class Channel:
    name: str = ""
    short_name: str = ""
    unit: str = ""
    time: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    freq: int = 0

# This function loads a channel as it is stored in a mat file exported by i2Pro
def load_channel(name:str, data):
    channel = Channel()
    channel.name = name
    channel.short_name = name
    if len(data["Units"][0, 0]) > 0:
        channel.unit = data["Units"][0, 0][0]
    else:
        channel.unit = "None"
    channel.time = data["Time"][0, 0][0, :]
    channel.data = data["Value"][0, 0][0, :]
    d_t = np.diff(channel.time)
    channel.freq = int(1/np.mean(d_t))
    return channel

# This also assumes a mat file exported by i2Pro
def parse_car_data_mat(data) -> dict[str, Channel]:
    channel_dict = {}
    for channel in data.keys():
        if channel[0] != "_":
            channel_dict[channel] = load_channel(channel, data[channel])
    return channel_dict