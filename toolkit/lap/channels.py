import numpy as np
from scipy.ndimage import uniform_filter1d

class Channel:
    name: str = ""
    short_name: str = ""
    unit: str = ""
    time: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    freq: int = 0

def derivative_chan(chan: Channel, name: str, short_name: str = None, unit: str = "", smooth: int = None) -> Channel:
    new_chan = Channel()
    new_chan.name = name
    if short_name is None:
        new_chan.short_name = name
    else:
        new_chan.short_name = short_name
    new_chan.unit = unit
    new_chan.time = chan.time
    if smooth is None:
        new_chan.data = np.gradient(chan.data, chan.time)
    else:
        new_chan.data = np.gradient(uniform_filter1d(chan.data, smooth), chan.time)
    return new_chan

def deg2rad_chan(chan: Channel) -> Channel:
    if chan.unit in ["deg", "degrees", "°/s", "°/s^2", "°"]:
        if chan.unit in ["deg", "degrees", "°"]:
            chan.unit = "rad"
        elif chan.unit in ["°/s", "deg/s", "degrees/s"]:
            chan.unit = "rad/s"
        else:
            chan.unit = "rad/s^2"
        chan.data = np.deg2rad(chan.data)
        return chan
    else:
        print(f"Channel {chan.name} is not in degrees, but rather {chan.unit}, conversion failed")
        return chan
    
def unit_conversion(chan: Channel, unit: str, factor: float, name: str = None, short_name: str = None) -> Channel:
    if name is None:
        name = chan.name
    if short_name is None:
        short_name = chan.short_name
    chan.name = name
    chan.short_name = short_name
    chan.unit = unit
    chan.data = chan.data * factor
    return chan

# This function loads a channel as it is stored in a mat file exported by i2Pro
def load_channel(name:str, data) -> Channel:
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

def null_channel(name: str, time_offset: float = 0) -> Channel:
    channel = Channel()
    channel.name = name
    channel.short_name = name
    channel.unit = "None"
    channel.time = np.array([0.0]) + time_offset
    channel.data = np.array([0.0])
    return channel

# This also assumes a mat file exported by i2Pro
def parse_car_data_mat(data) -> dict[str, Channel]:
    channel_dict = {}
    for channel in data.keys():
        if channel[0] != "_":
            channel_dict[channel] = load_channel(channel, data[channel])
    return channel_dict