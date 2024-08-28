import numpy as np
from scipy.ndimage import uniform_filter1d

class Channel:
    name: str = ""
    short_name: str = ""
    unit: str = ""
    time: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    freq: int = 0

def copy_chan(chan: Channel, new_data: np.ndarray = None) -> Channel:
    new_chan = Channel()
    new_chan.name = chan.name
    new_chan.short_name = chan.short_name
    new_chan.unit = chan.unit
    new_chan.time = chan.time
    if new_data is None:
        new_chan.data = chan.data.copy()
    else:
        new_chan.data = new_data
    new_chan.freq = chan.freq
    return new_chan

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
    if chan.unit in ["deg", "degrees", "deg/s", "degrees/s", "deg/s^2", "degrees/s^2", "°/s", "°/s^2", "°"]:
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

def pitch_roll_yaw_transform(x_c: Channel, y_c: Channel, z_c: Channel, roll: float, pitch: float, yaw: float) -> tuple[Channel, Channel, Channel]:
    """
    Perform roll, pitch, and yaw transformation on the given x, y, and z axis arrays.

    Parameters:
    - x: numpy array representing the x-axis
    - y: numpy array representing the y-axis
    - z: numpy array representing the z-axis
    - roll: angle in radians for roll (rotation around x-axis)
    - pitch: angle in radians for pitch (rotation around y-axis)
    - yaw: angle in radians for yaw (rotation around z-axis)

    Returns:
    - x_transformed: transformed x-axis numpy array
    - y_transformed: transformed y-axis numpy array
    - z_transformed: transformed z-axis numpy array
    """
    x = x_c.data
    y = y_c.data
    z = z_c.data
    # Create rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Stack the axes into a single matrix
    original_axes = np.stack([x, y, z])

    # Apply the rotation matrix to the axes
    transformed_axes = R @ original_axes

    # Unpack the transformed axes
    x_transformed, y_transformed, z_transformed = transformed_axes
    x_transformed_c = copy_chan(x_c, x_transformed)
    y_transformed_c = copy_chan(y_c, y_transformed)
    z_transformed_c = copy_chan(z_c, z_transformed)
    return x_transformed_c, y_transformed_c, z_transformed_c