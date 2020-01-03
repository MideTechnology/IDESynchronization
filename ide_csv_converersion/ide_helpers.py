channels_by_name = {"Main Accel": 8, "Temp": 36, "DC Accel": 32, "IMU": 70, "Gyro": 70, "TPH" : 59}

def channel_desc_to_id(channels):
    channel_ids = []
    for channel in channels:
        numeric_channel = channels_by_name.get(channel, channel)
        channel_ids.append(numeric_channel)
    return channel_ids
