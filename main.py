from pyarinst import ArinstDevice
import numpy as np
import cv2, argparse
from time import sleep

def get_image_from_ampl(amplitudes, gradient_max, gradient_min):
    image = []
    for amplitude_index in range(len(amplitudes)):
        image_row = []
        for time_index in range(len(amplitudes[0])):
            amplitude = amplitudes[amplitude_index][time_index]
            amplitude = max(min(amplitude, gradient_max), gradient_min)
            component = round(255 * (amplitude - gradient_min) / (gradient_max - gradient_min))
            pixel = (component, component, component)
            image_row.append(pixel)
        image.append(image_row)
    img = np.asarray(image, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mhz2hz(mhz):
    return int(mhz * 1e6)

def khz2hz(mhz):
    return int(mhz * 1e3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="device", type=str, default="COM17")
    parser.add_argument("--baudrate", help="baudrate", type=int, default=115200)
    parser.add_argument("--start", help="start MHz", type=float, default=925)
    parser.add_argument("--stop", help="stop MHz", type=float, default=955)
    parser.add_argument("--step", help="step kHz", type=float, default=200)
    parser.add_argument("--length", help="length", type=int, default=200)
    parser.add_argument("--gradient-max", help="max gradient", type=int, default=-60)
    parser.add_argument("--gradient-min", help="min gradient", type=int, default=-110)
    args = parser.parse_args()

    start = mhz2hz(args.start)
    stop = mhz2hz(args.stop)
    step = khz2hz(args.step)

    device = ArinstDevice(args.device, args.baudrate)
    data = []
    max_values = 0
    for _ in range(args.length):
        result = device.get_scan_range(start, stop, step)
        max_values = max(len(result), max_values)
        data.append(result)
        sleep(0.001)

    amplitude_data = [[None for _ in range(len(data))] for _ in range(max_values)]
    for time_index in range(len(amplitude_data[0])):
        for amplitude_index in range(len(amplitude_data)):
            amplitude_data[amplitude_index][time_index] = data[time_index][amplitude_index] if amplitude_index < len(data[time_index]) else -200

    while True:
        data = device.get_scan_range(start, stop, step)
        for index in range(0, len(amplitude_data)):
            amplitude_data[index].pop(0)
            amplitude_data[index].append(data[index] if index < len(data) else -200)

        cv2.imshow('SpectrumGrapher', get_image_from_ampl(amplitude_data, args.gradient_max, args.gradient_min))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
