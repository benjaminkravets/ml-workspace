import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
interval = 1
ranga = 100
x = np.arange(0, ranga, interval)


if(1):
    data = read_csv("datashop/spydaily.csv", header=0, usecols=[1])
    sin3 = data.values[0:ranga].flatten()

plt.figure(1)
plt.plot(x, sin3, color="purple")
plt.draw()

plt.figure(2)
import cmath

fft3 = np.fft.fft(sin3)

if(0):

    freqs = np.fft.fftfreq(len(x), interval)
    threshold = 0.0
    recomb = np.zeros((len(x),))
    middle = len(x) // 2 + 1
    for i in range(middle):
        if abs(fft3[i]) / (len(x)) > threshold:
            if i == 0:
                coeff = 2
            else:
                coeff = 1
            sinusoid = (
                1
                / (len(x) * coeff / 2)
                * (abs(fft3[i]) * np.cos(freqs[i] * 2 * np.pi * x + cmath.phase(fft3[i])))
            )
            recomb += sinusoid
            plt.plot(x, sinusoid)

if(1):
    freqs = np.fft.fftfreq(len(x), interval)
    print(x)
    threshold = 0.0
    recomb = np.zeros((len(x),))
    middle = len(x) // 2 + 1
    for i in range(middle):
        if abs(fft3[i]) / (len(x)) > threshold:
            if i == 0:
                coeff = 2
            else:
                coeff = 1
            sinusoid = (
                1
                / (len(x) * coeff / 2)
                * (abs(fft3[i]) * np.cos(freqs[i] * 2 * np.pi * x + cmath.phase(fft3[i])))
            )
            recomb += sinusoid
            plt.plot(x, sinusoid)
#print(recomb[-1])
plt.draw()

plt.figure(3)
plt.plot(x, recomb, x, sin3)
plt.show()
