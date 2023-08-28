import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.01)
x2 = np.arange(0, 20, 0.02)
sin1 = np.sin(x)
sin2 = np.sin(x2)
x2 /= 2
sin3 = sin1 + sin2
plt.figure(1)
plt.plot(x, sin1, color="blue")
plt.plot(x, sin2, color="red")
plt.plot(x, sin3, color="purple")
plt.draw()


def decompose_fft(data: list, threshold: float = 0.0):
    plt.figure(2)
    print(len(data))
    fft3 = np.fft.fft(data)
    print(len(fft3))
    x = np.arange(0, 10, 10 / len(data))
    freqs = np.fft.fftfreq(len(x), 0.01)
    recomb = np.zeros((len(x),))
    for i in range(len(fft3)):
        if abs(fft3[i]) / len(x) > threshold:
            sinewave = (
                1
                / len(x)
                * (
                    fft3[i].real * np.cos(freqs[i] * 2 * np.pi * x)
                    - fft3[i].imag * np.sin(freqs[i] * 2 * np.pi * x)
                )
            )
            recomb += sinewave
            plt.plot(x, sinewave)
    plt.draw()
    plt.figure(3)
    plt.plot(x, recomb, x, data)
    plt.show()


# decompose_fft(sin3, threshold=0)


plt.figure(2)
import cmath
import math

fft3 = np.fft.fft(sin3)
print(type(sin3))

""" 
for l in range(5):
    print(fft3[l])
    #print(cmath.phase(fft3[l]))
    print(np.angle(fft3[l]))
    print()
 """
freqs = np.fft.fftfreq(len(x), 0.01)
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
        #print(sinusoid)
        recomb += sinusoid
        plt.plot(x, sinusoid)

plt.draw()
plt.figure(3)

plt.plot(x, recomb, x, sin3)
plt.show()
