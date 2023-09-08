import numpy as np
import matplotlib.pyplot as plt


prox = [5,9,8,13,2]
z=4
x = np.arange(0,z,1)
y = np.array(prox[0:z])

print(x, y)
curve_fit = np.polynomial.polynomial.Polynomial.fit(x=x,y=y, deg=2)

x_poly, y_poly = curve_fit.linspace()
print(curve_fit, '\n')

#what = np.polynomial.polynomial.polyval(c=curve_fit.coef, x=0, tensor=False)

for r in range(z+1):
    print(curve_fit(r))
    plt.plot(curve_fit(r), linewidth=5)

#print(what)
#plt.plot(what, linewidth=5)

plt.plot(y, color='blue')
plt.plot(x_poly, y_poly, color='orange')
plt.show()