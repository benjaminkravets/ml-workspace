        

z=4
x = np.arange(0,z,1)
y = np.array(prox[0:z][::-1])

if(1):
    print(x, y)
    curve_fit = numpy.polynomial.polynomial.Polynomial.fit(x=x,y=y, deg=3)

    x_poly, y_poly = curve_fit.linspace()
    print(curve_fit, '\n')

    what = np.polynomial.polynomial.polyval(c=curve_fit.coef, x=0, tensor=False)

    
    
    print(what)
    plt.plot(what, linewidth=5)
    
    plt.plot(y, color='blue')
    plt.plot(x_poly, y_poly, color='orange')
    plt.show()