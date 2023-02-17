import matplotlib.pyplot as plt
import numpy as np

x = np.arange(start=0, stop=6.2, step=0.1)
y = np.sin(x)
y_cos = np.cos(x)

#figure, axis = plt.subplots(2, 1)

#axis[0].plot(x, y)
#axis[1].plot(x, y_cos)

#axis[0].set_title("Function sin(x)")
#axis[1].set_title("Function cos(x)")
#plt.plot(x, y)
#plt.ylabel("Function Y")
#plt.xlabel("Argument X")
#plt.title("Function sin(x)")
#plt.savefig('plot.tiff', dpi=200)
plt.plot(x, y, '*', label="sin", color='blue')
plt.plot(x, y_cos, "o", label="cos", color="red")
#plt.plot(x, y, x, y_cos)
#plt.colormaps("green")
plt.legend()
plt.grid()
plt.title("Sin и Cos на одном графике")
plt.show()

