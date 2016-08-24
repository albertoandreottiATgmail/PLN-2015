import pickle
import matplotlib.pyplot as plt
"""
a = pickle.load(open('losses1_0.pkl', 'rb'))
b = pickle.load(open('losses1_1.pkl', 'rb'))
c = pickle.load(open('losses2_2.pkl', 'rb'))
d = pickle.load(open('losses3_3.pkl', 'rb'))
e = pickle.load(open('losses4_4.pkl', 'rb'))
z = range(len(a))
"""
a = pickle.load(open('losses_240.pkl', 'rb'))
b = pickle.load(open('losses_480.pkl', 'rb'))
c = pickle.load(open('losses_960.pkl', 'rb'))

plt.plot(a, '-or', label='240')
plt.plot(b, '-ob', label='480')
plt.plot(c, '-oy', label='960')
#plt.plot(d, '-og', label='3:3')
#plt.plot(e, '-oc', label='4:4')
plt.legend()
plt.grid()
plt.show()
