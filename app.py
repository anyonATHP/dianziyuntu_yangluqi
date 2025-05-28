from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, assoc_laguerre
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    n = float(request.form['n'])
    l = float(request.form['l'])
    m = float(request.form['m'])

    x = np.linspace(-n**2*4, n**2*4, 500)
    y = 0 
    z = np.linspace(-n**2*4, n**2*4, 500)
    X, Z = np.meshgrid(x, z)
    rho = np.linalg.norm((X, y, Z), axis=0) / n
    Lag = assoc_laguerre(2 * rho, int(n - l - 1), int(2 * l + 1))
    Ylm  = sph_harm(m, l, np.arctan2(y, X), np.arctan2(np.linalg.norm((X, y), axis=0), Z))
    Psi = np.exp(-rho) * np.power((2 * rho), l) * Lag * Ylm
    density = np.conjugate(Psi) * Psi
    density = density.real

    plt.figure(figsize=(6,6))
    plt.imshow(density, extent=[-density.max()*0.1, density.max()*0.1,  
                                -density.max()*0.1, density.max()*0.1])
    plt.axis('off')
    plt.savefig('static/electron_density.png', bbox_inches='tight')
    plt.close()
    return render_template('result.html', n=n, l=l, m=m)
