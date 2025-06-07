from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, assoc_laguerre
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 记得把 <form> 的 method 改成 "get"

@app.route('/plot')  # 默认只接受 GET
def plot():
    # 从 URL 查询参数中获取 n, l, m
    try:
        n = float(request.args.get('n', 1.0))
        l = float(request.args.get('l', 0.0))
        m = float(request.args.get('m', 0.0))
    except (TypeError, ValueError):
        return "参数不合法，请检查 n, l, m 是否为数字。", 400

    # 生成网格
    x = np.linspace(-n**2 * 4, n**2 * 4, 500)
    y = 0
    z = np.linspace(-n**2 * 4, n**2 * 4, 500)
    X, Z = np.meshgrid(x, z)
    rho = np.linalg.norm((X, y, Z), axis=0) / n

    # 计算径向拉盖尔多项式和球谐函数
    Lag = assoc_laguerre(2 * rho, int(n - l - 1), int(2 * l + 1))
    Ylm = sph_harm(int(m), int(l),
                   np.arctan2(y, X),
                   np.arctan2(np.linalg.norm((X, y), axis=0), Z))

    # 波函数与密度
    Psi = np.exp(-rho) * (2 * rho)**l * Lag * Ylm
    density = (Psi.conj() * Psi).real

    # 绘图并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(density,
               extent=[-n**2, n**2, -n**2, n**2],
               origin='lower')
    plt.axis('off')
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/electron_density.png', bbox_inches='tight')
    plt.close()

    return render_template('result.html', n=n, l=l, m=m)
    
if __name__ == '__main__':
    app.run(port=8000, debug=True)


