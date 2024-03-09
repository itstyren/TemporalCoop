import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def cal_parameter(rho_c):
    q_cc = np.around(1 / (k - 1) + rho_c * (k - 2) / (k - 1), decimals=4)
    q_dc = np.around((1 - rho_c) * (k - 2) / (k - 1), decimals=4)
    q_cd = np.around(rho_c * (k - 2) / (k - 1), decimals=4)
    q_dd = np.around(1 - rho_c * (k - 2) / (k - 1), decimals=4)
    return q_cc, q_dc, q_cd, q_dd    # tuple


def payoff_difference(rho_c, p, r):
    parameter = cal_parameter(rho_c)
    q_dc = parameter[1]
    q_dd = parameter[3]

    f1 = 1 / (k - 1) + r * (k - p - 2 * p * k) / (p * (k + 1) * (k - 1))
    pai_ac_ad = ((pow((1 - p), k) * (f1 + r + sigma * k + 1)
                 + sigma * k * pow(p * q_dd + 1 - p, k) * (1 + p * q_dd))
                 + r * (1 - k / (p * (k + 1))) / (k + 1) - 1)

    # pai_ic_id is irrelevant to r
    pai_ic_id = sigma * ((p * q_dd + 1 - p) ** k - (p * q_dc + 1 - p) ** k
                         + k * p * (q_dc - q_dd) * pow(1 - p, k - 1))

    cal_fuc = (1+p) * pai_ac_ad + (1-p) * pai_ic_id
    return cal_fuc


# initial value
rho_c = 0.5
k = 4    # degree
sigma = 0.5


def equation(r, p):
    return sp.nsimplify(payoff_difference(rho_c, p, r), rational=True, tolerance=0.001)


def plot_inequality():
    # cal equation value
    p_list = np.around(np.linspace(0.01, 1, 51), decimals=2)
    r_list = np.around(np.linspace(0, 5, 51), decimals=2)
    r, p = np.meshgrid(r_list, p_list)
    Eq = np.array(equation(r, p))

    Eq_decimal = np.vectorize(lambda x: round(float(x), 4))(Eq)

    # Linear normalization to [-1,1]
    min_value = np.min(Eq_decimal)
    max_value = np.max(Eq_decimal)
    Eq_normalized = np.zeros_like(Eq_decimal)
    for i in range(Eq_decimal.shape[0]):
        for j in range(Eq_decimal.shape[1]):
            if Eq_decimal[i, j] > 0:
                Eq_normalized[i, j] = Eq_decimal[i,j] / max_value
            elif Eq_decimal[i, j] < 0:
                Eq_normalized[i, j] = -Eq_decimal[i,j] / min_value

    plt.imshow(Eq_normalized.astype(float), cmap='seismic_r', origin='lower')
    plt.colorbar()

    plt.xlabel('r')
    plt.ylabel('p')
    plt.title('Inequality Plot')

    plt.show()


if __name__ == "__main__":
    plot_inequality()