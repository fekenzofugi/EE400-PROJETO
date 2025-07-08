import numpy as np

# Constantes
v = 300  # velocidade da luz em km/ms
u = 3.986004418e5  # constante gravitacional da Terra (km^3/s^2)
alpha = 0.6  # taxa de aprendizado para o gradiente descendente

# Dados dos satélites
a = np.array([15300, 16100, 17800, 16400])        # km
e = np.array([0.41, 0.342, 0.235, 0.3725])        # excentricidade 0-1
w = np.radians([60, 10, 30, 60])                  # rad
i = np.radians([30, 30, 0, 20])                   # rad
o = np.radians([0, 40, 40, 40])                   # rad
tp = np.array([4708.5603, 5082.6453, 5908.5511, 5225.3666])  # s

TOA = np.array([60000] * 4)  
TOT = np.array([13581.1080927 , 19719.32768037, 11757.73393255, 20172.46081236])  

# Funções de rotação
def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def rot_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def kepler(E, M, e):
    return E - e * np.sin(E) - M

def d_kepler(E, e):
    return 1 - e * np.cos(E)

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    E = M  # chute inicial
    for _ in range(max_iter):
        f = kepler(E, M, e)
        df = d_kepler(E, e)
        E_new = E - f / df
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E 

def sat_pos_eci(a, e, w, i, o, tp):
    n = np.sqrt(u / a**3) 
    M = n * tp
    E = solve_kepler(M, e)

    # Coordenadas no sistema perifocal
    x_p = a * (np.cos(E) - e)
    y_p = a * np.sqrt(1 - e**2) * np.sin(E)
    z_p = 0
    r_pf = np.array([x_p, y_p, z_p])

    # Transformação para ECI
    T = rot_z(o) @ rot_x(i) @ rot_z(w)
    r_eci = T @ r_pf
    return r_eci

def calcula_gradiente(satelites_eci, ponto_estimado, tempos_medidos):
    resultado = np.zeros(3)
    for i in range(len(satelites_eci)):
        origem = satelites_eci[i]
        tempo = tempos_medidos[i]
        alcance = v * tempo

        delta = ponto_estimado - origem
        distancia2 = np.dot(delta, delta)
        if distancia2 < 1e-10:
            continue

        distancia = np.sqrt(distancia2)
        peso = 1.0 - (alcance / distancia)
        resultado += peso * delta
    return resultado

r_positions = []
for k in range(4):
    pos = sat_pos_eci(a[k], e[k], w[k], i[k], o[k], tp[k])
    r_positions.append(pos)

# TOF - Time of Flight (TOA - TOT)
TOF = (TOA - TOT) / 1000 

# Chute inicial
r = np.array([-6420., -6432., 6325.])

# Gradiente descendente
for _ in range(1000):
    G = calcula_gradiente(r_positions, r, TOF)
    r = r - alpha * G

print("Posição final estimada (km):", r)