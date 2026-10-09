import numpy as np
def quat_xyzw_to_wxyz(qxyzw):
    x, y, z, w= qxyzw
    return np.array([w, x, y, z], dtype=float) / np.linalg.norm(qxyzw)

def quat_wxyz_to_xyzw(qwxyz):
    w, x, y, z = qwxyz
    return np.array([x, y, z, w], dtype=float) / np.linalg.norm(qwxyz)

def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2 
    ], dtype=float) / np.linalg.norm(q1) / np.linalg.norm(q2)

def quat_inv_wxyz(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float) / np.linalg.norm(q)

def quat_to_axis_angle_wxyz(q, eps=5e-4):
    w, x, y, z = q
    if w > 1: q = q / np.linalg.norm(q)
    angle = 2*np.arccos(w)
    s = np.sqrt(max(1-w*w, 0.0))
    if s < eps:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return np.array([x/s, y/s, z/s]), angle

def axis_angle_to_quat_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([0,0,0,1], dtype=float)
    axis = axis / n
    s = np.sin(angle/2.0)
    return np.array([np.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)
