import math
def lr_cosine_schedule(
    t,
    a_max,
    a_min,
    T_w,
    T_c,
):
    if(t < T_w):
        a_t = t * a_max / T_w
    elif(t <= T_c):
        a_t = a_min + 0.5 *(1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (a_max - a_min)
    else:
        a_t = a_min
    return a_t