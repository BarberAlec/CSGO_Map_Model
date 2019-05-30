import matplotlib.pyplot as plt
import numpy as np

def print_single_frame(frame, fig=None, ax=None):
    plt.clf()
    if not fig or not ax:
        fig, ax = plt.subplots()
    fig.clf()
    #There is a better way of doing this but too lazy to find function...
    semi_flat_frame = 0
    for i in range(frame.shape[-1]):
        semi_flat_frame += frame[:,:,i]
        
    
    ax.imshow(semi_flat_frame)
    plt.show()
    #Yea this is broken...
    return [fig, ax]
    
#def print_round(game_round):