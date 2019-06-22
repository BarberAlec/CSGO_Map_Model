import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Implemented animation using actual animation lib, now exports to mp4
# for easier veiwing :)
def __begin_animation(game_round, filename):
    num_frames = max([len(p.states) for p in game_round.players])
    num_players = len(game_round.players)
    stats = np.array([play.states for play in game_round.players])
    
    # Turn interactive plotting off (stops figure from launching)
    plt.ioff()
    
    # 7*7 Inch display
    fig = plt.figure(figsize=(7,7))
    
    #Load and display map
    img = plt.imread("de_dust_map.png")
    plt.imshow(img, extent=[-2600,2100,-1200,3200])
    
    x=[0]*num_players
    y=[0]*num_players
    
    # checks for spectators and others should have been conducted before this
    team_c = ['cyan' if play.m_team == 'CT' else 'pink' for play in game_round.players]

    scat = plt.scatter(x,y,c=team_c,edgecolors='white')
    ani = animation.FuncAnimation(fig, __update_plot, frames=range(num_frames),
                                 fargs=(stats, team_c, scat),blit=True)
    
    # Using high FPS because a man has things to do and people to see.
    ani.save(filename+'.mp4', writer='ffmpeg',fps=30);
    
def __update_plot(i,data, team_c,scat):
    scat.set_offsets(list(data[:,i,0]))
    scat.set_edgecolor(['black' if spot else 'white' for spot in data[:,i,1]])
    
    # Scatter does not support changing markers for individual points...
    # These lines check rgb values to test what team a marker belongs to
    face_cols = team_c
    for idx in range(len(scat.get_offsets())):
        if data[idx,i,2]==True:
            face_cols[idx] = 'black'
            
    '''scat.set_facecolor(['blue' if dead and not (scat.get_facecolor()==1)[0][0] 
                        else 'cyan' for dead in data[:,i,2]])
    scat.set_facecolor(['red' if dead and (scat.get_facecolor()==1)[0][0] 
                        else 'pink' for dead in data[:,i,2]])'''
    scat.set_facecolors(face_cols)
    return scat,
    
def AnimateRound(r,filename):
    __begin_animation(r,filename)
    