import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math


#WARNING hard coding network sizes here.... CAREFULL
network_z, network_x, network_y = 10, 128, 128
blob_size = 2.5

frame = None

# Implemented animation using actual animation lib, now exports to mp4
# for easier veiwing :)
def __begin_animation_no_heat(game_round, filename):
    num_frames = max([len(p.states) for p in game_round.players])
    num_players = len(game_round.players)
    player_states = np.array([play.states for play in game_round.players])
    
    # Turn interactive plotting off (stops figure from launching)
    plt.ioff()
    
    # 7*7 Inch display
    fig = plt.figure(figsize=(7,7))
    
    #Load and display map
    img = plt.imread("Assets/de_dust_map.png")
    plt.imshow(img, extent=[-2600,2100,-1200,3200])
    
    x=[0]*num_players
    y=[0]*num_players
    
    # checks for spectators and others should have been conducted before this
    team_c = ['blue' if play.m_team == 'CT' else 'red' for play in game_round.players]

    scat = plt.scatter(x,y,c=team_c,edgecolors='white')
    ani = animation.FuncAnimation(fig, __update_plot, frames=range(num_frames),
                                 fargs=(player_states, team_c, scat),blit=True)
    
    # Using high FPS because a man has things to do and people to see.
    ani.save(filename+'.mp4', writer='ffmpeg',fps=30);
    
def __update_plot(i,data, team_c,scat):
    scat.set_offsets(list(data[:,i,0]))
    scat.set_edgecolor(['black' if spot else 'white' for spot in data[:,i,1]])
    
    # Scatter does not support changing markers for individual points...
    face_cols = team_c
    for idx in range(len(scat.get_offsets())):
        if data[idx,i,2]==True:
            face_cols[idx] = 'black'
    scat.set_facecolors(face_cols)
    #TODO: Distinguish between dead players of opposing teams
    return scat,
    
    
def __begin_animation_with_heat(game_round, filename, model):
    # Dont think this line is needed as new loaders method gives all players equal number of states now...
    num_frames = max([len(p.states) for p in game_round.players])
    num_players = len(game_round.players)
    
    # Container for heatmap frame predictions (see __update_plot_with_heat)
    global frame
    frame = None
    
    # Numpy array of lists of player states for each tick of round
    player_states = np.array([play.states for play in game_round.players])
    
    # Turn interactive plotting off (stops figure from launching)
    #plt.ioff()
    
    # 7*7 Inch display
    fig = plt.figure(figsize=(7,7))
    
    #Load and display map (backgroud)
    img = plt.imread("de_dust_map.png")
    plt.imshow(img, extent=[-2600,2100,-1200,3200])
    
    x=[0]*num_players
    y=[0]*num_players
    
    # checks for spectators and others should have been conducted before this
    team_c = ['blue' if play.m_team == 'CT' else 'red' for play in game_round.players]

    scat = plt.scatter(x,y,c=team_c,edgecolors='white')
    ani = animation.FuncAnimation(fig, __update_plot_with_heat, frames=range(num_frames),
                                 fargs=(player_states, team_c, scat, model),blit=True)
    
    # Using high FPS because a man has things to do and people to see.
    ani.save(filename+'.mp4', writer='ffmpeg',fps=30);
    
def __update_plot_with_heat(i,data,team_c,scat,model):
    true_pos = list(data[:,i,0])
    global frame
    
    #import pdb; pdb.set_trace()
    #Generate frame if first of round
    if not frame:
        print('First frame: Generating frame from data')
        frame = [__frame_gen(i,data)]
    
    #Create heatmap to overlay on map
    heat_map,pred_frame = heat_map_pred(model,frame[0],data,i)
    #Currently using list to pass by reference, TODO: Find better method :l
    frame[0] = pred_frame
    #import pdb; pdb.set_trace()
    
    scat.set_offsets(true_pos)
    scat.set_edgecolor(['black' if spot else 'white' for spot in data[:,i,1]])
    
    # Scatter does not support changing markers for individual points...
    face_cols = team_c
    for idx in range(len(true_pos)):
        if data[idx,i,2]==True:
            face_cols[idx] = 'black'
    scat.set_facecolors(face_cols)
    plt_heat = plt.imshow(heat_map)
    #TODO: Distinguish between dead players of opposing teams
    return scat,
    
    
def __put_heatmap(heatmap, center):
    #TODO make a new file to put this function to avoid duplication(currently in main notebook)
    center_x, center_y = center
    
    # Rescale ..
    x1, x2, y1, y2 = (-2600., 2100., -1200., 3200.)

    center_x = (center_x - x1) / (x2 - x1) * network_x
    center_y = (center_y - y1) / (y2 - y1) * network_y
    
    height, width = heatmap.shape

    th = 4.6052
    
    delta = math.sqrt(th * 2)

    # Vectorize
    sigma = blob_size
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width - 1, center_x + delta * sigma))
    y1 = int(min(height - 1, center_y + delta * sigma))

    exp_factor = 1 / 2.0 / sigma / sigma
    arr_heatmap = heatmap[y0:y1 + 1, x0:x1 + 1] # Not nessasary unless we have > 1 blob per input
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    
    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    
    
def __frame_gen(idx, data):
    
    Y = np.zeros((network_z, network_x, network_y))
    num_play = len(data)
    #import pdb; pdb.set_trace()
    for p_ix in range(num_play):
        (pos, vis, dead) = data[p_ix][idx]
        #Only add friendly players that are not dead (TODO: create more robust method of determining team)
        if p_ix < num_play/2 and not dead:
            __put_heatmap(Y[p_ix], pos)
    Y = np.moveaxis(Y, 0, -1)
    return Y
    
    
def heat_map_pred(mdl, frm, state, idx):
    #Make prediction about current frame using guess(plus know info) about previous frame
    pred = mdl.predict(np.expand_dims(frm, axis=0))[0]
    
    #Deduce frame truth for this tick
    frame_truth = __frame_gen(idx,state)
    
    #Correct prediction with known information about the system
    for p_id in range(len(state)):
        pos, is_spotted, is_dead = state[p_id][idx]
        if p_id < 5 or is_spotted or is_dead:
            pred[:,:,p_id] = frame_truth[:,:,p_id]
    
    #Flatten prediction into heatmap 2D array
    heat = np.sum(pred,axis=2)
    
    #stretch heat to match map
    x1, x2, y1, y2 = (-2600., 2100., -1200., 3200.)
    x_len,y_len = x2-x1,y2-y1
    x_block_len,y_block_len = round(x_len/network_x),round(y_len/network_y)
    stretch_heat = np.repeat(np.repeat(heat,y_block_len,axis=0),x_block_len,axis=1)
    
    #Return heatmap with prediction about current tick
    return stretch_heat,pred
    
def AnimateRound(r,filename,model=None):
    """Create mp4 animation of round
    If model is specified, then predictive heatmapping will
    be used, otherwise no heatmap will be used.
    """
    if not (model): __begin_animation_no_heat(r,filename)
    else: __begin_animation_with_heat(r,filename,model)
    
    