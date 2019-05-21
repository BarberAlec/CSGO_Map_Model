import numpy as np
import keras
from keras import backend as K
import matplotlib.pyplot as plt

num_epochs = 1
num_games = 1

batch_size = 128

network_z, network_x, network_y = 10, 128, 128
blob_size = 5.0

debug = True

# Dumb queriable wrapper for games.
class Game:
    def __init__(self, game_id):
        self.current_frame = 0
        print("Loading game", game_id)

    # Get a batch of frames.
    # This only gives the network target output, we have to generate it's input ourselves.
    def get_next_batch():
        Y = np.empty((batch_size, network_z, network_x, network_y))

        # Only bother with full batches
        if current_frame + batch_size > len(something):
            return None

        for i in range(batch_size):
            Y[i,:] = get_frame(current_frame)
            current_frame += 1
        return Y

    # Generate frame i
    def get_frame(i):
        f = np.zeros((network_z, network_x, network_y))

        for p_ix in range(network_z):
            if is_alive(p_ix, i):
                put_heatmap(f[p_ix], x, y)

        return np.empty()

    # Put a blob in the required place of a 2d matrix
    # Can do this way faster, but it's fine for now..
    @staticmethod
    def put_heatmap(heatmap, x, y):
        center_x, center_y = center
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


    def is_visible(player_ix, frame_ix):
        return True

    def is_dead(player_ix, frame_ix):
        return True

def game_loader():
    for game_id in range(num_games):
        yield Game(game_id)


# Take a thicc vector of each player separately, and plot a viewer-friendly image of em.
def show_frame(X):
    # TODO: Take first/ last 5 and colour separately.
    a = np.sum(X, axis=0)
    print("Image shape:", a.shape)
    plt.imshow(a)

def train_network():
    # Training loop
    for epoch in range(num_epochs):
        game_it = game_loader()

        for game in game_it:

            # Initial state is just the very first frame of the game,
            # which is known to both sides anyway(!)
            # After the first loop, it'll be a prediction of the first frame in the batch.
            c_frame = game.get_frame(0)
            game.current_frame += 1

            # Frame index for input vs output.
            # k(ix) = global knowledge from frame ix

            # input (X)       | output (Y)
            # ----------------------------
            # 0               | 1
            # pred(0) + k(1)  | 2
            # pred(x1) + k(2) | 3
            # pred(x2) + k(3) | 4

            while True:
                # Each loop here creates a new training batch and updates the network based on it.

                starting_frame = game.current_frame
                # Generate target output for all game 'frames'.
                Y = game.get_next_batch()
                if batch is None:
                    break

                # Make batch input
                X = np.empty((batch_size, network_z, network_x, network_y))
                for f_ix in range(batch_size):
                    # Add current frame to batch (copying)
                    # This way, X always lags behind Y by one frame.
                    X[f_ix,:] = c_frame

                    # Predict next frame (f_ix) based on current one
                    c_frame = model.predict(c_frame)

                    # Update with any knowledge we have. (i.e. player visible or dead)
                    ix = starting_frame + f_ix  # current frame index
                    for p_id in network_z:
                        if p_id < 5 or game.is_dead(player, ix) or game.is_visible(player, ix):
                            # If player is dead or visible, we now know what it's doing...
                            # update it's layer to the ground truth. (of what will be effectively the previous frame)
                            c_frame[p_id,:] = Y[f_ix, p_id]



                    # Debug!! View input vs target output for training
                    if debug:
                        plt.subplot(1, 2, 1)
                        show_frame(X[f_ix])
                        plt.subplot(1, 2, 2)
                        show_frame(Y[f_ix])
                        # Wait small amount? Somehow actually fucking show the thing?

                # Train model!!! on our newly created X and Y batches (:D)
                loss = model.train_on_batch(X, Y)
                print(loss)


def visualise_network(game_id):
    game = Game(game_id)

    # Get first frame
    c_frame = game.get_frame(0)

    for ix in range(num_frames):
        c_frame = model.predict(c_frame)

        # display (always predicted, cause you learn more)
        # But more accurate to show after updated with gt.
        show_frame(c_frame)

        # update with g/truth
        gt = game.get_frame(ix)
        for p_id in network_z:
            if game.is_dead(player, ix) or game.is_visible(player, ix):
                # If player is dead or visible, we now know what it's doing...
                # Update it's layer to the ground truth.
                c_frame[p_id,:] = gt[p_id]

