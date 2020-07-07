import zipfile
import os
import shutil
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image


class game_round:
    '''
    Encapsulation for all data in a csgo game round this is the largest data structure used in training.

    Note: frame interval is controlled by a higher level structure.
    '''

    def __init__(self, id, map_dim=(128, 128)):
        self.round_id = id
        self.dim = map_dim

        # Frames are in the format of 128x128 'heatmaps'
        self.CT_frames = None
        self.T_frames = None
        self.true_frames = None

    def add_CT_frames(self, frames):
        '''Add game round frames from the perspective of CT team.'''
        self.CT_frames = frames

    def add_T_frames(self, frames):
        '''Add game round frames from the perspective of CT team.'''
        self.T_frames = frames

    def add_true_frames(self, frames):
        '''Add game round frames given total God knowledge.'''
        self.true_frames = frames


class csgo:
    '''
    Project encapsulation, handles parsing of files and printing of rounds.
    '''
    @staticmethod
    def unzipDemoFiles(directory):
        '''
        Takes all the zip files in the given directory, extracts them and pulls the demo into the main folder
        '''

        for idx, filename in enumerate(os.listdir(directory)):
            # Unzip
            if filename.endswith(".zip"):
                new_folder_file = os.path.join(
                    directory, filename.split('.')[0])
                with zipfile.ZipFile(os.path.join(directory, filename), 'r') as zip_ref:
                    zip_ref.extractall(new_folder_file)
            else:
                continue

            # Take demo out of folder
            source_demo_path = os.path.join(
                new_folder_file, filename.split('.')[0]+'.dem')
            dest_demo_path = directory

            shutil.move(source_demo_path, dest_demo_path)

            # delete zip folder (now empty)
            os.removedirs(new_folder_file)

    @staticmethod
    def parseDemos(directory):
        '''
        Given a directory, parse all demos and create csvs and save in CSGO_PARSED_FILES
        '''
        path2parser = r'demoinfogo/Debug/demoinfogo.exe'
        for idx, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".dem"):
                print(f"Parsing: {os.path.join(directory,filename)}")
                output = subprocess.Popen([path2parser, os.path.join(
                    directory, filename), '-gameevents', '-extrainfo'], stdout=subprocess.PIPE).communicate()[0]

                # Saving to file
                print(f"Saving parsed {filename} to file")
                text_file = open(
                    f"CSGO_PARSED_FILES\{filename.split('.')[0]}.csv", "w")
                text_file.write(output.decode("utf-8"))
                text_file.close()

    def __init__(self):
        self.map_extent = (-2486, 2127, -1155, 3455)
        self.animation_multiplier = 20
        self.num_players = None
        self.tick_res = None
        self.csv_folder = 'CSGO_PARSED_FILES'
        self.fig, self.ax = plt.subplots(figsize=(9, 9))

        # Draw Heatmap
        map_extent = (-2486, 2127, -1155, 3455)
        dim = (32, 32)

        dx = (map_extent[1] - map_extent[0])/dim[0]
        dy = (map_extent[3] - map_extent[2])/dim[1]
        y, x = np.mgrid[slice(map_extent[2], map_extent[3] + dy, dy),
                        slice(map_extent[0], map_extent[1] + dx, dx)]
        z = x**2+y**2
        z = z[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        #self.ax.pcolor(x, y, z, cmap='Reds_r', vmin=z_min, vmax=z_max,alpha=0.7)

    def printRound(self, directory, round_num=0, mp4_save_file=None):
        '''
        Takes csv game, processes it and then prints a round.

        directory: location of csv file.

        round_num: round to draw.

        mp4_save_file: if None do not save to file, else save as mp4_save_file.
        '''

        # Load csv data
        round_list = self.readCSVFile(directory)
        self.players, self.players_df = self.sortPlayers(round_list[round_num])

        # Calculate, in millieseconds, how period between data points
        self.prev_posX = np.zeros(self.num_players)
        self.prev_posY = np.zeros(self.num_players)
        inter = 1000 * 1/64 * self.tick_res * 1/self.animation_multiplier

        self.ani = animation.FuncAnimation(
            self.fig, self._update, init_func=self._setup_plot, interval=inter, blit=True,
            frames=self.players_df[self.players[0]].shape[0], repeat=False)
        plt.show()
        if mp4_save_file:
            self.saveMP4(mp4_save_file)

    def loadGames(self, source_directory='CSGO_PARSED_FILES', dest_directory='CSGO_PARSED_ROUNDS'):
        '''
        Iterates through all the CSVs in the source directory, processes them into game_round objects,
        saves them as pickle files and saves to destination directory.
        '''

        if source_directory.endswith('.csv'):
            filename = source_directory
            rounds = self.readCSVFile(filename)
            for r in rounds:
                frames_CT = self._gen_frames(r, 'CT', dim=(64, 64))
                frames_T = self._gen_frames(r, 'T', dim=(64, 64))
                frames_full = self._gen_frames(r, dim=(64, 64))

                ensemble_frames = np.zeros((3,)+frames_CT.shape)
                ensemble_frames[0, :, :, :] = frames_CT
                ensemble_frames[1, :, :, :] = frames_T
                ensemble_frames[2, :, :, :] = frames_full

                # print(sum(frames_CT[36]))
                # print(sum(frames_T[36]))
                # print(sum(frames_full[36]))
                test_img = frames_full[36]
                pic = (test_img+5)*255/10
                img = Image.fromarray(pic)
                img.show()

                np.save(os.path.join(dest_directory,
                                     f"{filename.split('.')[0]}.npy"), ensemble_frames)
                return

        for filename in os.listdir(source_directory):
            if filename.endswith(".csv"):
                rounds = self.readCSVFile(
                    os.path.join(source_directory, filename))
                for r in rounds:
                    frames_CT = self._gen_frames(r, 'CT')
                    frames_T = self._gen_frames(r, 'T')
                    frames_full = self._gen_frames(r)

                    ensemble_frames = np.zeros((3,)+frames_CT.shape)
                    ensemble_frames[0, :, :, :] = frames_CT
                    ensemble_frames[1, :, :, :] = frames_T
                    ensemble_frames[2, :, :, :] = frames_full
                    np.save(os.path.join(dest_directory,
                                         f"{filename.split('.')[0]}.npy"), ensemble_frames)
                    return

    def _gen_frames(self, round, team_perspective=None, ticks_per_frame=64, dim=(32, 32)):
        '''
        Givena a df round, parse the file and return a set of frames.
        '''
        start_tick = round.iloc[0]['tick']
        end_tick = round.iloc[-1]['tick']

        ticks_2_sample = np.arange(start_tick, end_tick, ticks_per_frame)
        frames = self._subsample_wBluredSplotted(
            round, ticks_2_sample, ticks_per_frame, team_perspective, dim)

        return frames

    def _gen_single_frame(self, df, team_perspective, dim=(32, 32)):
        '''
        Draw a frame
        '''
        frame = np.zeros(dim)

        leftX, rightX, bottomY, topY = self.map_extent
        dx = (rightX - leftX)/dim[0]
        dy = (topY - bottomY)/dim[1]

        def get_coord(posX, posY):
            row = int((topY-posY)//dy)
            col = int((posX - leftX)//dx)
            if row >= dim[0]:
                row = 31
            return row, col

        if not team_perspective:
            # Ground truth
            for _, player in df.iterrows():
                team = player['team']
                r, c = get_coord(player['x'], player['y'])
                if team == 'CT':
                    frame[r, c] += 1
                else:
                    frame[r, c] -= 1

            # pic = (frame+1)*255/2
            # img = Image.fromarray(pic)
            # img.show()
            return frame

        for _, player in df.iterrows():
            team = player['team']
            if team == team_perspective or player['spot'] == 1:
                if player['spot'] == 1:
                    print('something')
                r, c = get_coord(player['x'], player['y'])
                if team == 'CT':
                    frame[r, c] += 1
                else:
                    frame[r, c] -= 1
        return frame

    def _subsample_wBluredSplotted(self, round, ticks_2_sample, ticks_per_frame, team_perspective, dim=(32, 32)):
        '''
        Given a round, blur spot attribute
        '''
        # sort round into seperate player dfs for purpose of blurring
        players, play_df = self.sortPlayers(round)
        frames = np.zeros((len(ticks_2_sample),)+dim)

        # If a player is spotted in a frame length, then make the sample spot = True
        sample_shift = ticks_per_frame//self.tick_res
        for play in players:
            local_df = play_df[play].copy()
            local_df['spot'] = local_df['spot'].rolling(
                window=sample_shift).mean().shift(-sample_shift//2).fillna(0)

            local_df[local_df['spot'] > 0] = 1
            local_df['spot'] = local_df['spot'].astype(int)
            play_df[play] = local_df

            # local_df = play_df[play]
            # play_df[play].loc[:,('spot',)] = play_df[play].loc[:,('spot',)].rolling(
            #     window=sample_shift).mean().shift(-sample_shift//2)

            # local_df[local_df['spot'] > 0] = True
            # play_df[play] = local_df

        # Concat al the players back into one df
        blur_df = pd.concat([play_df[play] for play in players])

        # Sample at the target ticks and make frames
        for idx, tick_sample in enumerate(ticks_2_sample):
            if idx == 0:
                print('at 24')
            frame_df = blur_df[blur_df['tick'] == tick_sample]
            frames[idx, :, :] = self._gen_single_frame(
                frame_df, team_perspective, dim)

        return frames

    def sortPlayers(self, df):
        '''Returns a array of players and a dict of dataframes for each player'''
        players = df['name'].unique()

        players_df = dict()
        id_2_del = []
        for idx, play in enumerate(players):
            pldf = df[df['name'] == play]
            if pldf.iloc[0]['team'] == 'CT' or pldf.iloc[0]['team'] == 'T':
                players_df[play] = pldf
            else:
                id_2_del.append(idx)

        players = np.delete(players, id_2_del)
        self.num_players = len(players)

        self.team_col = np.full(self.num_players, 'r')

        for idx, play in enumerate(players):
            curr = players_df[play]
            if curr.iloc[0]['team'] == 'CT':
                self.team_col[idx] = 'b'

        self.tick_res = int(pldf.iloc[1]['tick']-pldf.iloc[0]['tick'])
        return players, players_df

    def split_df(self, df):
        '''Takes a df of entire csgo game, returns a list of split dfs of each individual round'''
        round_list = []

        # find cols with name 'ROUND START'
        round_values_df = df[df['name'] == 'ROUND START']
        df = df[df['name'] != 'ROUND START']

        # For each round, seperate df
        old_tick = 0
        for _, row in round_values_df.iterrows():
            new_round = df[(df['tick'] <= row['tick'])
                           & (df['tick'] > old_tick)]
            old_tick = row['tick']
            if not new_round.empty:
                round_list.append(new_round)

        # Return list of rounds
        return round_list

    def readCSVFile(self, filename):
        if filename.split('.')[-1] == 'csv':
            df = pd.read_csv(os.path.join(self.csv_folder, filename))
        elif filename.split('.')[-1] != '':
            df = pd.read_csv(os.path.join(self.csv_folder, f"{filename}.csv"))
        else:
            Exception('Bad filename')

        # Split df
        round_list = self.split_df(df)

        return round_list  # [self.sortPlayers(r_df) for r_df in round_list]

    def saveMP4(self, filename):
        '''Save animation mp4 file'''
        self.ani.save(f"{filename}.mp4", writer="ffmpeg")

    def _setup_plot(self):
        '''Setup for animation function'''
        x = np.zeros(self.num_players)
        y = np.zeros(self.num_players)

        self.scat = self.ax.scatter(x, y, edgecolor="k")
        self.scat.set_edgecolor(np.full(self.num_players, 'white'))
        self.ax.axis(self.map_extent)
        self.ax.axes.xaxis.set_ticks([])
        self.ax.axes.yaxis.set_ticks([])

        img = plt.imread("ASSETS/de_dust_map_kaggle.png")
        self.ax.imshow(img, extent=self.map_extent)

        return self.scat,

    def _update(self, i):
        '''updater function for animation'''
        x = np.zeros(self.num_players)
        y = np.zeros(self.num_players)
        colours = self.team_col.copy()
        edgeColours = np.full(self.num_players, 'white')

        for idx, play in enumerate(self.players):
            play_coord = self.players_df[play].iloc[i]
            if play_coord['dead'] > 0:
                x[idx] = play_coord['x']
                y[idx] = play_coord['y']
            else:
                # Hmmm, maybe find a better solution
                x[idx] = self.prev_posX[idx]
                y[idx] = self.prev_posY[idx]
                colours[idx] = 'k'
            if play_coord['spot'] == 1:
                edgeColours[idx] = 'black'
        self.prev_posX = x
        self.prev_posY = y

        # Player positions
        data = np.stack((x, y)).T
        self.scat.set_offsets(data)
        # Team colour and dead/alive status
        self.scat.set_facecolors(colours)
        # Edgecolour shows if they have been spotted
        self.scat.set_edgecolor(edgeColours)
        return self.scat,


if __name__ == '__main__':
    a = csgo()
    #a.loadGames(source_directory='esea_match_16104611.csv')
    a.printRound('esea_match_16104611.csv', round_num=0)
