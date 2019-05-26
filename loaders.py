#!/usr/bin/env python
import sys
from demoparser.demofile import DemoFile
import matplotlib.pyplot as plt
import time


#Encapsulates all information about each individual player
class Person:
    def __init__ (self, name, team):
        self.m_name = name
        self.m_team = team

        self.is_dead = False

        self.states = []

    def add_position(self, pos, spotted):
        self.states.append((pos, spotted, self.is_dead))

    def die(self):
        self.is_dead = True


#Encapsulates parsed information from DEM file about each round, no interesting processing is conducted here.
class Round:
    def __init__ (self, rid):
        self.m_rid = rid

# Dumb queriable wrapper for games.
class Game:
    def __init__(self, game_id):
        self.current_frame = 0
        print("Loading game", game_id)
        # TODO: Use game_id to determine file

        #Open and extract DEM file
        data = open("deRust.dem", 'rb').read()
        d = DemoFile(data)
        outFile = open('out.txt', 'w+')


        current_round = 0
        round_active = False

        # Internal state classes used by the parser! CAREFUL HERE :O
        s_players = []

        #Callback Functions
        def tick_start(msg):
            #If first round has not begun yet, return
            if not round_active:
                return

            # print(msg) # Updates every 8 ticks?

            # Save from their player state to ours
            for o_pl, th_pl in zip(self.m_rounds[-1].players, s_players):
                spotted = th_pl.get_prop('DT_BaseEntity', 'm_bSpotted')

                pos = int(th_pl.position['x']), int(th_pl.position['y'])
                o_pl.add_position(pos, spotted)

        def round_start(msg, msg1):
            nonlocal round_active
            nonlocal s_players

            round_active = True
           
            print("Round ", current_round)
            self.m_rounds.append(Round(current_round))

            #In professional matches, there are sometimes players that have no team or are spectators
            #Store refrences to the internal state of the parser, so we can grab em again later.
            s_players = [p for p in d.entities.players
                         if p.team and p.team.name != 'Spectator']

            s_players = sorted(s_players, key=lambda p: p.team.name)

            # Generate our players
            players = [Person(p.name, p.team.name) for p in s_players]
            # And add to round
            # [self.m_rounds[-1].add_player(p) for p in players]
            self.m_rounds[-1].players = players


        def round_end(event, msg):
            nonlocal current_round
            nonlocal round_active

            round_active = False
            current_round += 1

        def death(event, msg):
            #Stolen bit of code, left in unused features...
            for idx, key in enumerate(event['event'].keys):
                if key.name == 'userid':
                    user_id = msg.keys[idx].val_short
                    victim = d.entities.get_by_user_id(user_id)

            if victim and round_active:
                # Find player index
                for p in self.m_rounds[-1].players:
                    if p.m_name == victim.name:
                        p.die()



                # print("Player: ",victim.name," died at tick: ",d.current_tick)
                # self.m_rounds[-1].update_player_position(victim.name,int(victim.position['x']), int(victim.position['y']), victim.view_angle, False,'Y')

        #Add callback events (calls function whenever event occurs in file)
        d.add_callback('tick_start', tick_start)
        d.add_callback('round_start', round_start)
        d.add_callback('round_end', round_end)
        d.add_callback('player_death', death)

        #create list of rounds
        self.m_rounds = []

        #Begin parseing file (may take a while)
        d.parse()

        #Close file
        outFile.close()
        print("Finished loading game!")

        # TODO: Sort players based on team!


    def is_visible(player_ix, frame_ix):
        return True

    def is_dead(player_ix, frame_ix):
        return True
