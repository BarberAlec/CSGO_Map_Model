
Counter Strike Global Offensive (CSGO) is a popular competitve online, team based shooter. This is project is attempting to parse data from replay files (known as DEMOs) of past games to make a player prediction model. The model takes as inputs the counter terrorists team positions (blue team) and the postitions of the enemy terrorist's (red team) positions when they are visible.

Below is an example of a round of CSGO between CMSTORM and another professional team. Black rings around a player indicates that they are visible (and white rings shows that they are not visible). When a players icon turns black, they have died. This game mode does not have any respawn.

![example animation gif](https://github.com/BarberAlec/CSGO_Map_Model/blob/master/ASSETS/de_dust_no_heat_example.gif)


# Objectives

- [X] Create DEM file parser

- [X] Build simply print round functionality

- [X] Create basic structure for model training

- [ ] More advanced print round functionality that includes heatmap support

- [ ] Use large dataset to train model
