
Counter Strike Global Offensive (CSGO) is a popular competitve online, team based shooter. This is project is attempting to parse data from replay files (known as DEMOs) of past games to make a player prediction model. The model takes as inputs the counter terrorists team positions (blue team) and the postitions of the enemy terrorist's (red team) positions when they are visible.

Below is an example of a round of CSGO between CMSTORM and another professional team. Black rings around a player indicates that they are visible (and white rings shows that they are not visible). When a players icon turns black, they have died. This game mode does not have any respawn.

![example animation gif](https://github.com/BarberAlec/CSGO_Map_Model/blob/master/ASSETS/de_dust_no_heat_example.gif)

~A problem currently being encountered is the lack of archive of DEMO files. As far as I am aware, no central archive is available. All DEMO files used in this project have been sourced through reddit where people have uploaded thier personal games for critics to analyse or from an analysis website which is no longer online. If anyone knows of a large archive, please let me know :).~

Currently I am using demos found at https://play.esea.net/. 

Currently working on rebuilding parsing functionality using Valve's parser https://github.com/ValveSoftware/csgo-demoinfo as original parsing functonality nolonger appears to work.


# Objectives

- [X] DEM file parser

- [X] Round Animator

- [X] NN model infrastructure + model

- [ ] Round Animator heatmap support (to display model predicitions)

- [ ] Larger dataset
