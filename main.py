from csgo import csgo, game_round
'''
Orginal python package is no longer supported, now using a C++ package (https://github.com/ValveSoftware/csgo-demoinfo) by Valve.

All parsing of demo files is done by this package (modified somewhat by me). This file calls the parser for every file in the CSGO_DEM_FILES
and puts the result in CSGO_PARSED_FILES.
'''


def main():
    a = csgo()

    # Unzip and Parse all demos
    # a.unzipDemoFiles(directory)
    # a.parseDemos('CSGO_DEM_FILES_test')
    
    #a.printRound('esea_match_16104611.csv', round_num=0)
    a.loadGames(source_directory='esea_match_16104611.csv')

if __name__ == '__main__':
    main()
