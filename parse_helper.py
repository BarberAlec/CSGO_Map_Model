'''
Orginal python package is no longer supported, now using a C++ package (https://github.com/ValveSoftware/csgo-demoinfo) by Valve.

All parsing of demo files is done by this package (modified somewhat by me). This file calls the parser for every file in the CSGO_DEM_FILES
and puts the result in CSGO_PARSED_FILES.
'''
import os
import subprocess

def parse_demos_by_dir(dir):
    path2parser = r'demoinfogo/Debug/demoinfogo.exe'
    for idx, filename in enumerate(os.listdir(dir)):
        if filename.endswith(".dem"):
            print(f"Parsing: {os.path.join(dir,filename)}")
            output = subprocess.Popen([path2parser,os.path.join(dir,filename),'-gameevents','-extrainfo'], stdout=subprocess.PIPE).communicate()[0]
            
            # Saving to file
            print(f"Saving parsed {filename} to file")
            text_file = open(f"CSGO_PARSED_FILES\{filename.split('.')[0]}.csv", "w")
            text_file.write(output.decode("utf-8") )
            text_file.close()

def main():
    parse_demos_by_dir('CSGO_DEM_FILES_test')



if __name__ == '__main__':
    main()