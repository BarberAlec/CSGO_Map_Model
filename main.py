import zipfile
import os
import shutil

def unzipDemoFiles(directory):
    '''
    Takes all the zip files in the given directory, extracts them and pulls the demo into the main folder
    '''

    for idx, filename in enumerate(os.listdir(directory)):
        # Unzip
        if filename.endswith(".zip"):
            new_folder_file = os.path.join(directory,filename.split('.')[0])
            with zipfile.ZipFile(os.path.join(directory,filename), 'r') as zip_ref:
                zip_ref.extractall(new_folder_file)
        else:
            continue
        
        # Take demo out of folder
        source_demo_path = os.path.join(new_folder_file,filename.split('.')[0]+'.dem')
        dest_demo_path = directory

        shutil.move(source_demo_path, dest_demo_path)  

        # delete zip folder (now empty)
        os.removedirs(new_folder_file)


def parseDemos(directory):
    '''
    Iterates through demo files in given directory, parsing each and creating an assocaited CSV
    '''

    # Load demos one at a time
    for idx, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".dem"):
            file_dir = os.path.join(directory,filename)
            # Launch executable 

def main():
    directory = 'CSGO_DEM_FILES'

    unzipDemoFiles(directory)


if __name__ == '__main__':
    main()