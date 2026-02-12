import zipfile
import glob
import os
import subprocess
import copy
os.system("git ls-files")
list_of_files = subprocess.check_output("git ls-files", shell=False).splitlines()
all_files = glob.glob('**/*.*',recursive=True)

# remove some files that are defined in other ignore directly, only stored localaly not on gitopen .gitignore file and read it
if os.path.exists('.ignore_for_external'):
    with open('.ignore_for_external') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

out_file = '../perceive_em_12.18.25_External.zip'

temp_list_of_files = copy.copy(list_of_files)

for file in temp_list_of_files:
    for line in lines:
        if line.lower() in file.decode('utf-8').lower():
            try:
                list_of_files.remove(file)
                print('removing file:', file.decode('utf-8'))
            except ValueError:
                print('file no longer in list:', file.decode('utf-8'))

zip = zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED)
for file in list_of_files:
    zip.write(file.decode('utf-8'))
zip.close()
