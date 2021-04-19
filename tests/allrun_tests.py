import os

for file in os.listdir():

    if file.startswith("_") and file.endswith(".py") :
        os.system("python "+file)