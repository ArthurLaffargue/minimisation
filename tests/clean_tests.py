import os


listdir = os.listdir()

for file in listdir :

    if not( os.path.isdir(file) or file.endswith(".py") ):
        os.remove(file)
        print("REMOVE : ",file)
