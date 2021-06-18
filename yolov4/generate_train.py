import os

image_files = []
os.chdir(os.path.join("data", "train"))
for filename in os.listdir(os.getcwd()):
    if not filename.endswith(".txt"):
        image_files.append("data/train/" + filename)
os.chdir("..")
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")