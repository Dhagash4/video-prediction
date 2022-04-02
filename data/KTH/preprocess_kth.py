import imageio
import numpy as np
import os
import pickle
import re

from PIL import Image

CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

""" Dataset are divided according to the instruction at: http://www.nada.kth.se/cvap/actions/00sequences.txt """

TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
VAL_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]

def make_raw_dataset(dataset="train"):
    if dataset == "train":
        ID = TRAIN_PEOPLE_ID
    elif dataset == "dev":
        ID = VAL_PEOPLE_ID
    else:
        ID = TEST_PEOPLE_ID

    frames_idx = parse_sequence_file()

    data = []

    for category in CATEGORIES:
        folder_path = os.path.join("data/KTH/kth", category)
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            filepath = os.path.join("data/KTH/kth", category, filename)
            person_id = int(filename.split("_")[0][6:])
            if person_id not in ID:
                continue

            vid = imageio.get_reader(filepath, "ffmpeg")

            frames = []

            for i, frame in enumerate(vid):
                ok = False
                for seg in frames_idx[filename]:
                    if i >= seg[0] and i <= seg[1]:
                        ok = True
                        break
                if not ok:
                    continue

                frame = Image.fromarray(np.array(frame))
                frame = frame.convert("L")
                frame = np.array(frame.getdata(),
                                 dtype=np.uint8).reshape((120, 160))
                frame = np.array(Image.fromarray(frame).resize(size=(64, 64)))

                frames.append(frame)

            data.append({
                "filename": filename,
                "category": category,
                "frames": frames    
            })

    pickle.dump(data, open("data/KTH/kth/%s.p" % dataset, "wb"))
    
def parse_sequence_file():
    print("Parsing sequences.txt")
    
    txt_path = os.path.join(os.getcwd(), "data/KTH/sequences.txt")
    with open(txt_path, 'r') as content_file:
        content = content_file.read()

    content = re.sub("[\t\n]", " ", content).split()
    
    frames_idx = {}
    current_filename = ""

    for s in content:
        if s == "frames":
            continue
        elif s.find("-") >= 0:
           
            if s[len(s) - 1] == ',':
               
                s = s[:-1]

            
            idx = s.split("-")

            if not current_filename in frames_idx:
                frames_idx[current_filename] = []
            frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            current_filename = s + "_uncomp.avi"

    return frames_idx

if __name__ == "__main__":
    print("Making raw train dataset")
    make_raw_dataset(dataset="train")

    print("Making raw dev dataset")
    make_raw_dataset(dataset="val")
    
    print("Making raw test dataset")
    make_raw_dataset(dataset="test")
