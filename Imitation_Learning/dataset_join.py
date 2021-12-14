import os
import pandas as pd
import cv2

PARENT_DIR = os.sep.join(os.getcwd().split(os.sep)[:-1])
INPUT_PATH = os.path.join(PARENT_DIR, 'datasets')
IMAGES_PATH = os.path.join(PARENT_DIR, 'dataset', 'images')
CSV_PATH = os.path.join(PARENT_DIR, 'dataset', 'dataset.csv')
count = 0
filename_csv = []
throttle_csv = []
steering_csv = []
command_csv = []

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for dataset_dir in os.listdir(INPUT_PATH):
    df = pd.read_csv(os.path.join(INPUT_PATH, dataset_dir, 'dataset.csv'))
    throttle_csv.extend(df['throttle'])
    steering_csv.extend(df['steering'])
    command_csv.extend((df['command']))

    for filename in os.listdir(os.path.join(INPUT_PATH, dataset_dir, 'images')):
        count += 1
        output_filename = '{:05d}.jpg'.format(count)
        image = cv2.imread(os.path.join(INPUT_PATH, dataset_dir, 'images', filename))
        cv2.imwrite(os.path.join(IMAGES_PATH, output_filename), image)
        filename_csv.append(output_filename)

# Write csv dataframe
columns = ['filename', 'throttle', 'steering', 'command']
dataset_dict = {'filename': filename_csv,
                'throttle': throttle_csv,
                'steering': steering_csv,
                'command': command_csv}
dataset = pd.DataFrame(dataset_dict)
dataset.to_csv(CSV_PATH, index=False, columns=columns)
