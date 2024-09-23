import os,cv2
import json
import argparse
import numpy as np

def process_video_to_frames(videopath, output_videopath, annotations, action_dict, actionname):
    video_cap = cv2.VideoCapture(videopath)
    success, image = video_cap.read()
    rate = video_cap.get(5)
    framenumber = video_cap.get(7)
    if not success or framenumber < 61:
        return 0
    anno = {}
    anno['dataset_name'] = 'kinetics700'
    anno['video_name'] = videopath.split("/")[-1]
    anno['file_name'] = ['{}/{}_{:02d}.jpg'.format(actionname, anno['video_name'].split(".mp4")[0], i) for i in range(16)]
    anno['action_annotations'] = [{"action_name": actionname, "action_id": action_dict[actionname]}]
    annotations.append(anno)
    count = 0
    current_frame = np.random.randint(framenumber - 60)
    while success:
        frame_step = 4 
        temp = video_cap.get(0)
        cv2.imwrite(os.path.join(output_videopath, anno['file_name'][count]), image)
        count += 1
        if count == 16:
            break
        current_frame += frame_step
        video_cap.set(cv2.CAP_PROP_POS_FRAMES,  current_frame)
        success, image = video_cap.read()

    if count < 16:
        print("wrong count", videopath)
    return 1

def parse_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_videos', default=117266, type=int, help='Number of sample videos')
    parser.add_argument('--file_path', type=str, help='Path of kinetics-700 videos', default='./data/action/kinetics-700/videos')
    parser.add_argument('--output_path', type=str, default='./data/action/kinetics-700')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(os.path.join(args.output_path, "images")):
        os.mkdir(os.path.join(args.output_path, "images"))
    if not os.path.exists(os.path.join(args.output_path, "annotations")):
        os.mkdir(os.path.join(args.output_path, "annotations"))
    annotations = []
    action_dict = {}
    num_action = 0
    for actionname in os.listdir(args.file_path):
        action_dict[actionname] = num_action
        num_action += 1
    num_actions = len(action_dict)
    quotient = args.num_videos // num_actions
    remainder = args.num_videos % num_actions
    num_videosamples = [quotient] * (num_actions - remainder) + [quotient + 1] * remainder
    for actionname in os.listdir(args.file_path):
        number = 0
        actionpath = os.path.join(args.file_path, actionname)
        for videoname in os.listdir(actionpath):
            videopath = os.path.join(actionpath, videoname)
            output_videopath = os.path.join(args.output_path, "images")
            if not os.path.exists(os.path.join(output_videopath, actionname)):
                os.mkdir(os.path.join(output_videopath, actionname))
            number += process_video_to_frames(videopath, output_videopath, annotations, action_dict, actionname)
            if number >= num_videosamples[action_dict[actionname]]:
                break
    print("sample videos: ", len(annotations))
    with open(os.path.join(args.output_path, "annotations/train_kinetics700.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(annotations, ensure_ascii=False))
