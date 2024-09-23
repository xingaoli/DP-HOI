import os,cv2
import json
import argparse

def process_video_to_frames(videopath, output_path, annotations, action_dict, actionname, frames_num):
    video_cap = cv2.VideoCapture(videopath)
    success, image = video_cap.read()
    rate = video_cap.get(5)
    framenumber = video_cap.get(7)
    duration = framenumber / rate
    count = 0
    if duration >= 2:
        while success:
            anno = {}
            frame_name = 'Haa500_{:05d}.jpg'.format(frames_num)
            anno['dataset_name'] = 'Haa500'
            anno['file_name'] = frame_name
            anno['action_annotations'] = [{"action_name": actionname, "action_id": action_dict[actionname]}]
            annotations.append(anno)

            temp = video_cap.get(0)
            cv2.imwrite(os.path.join(output_path, frame_name), image)
            frames_num += 1
            count += 1
            video_cap.set(cv2.CAP_PROP_POS_MSEC, 0.5 * 1000 * count)
            success, image = video_cap.read()
    else:
        current_frame = 0
        while success:
            anno = {}
            frame_name = 'Haa500_{:05d}.jpg'.format(frames_num)
            anno['dataset_name'] = 'Haa500'
            anno['file_name'] = frame_name
            anno['action_annotations'] = [{"action_name": actionname, "action_id": action_dict[actionname]}]
            annotations.append(anno)

            frame_step = (framenumber - 1) // 3 
            temp = video_cap.get(0)
            cv2.imwrite(os.path.join(output_path, frame_name), image)
            if frame_step == 0:
                break
            frames_num += 1
            count += 1
            if count == 4:
                break
            current_frame += frame_step
            video_cap.set(cv2.CAP_PROP_POS_FRAMES,  current_frame)
            success, image = video_cap.read()
    return frames_num

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='Path of Haa500 videos', default='./data/action/haa500/videos')
    parser.add_argument('--output_path', type=str, default='./data/action/haa500')
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
    frames_num = 0
    for actionname in os.listdir(args.file_path):
        actionpath = os.path.join(args.file_path, actionname)
        for videoname in os.listdir(actionpath):
            videopath = os.path.join(actionpath, videoname)
            frames_num = process_video_to_frames(videopath, os.path.join(args.output_path, "images"), annotations, action_dict, actionname, frames_num)
    print("total frames: ", frames_num)
    with open(os.path.join(args.output_path, "annotations/train_haa500.json"), 'w', encoding='utf-8') as file:
        file.write(json.dumps(annotations, ensure_ascii=False))