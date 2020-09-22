#Command line tool for evaluating tracking
'''
Input
--prediction_folder path to the txt files of predicted outputs
--video_folder path to the training sequence with gt
--evaluation_type single multiple or cell
--writeout True if video output is needed

Single object tracking evaluation metrics
Accuracy: The accuracy is the average overlap between the predicted [Done]
and ground truth bounding boxes during successful tracking periods. 
Robustness: The robustness measures how many times the tracker loses the target (fails) during tracking.
Expected Average Overlap (EAO) is measured by averaging the accuracy over all videos

Multiple object tracking evalutation metrics
MOTA: Ratio of object configuration errors made by the tracker, false positives, misses, mismatches, over all frames.
MOTP: Sum matched object-hypothesis pairs over all frames, averaged by the total number of matches made
Mostly Tracked: The ratio of ground-truth trajectories that are covered by a track hypothesis for at least 80% 
of their respective life span.
Mostly Lost: The ratio of ground-truth trajectories that are covered by a track hypothesis for at most 20% 
of their respective life span.

Cell Tracking evaluation metrics

Association accuracy assigned to a track (computer-generated) for each frame. The association accuracy was computed as the
number of true positive associations divided by the number of associations in the ground-truth.
Target effectiveness is computed as the number of the assigned track observations
over the total number of frames of the target

All video folders should be in the same dir, to process LaSOT
find . -maxdepth 2  -print -exec mv {} . \;
'''

'''
example
python evaluation.py --predcition_folder ../results/sot_results/TB-100 \
--prediction video_folder ../data/single_tracking/OTB \
--evaluation_type single \
--writeout Ture

'''

import os
import cv2
import argparse
from glob import glob
from sklearn.metrics import jaccard_score
from collections import defaultdict, OrderedDict
import xml.etree.ElementTree as ET

'''
helper function for getting the frames. Video name is either the name of the video or the name of the folder that contains
a folder that has the images
'''
def get_frames(video_name):
    #slice videos into frames
    if video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        #reading images
        images = glob(os.path.join(video_name, '*/*/*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

# compute IoU between prediction and ground truth, bounding box input is x1,y1,x2,y2
def compute_iou(prediction, gt):
    #ensure the bounding boxes exist
    assert(prediction[0] <= prediction[2])
    assert(prediction[1] <= prediction[3])
    assert(gt[0] <= gt[2])
    assert(gt[1] <= gt[3])

    #intersection rectangule
    xA = max(prediction[0], gt[0])
    yA = max(prediction[1], gt[1])
    xB = min(prediction[2], gt[2])
    yB = min(prediction[3], gt[3])

    #compute area of intersection
    interArea = max(0, xB-xA + 1) * max(0, yB - yA + 1)

    #compute the area of the prection and gt
    predictionArea = (prediction[2] - prediction[0] +1) * (prediction[3] - prediction[1] +1)
    gtArea = (gt[2] - gt[0] + 1) * (gt[3]-gt[1]+1)

    #computer intersection over union
    iou = interArea / float(predictionArea+gtArea-interArea)
    return iou

#mask iou is computed via sklearn
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score

#Process single object ground truth
#ground truth format <x>, <y>, <w>, <h>
#return a ditionary with frame as key and a list of [<x>, <y>, <w>, <h>] presented in the frame
def process_single(txt_file):
    gt = defaultdict(list)
    a = 1
    with open(txt_file) as file:
        for line in file:
            try:
                gt[a].append(list(eval(line)))
            except:
                line = [int(item) for item in line.split()]
                gt[a].append(line)
            a+=1
    return gt


#Process multiple object ground truth
#ground truth format is <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <conf>, <class>, <visibility>
#return a ditionary with frame as key and a list of [<x>, <y>, <w>, <h>, <conf>, <class>, <visibility>] presented in the frame
def process_multiple(txt_file):
    gt = defaultdict(list)
    with open(txt_file) as file:
        #prediction format is <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        for line in file:
            line = list(eval(line))
            gt[line[0]].append(line[1:])
    return gt


#process C2C12 cell tracking ground truth that uses xml annotation
#return a dictionary with frame as key and a list of [<id>, <x>, <y>] presented in the frame
def porcess_cellgt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    if root.tag:
        print('XML successfully loaded')
        print('Root is ', root.tag, 'Attribute is', root.attrib)

    # extract xml file by id <id> <frame> <x> <y>
    dict_by_id = defaultdict(list)

    for ele in root.iter():
        #print(ele.tag, ele.attrib)
        if ele.tag == 'fs' or ele.tag =='as':
            pass
        if ele.tag == 'a':
            #print(ele.attrib)
            current_id = ele.attrib['id']
        if ele.tag == 's':
            dict_by_id[current_id].append([ele.attrib['i'], ele.attrib['x'], ele.attrib['y']])

    # process the file to be a gt dictionary with key being frame number
    gt = defaultdict(list)
    for key in dict_by_id.keys():
        for item in dict_by_id.get(key):
            gt[int(item[0])+1].append([int(key), int(float(item[1])), int(float(item[2]))])
    #print(gt.keys())
    return gt

def single_eval(prediction, gt, threshold, single=True):
    #producing evaluation results for single video
    if single:
        pass
    #producing evaluation results for multiple videos
    else:
        pass

def multi_eval(predition, gt, threshold,single=True):
    pass

def cell_eval(prediction, gt, threshold, single=True):
    pass    

def visualization(prediction, gt, threshold):
    pass

def main():
    parser = argparse.ArgumentParser(description='tracker evaluation')
    parser.add_argument('--predcition_folder', type=str, help = 'path to the txt files of the predicted outputs')
    parser.add_argument('--video_folder', type=str, help = 'path to the folders that contains image sequence and gt')
    parser.add_argument('--evaluation_type', type=str, help= "flag for single mulitple or cell")


if __name__ == '__main__':
    main()
