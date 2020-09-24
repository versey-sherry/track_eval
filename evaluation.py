#Command line tool for evaluating tracking
'''
Input
--prediction_dir path to the txt files of predicted outputs
--video_dir path to the training sequence with gt
--results_dir folder to save the results
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

All video dirs should be in the same dir, to process LaSOT
find . -maxdepth 2  -print -exec mv {} . \;
'''

'''
example
python evaluation.py --prediction_dir result_example/sot_results/TB-100/goturn \
--video_dir gt_example/TB-100 \
--gt_name groundtruth_rect.txt \
--evaluation_type single \
--threshold 0.5 \
--writeout Ture

python evaluation.py --prediction_dir result_example/mot_results/MOT20/FairMOT \
--video_dir gt_example/MOT20 \
--gt_name gt/gt.txt \
--evaluation_type multiple \
--threshold 0.5 \
--writeout Ture



'''

import os
import cv2
import argparse
from glob import glob
from sklearn.metrics import jaccard_score
from collections import defaultdict, OrderedDict
import xml.etree.ElementTree as ET
import motmetrics as mm
import numpy as np

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

#Process single object ground truth and predition
#ground truth format <x>, <y>, <w>, <h>
#predition format <x>, <y>, <w>, <h>
#return a ditionary with frame as key and a list of [<x>, <y>, <w>, <h>] presented in the frame
def process_single(txt_file):
    info = defaultdict(list)
    a = 1
    with open(txt_file) as file:
        for line in file:
            try:
                info[a].append(list(eval(line)))
            except:
                line = [int(item) for item in line.split()]
                info[a].append(line)
            a+=1
    return info


#Process multiple object ground truth and prediction and cell tracking prediction
#ground truth format is <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <conf>, <class>, <visibility>
#prediction format <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
# cell tracking prediction format <frame> <id> <x> <y>
# return a ditionary with frame as key and a list of [<id>, <x>, <y>, <w>, <h>, <conf>, ...] for multiple gt and pd
# return a ditionary with frame as key and a list of [<id>, <x>, <y>] for cell prediction
def process_multiple(txt_file):
    info = defaultdict(list)
    with open(txt_file) as file:
        #prediction format is 
        for line in file:
            line = list(eval(line))
            info[line[0]].append(line[1:])
    return info

#process C2C12 cell tracking ground truth that uses xml annotation
#return a dictionary with frame as key and a list of [<id>, <x>, <y>] presented in the frame
def process_cellgt(xml_file):
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

# Evaluation for one video
# output video name, accuracy and robustness
'''
video_dir = 'gt_example/TB-100'
prediction_dir = 'result_example/sot_results/TB-100/goturn'
save_dir = './results'
prediction = os.path.join(prediction_dir, 'Biker.txt')
gt = os.path.join(video_dir, 'Biker', 'groundtruth_rect.txt')
'''
def single_eval(prediction, gt, save_dir, threshold, writeout=False):
    video_name = prediction.split('/')[-1].split('.')[0]
    prediction_dict = process_single(prediction)
    gt_dict = process_single(gt)

    if os.path.exists(save_dir):
        print('Save directory already exists')
    else:
        os.makedirs(save_dir)
        print('Making new save dir')

    #preparation for writing out the video
    if writeout:
        img_dir = os.path.join('/'.join(gt.split('/')[0:-1]), 'img')
        #print(img_dir)
        video_name = prediction.split('/')[-1].split('.')[0]
        images = sorted([file for file in os.listdir(img_dir) if '.jpg' in file or '.jpeg' in file])
        img = cv2.imread(os.path.join(img_dir, images[0]))
        height, width, _ = img.shape
        size = (width, height)
        file_name = file_name = '_'.join([prediction.split('/')[-2], video_name, 'video.mp4'])
        file_name = os.path.join(save_dir, file_name)
        print('Video saved to', file_name)
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MP4V'), 20, size)

    acc_list = []
    robustness = 0
    frame_num = 1

    #loop through every frame in ground truth
    while frame_num <= len(gt_dict):
        prediction = prediction_dict.get(frame_num)[0]
        gt = gt_dict.get(frame_num)[0]

        prediction = [prediction[0], prediction[1], prediction[0]+prediction[2], prediction[1]+prediction[3]]
        gt = [gt[0], gt[1], gt[0] + gt[2], gt[1]+gt[3]]

        acc = compute_iou(prediction, gt)
        acc_list.append(acc)

        if acc > threshold:
            robustness +=1
        #print(prediction)
        #print(gt)
        
        #output videos
        if writeout:
            img = cv2.imread(os.path.join(img_dir, images[frame_num-1]))
            #print(images[frame_num-1])
            #ploting prediction with red box
            img = cv2.rectangle(img, (prediction[0], prediction[1]), (prediction[2], prediction[3]), (0, 0, 255), 2)
            #ploting ground truth with green box
            img = cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0,255,0), 2)
            #Put stats on frame
            text = 'Frame {}: IoU is {}%'.format(frame_num,round((acc *100),2))
            img = cv2.putText(img, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA) 
            out.write(img)
        
        frame_num+=1
    
    if writeout:
        out.release()

    if len(acc_list) >0:
        accuracy = sum(acc_list)/len(acc_list)
    else:
        print('No IoU')

    if robustness >0:
        robustness = robustness/len(gt_dict)
        
    return video_name, accuracy, robustness


def multiple_eval(prediction, gt, save_dir, threshold, writeout=False):
    video_name = prediction.split('/')[-1].split('.')[0]
    prediction_dict = process_multiple(prediction)
    gt_dict= process_multiple(gt)
    #print(prediction_dict.keys())
    #print('prediction is', prediction_dict.get(1))
    #print('--------------------------------------------------------------------------')
    #print(gt_dict.keys())
    #print('ground truth is', gt_dict.get(1))

    if os.path.exists(save_dir):
        print('Save directory already exists')
    else:
        os.makedirs(save_dir)
        print('Making new save dir')

    #preparation for writing out the video
    if writeout:
        img_dir = os.path.join('/'.join(gt.split('/')[0:-2]), 'img1')
        print(img_dir)
        video_name = prediction.split('/')[-1].split('.')[0]
        images = sorted([file for file in os.listdir(img_dir) if '.jpg' in file or '.jpeg' in file])
        img = cv2.imread(os.path.join(img_dir, images[0]))
        height, width, _ = img.shape
        size = (width, height)
        file_name = file_name = '_'.join([prediction.split('/')[-2], video_name, 'video.mp4'])
        file_name = os.path.join(save_dir, file_name)
        print('Video saved to', file_name)
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MP4V'), 20, size)

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    #predcition needs a few frame to start
    for key in sorted(prediction_dict.keys()):
        predcition_in_frame = prediction_dict.get(key)
        gt_in_frame = gt_dict.get(key)

        #process the file for motmetrics
        prediction_object = []   
        prediction_bbox_list = []

        gt_object = []
        gt_bbox_list = []

        for item in predcition_in_frame:
            if item[5] > threshold:
                prediction_object.append(item[0])
                prediction_bbox_list.append([item[1], item[2], item[3], item[4]])
        prediction_bbox_list = np.array(prediction_bbox_list)
        #print('prediction_object', len(prediction_object))
        #print('prediction box', len(prediction_bbox_list))

        for item in gt_in_frame:
            if item[5]>0:
                gt_object.append(item[0])
                gt_bbox_list.append([item[1], item[2], item[3], item[4]])
        gt_bbox_list = np.array(gt_bbox_list)
        #print('gt_object', len(gt_object))
        #print('gt box', len(gt_bbox_list))

        #compute distance using solver
        # 0 when the rectangles overlap perfectly and 1 when the overlap is zero, therefore use 1-threshold instead of threshold
        # nan indicate the ground truth and hypothesis can't be paried
        gt_pred_distance = mm.distances.iou_matrix(gt_bbox_list, prediction_bbox_list, max_iou = 1-threshold)
        #print('gt len', len(gt_bbox_list))
        #print('pred len', len(prediction_bbox_list))
        #print('distance matrix', np.array(gt_pred_distance).shape) #dimension correct
        
        #update the event accumulator
        #acc.update update the pairwise relation solver and returns the frame id
        frameid = acc.update(gt_object, prediction_object, gt_pred_distance)
        #print(acc.mot_events.loc[frameid])


    # pandas dataframe has events for all the frames
    #print(acc.mot_events)

    #Accumulator has been populated, compute metrics
    mh = mm.metrics.create()
    # Change this line to include useful metrics
    # more detail regarding the metrics is here https://github.com/cheind/py-motmetrics
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'precision', 'recall',
        'num_switches', 'num_false_positives', 'num_misses', 'mostly_tracked', 'mostly_lost'], name=video_name)
    #summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    report = ''
    for col in summary.columns:
        text = ' '.join([str(col), 'is', str(round(summary.iloc[0][col],4))])+'\n'
        report = ''.join([report, text])

    mota = summary.iloc[0]['mota']
    motp = summary.iloc[0]['motp']
    precision = summary.iloc[0]['precision']
    recall = summary.iloc[0]['recall']
    return video_name, report, mota, motp, precision, recall

def cell_eval(prediction, gt, threshold, single=True):
    pass    

def visualization(prediction, gt, threshold):
    pass

def main():
    parser = argparse.ArgumentParser(description='tracker evaluation')
    parser.add_argument('--prediction_dir', type=str, help = 'path to the txt files of the predicted outputs')
    parser.add_argument('--video_dir', type=str, help = 'path to the directory that contains image sequence and gt')
    parser.add_argument('--gt_name', type=str, help= "naming rules for ground truth")
    parser.add_argument('--save_dir', type=str, default= './results')
    parser.add_argument('--evaluation_type', type=str, help= "flag for single mulitple or cell")
    parser.add_argument('--threshold', type=float, default = 0.5, help= "threshold for correctly tracked")
    parser.add_argument('--writeout', type=bool, default = False, help= "True if there is video output needed")
    args= parser.parse_args()

    #list all the videos to be evaluated
    video_list = [folder for folder in os.listdir(args.video_dir) if '.' not in folder]

    print('The video(s) to be evaluated', video_list)
    print('Evaluation type is', args.evaluation_type)
    # A list to hold the summary for reporting
    output_list = []

    if args.evaluation_type == 'single':
        accuracy_list = []
        robustness_list = []
    elif args.evaluation_type == 'multiple':
        mota_list = []
        motp_list = []
        precision_list = []
        recall_list = []
    else:
        pass

    for folder in video_list:
        prediction = os.path.join(args.prediction_dir, folder+'.txt')
        print('Prediction file is', prediction)
        gt = os.path.join(args.video_dir, folder, args.gt_name)
        print('Ground truth is', gt)

        if args.evaluation_type == 'single':
            name, accuracy, robustness = single_eval(prediction, gt, args.save_dir, args.threshold, writeout = args.writeout)
            
            accuracy_list.append(accuracy)
            robustness_list.append(robustness)
            text = '{}\nAccuracy is {}%, robustness is {}%'.format(name, round(accuracy*100, 2), round (robustness*100, 2))
            print(text)
            output_list.append(text)
        elif args.evaluation_type == 'multiple':
            name, report, mota, motp, precision, recall = multiple_eval(prediction, gt, args.save_dir, args.threshold, writeout = args.writeout)

            mota_list.append(mota)
            motp_list.append(motp)
            precision_list.append(precision)
            recall_list.append(recall)
            text = '{}\n{}'.format(name, report)
            print(text)
            output_list.append(text)
        else:
            pass

    if args.evaluation_type == 'single':
        if len(accuracy_list) >0:
            total_accuracy = sum(accuracy_list)/len(accuracy_list)
        else:
            print('No accuracy')
        
        if len(robustness_list) >0:
            total_robustness = sum(robustness_list)/len(robustness_list)
        else:
            print('No robustness')

        text = '{}: Accuracy is {}% Robustness is {}%'.format(
            args.video_dir.split('/')[-1], 
            round(total_accuracy*100, 2), 
            round(total_robustness *100, 2))
        #print(text)
        output_list.append(text)
    elif args.evaluation_type == 'multiple':
        if len(mota_list) >0:
            total_mota = sum(mota_list)/len(mota_list)
        else:
            print('No MOTA')

        if len(motp_list) >0:
            total_motp = sum(motp_list)/len(motp_list)
        else:
            print('No MOTP')

        if len(precision_list) >0:
            total_precision = sum(precision_list)/len(precision_list)
        else:
            print('No Precision')

        if len(recall_list) >0:
            total_recall = sum(recall_list)/len(recall_list)
        else:
            print('No Recall')

        text = '{}: M0TA is {}% MOTP is {}% Precision is {}% Recall is {}%'.format(
            args.video_dir.split('/')[-1], 
            round(total_mota*100, 2), 
            round(total_motp *100, 2),
            round(total_precision *100, 2),
            round(total_recall *100, 2))
        print(text)
        output_list.append(text)
    else:
        pass

    #writing out the results
    with open(os.path.join(args.save_dir, args.prediction_dir.split('/')[-1]+'_results.txt'), 'w') as file:
        for item in output_list:
            file.write(''.join([item, '\n']))

if __name__ == '__main__':
    main()
