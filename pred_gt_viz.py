import os
import cv2
import argparse
from glob import glob
from collections import defaultdict, OrderedDict

'''
python pred_gt_viz.py --video_name data/multi_tracking/MOT17/train/MOT17-02-FRCNN/img1 \
--prediction sort_result/MOT17-02-FRCNN.txt \
--gt data/multi_tracking/MOT17/train/MOT17-02-FRCNN/gt/gt.txt \
--fps 20

python pred_gt_viz.py --video_name data/multi_tracking/MOT17/train/MOT17-04-FRCNN/img1 \
--prediction sort_result/MOT17-04-FRCNN.txt \
--gt data/multi_tracking/MOT17/train/MOT17-04-FRCNN/gt/gt.txt \
--fps 20

'''

parser = argparse.ArgumentParser(description='mot demo')
parser.add_argument('--video_name', type=str, help='video to be processed')
parser.add_argument('--prediction', type=str, help ='generated prediction file')
parser.add_argument('--gt', type=str, help='gt text file')
parser.add_argument('--fps', type=int, help='frame rate for generated file')
args= parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        #print('reading image')
        #print(os.path.join(video_name, '*.jp*'))
        images = glob(os.path.join(video_name, '*.jp*'))
        #print(images)
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))[0:500]
        for img in images:
            frame = cv2.imread(img)
            yield frame

#get color for different id
#https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracking_utils/visualization.py
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


# compute IoU between prediction and ground truth
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

def main():
    images = glob(os.path.join(args.video_name, '*.jp*'))
    images = sorted([item.split('/')[-1] for item in images])
    #print(images)
    #print(len(images))
    threshold =0.4

    #reading prediction file
    prediction = defaultdict(list)
    with open(args.prediction) as file:
        for line in file:
            #prediction format is <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            prediction[images[eval(line)[0]-1]].append(list(eval(line)[1:]))
    #print('raw pred',prediction)

    #reading gt file
    gt = defaultdict(list)
    with open(args.gt) as file:
        #gt format is <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <conf>, <class>, <visibility>
        for line in file:
            gt[images[eval(line)[0]-1]].append(list(eval(line)[1:]))
            #print('raw gt',gt)

    img = cv2.imread(os.path.join(args.video_name, images[0]))
    #print(img.shape)
    height, width, _ = img.shape
    size = (width, height)
    file_name = '_'.join([args.video_name.split('/')[-2], 'video.mp4'])
    out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'MP4V'), args.fps, size)


    #outputing the video
    a = 0
    #keep track of the ground truth and track id pair
    gt_track_id = {}
    iou_array = []
    
    for img in images:
        a+=1
        gt_blist = []
        det_img = cv2.imread(os.path.join(args.video_name, img))
        gt_result = gt.get(img)
        prediction_result = prediction.get(img)
        #print('gt result', gt_result)
        #print('pred result', prediction_result)
        #keep track of iou in the current img iou
        img_iou = []

        #make sure gt_result and prediction exist
        if gt_result and prediction_result:
            
            #relate gt_id with track id
            gt_track = defaultdict(list)

            #copy image for plotting ground truth
            overlay = det_img.copy()

            for gt_box in gt_result:
                gt_label = gt_box[0]
                gt_box = [gt_box[1], gt_box[2], gt_box[1]+gt_box[3], gt_box[2]+gt_box[4]]
                for pred_box in prediction_result:
                    bbox_track = [int(pred_box[1]), int(pred_box[2]), int(pred_box[1]+pred_box[3]), int(pred_box[2]+pred_box[4])]
                    temp_iou = compute_iou(bbox_track, gt_box)
                    if temp_iou >threshold:
                        if gt_label in gt_track.keys():
                            if gt_track[gt_label][1] < temp_iou:
                                gt_track[gt_label] = [int(pred_box[0]), temp_iou, bbox_track]
                        else:
                            gt_track[gt_label] = [int(pred_box[0]), temp_iou, bbox_track]
                
                #check for identity switches
                if gt_track.get(gt_label) and gt_label not in gt_blist:
                    if gt_track_id.get(gt_label):
                        if int(gt_track_id.get(gt_label)) != int(gt_track[gt_label][0]):
                            #print('ID mismatch',int(gt_track_id.get(gt_label)), int(gt_track[gt_label][0]))
                            gt_blist.append(gt_label)
                    else:
                        #print('adding label to track list', gt_label)
                        gt_track_id[gt_label] = int(gt_track[gt_label][0])

                #plotting the gt result
                if gt_label in gt_blist:
                    #plot lost track with red
                    cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0,0,255), -1)
                else:
                    if gt_track.get(gt_label):
                        if gt_track[gt_label][1] > threshold:
                            #plot correctly tracked with green
                            cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0,255,0), -1)
                            print(gt_track[gt_label][1])
                            img_iou.append(gt_track[gt_label][1])
                        else:
                            #plot lower than threshold with blue
                            cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255,0,0), -1)
                    else:
                        #plot untracked with white
                        cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255,255,255), -1)
            alpha = 0.3
            det_img =cv2.addWeighted(overlay, alpha, det_img, 1 - alpha, 0)
            #print(gt_blist)

            #plotting prediction results
            for pred_box in prediction_result:
                bbox_track = [int(pred_box[1]), int(pred_box[2]), int(pred_box[1]+pred_box[3]), int(pred_box[2]+pred_box[4])]
                track_label = str(int(pred_box[0]))
                cv2.rectangle(det_img, (int(bbox_track[0]), int(bbox_track[1])), (int(bbox_track[2]), int(bbox_track[3])), get_color(int(pred_box[0])), 3)
                cv2.putText(det_img, track_label, (bbox_track[0]+5, bbox_track[1]+20), 0,0.6, (255,255,255),thickness=2)
                #cv2.rectangle(det_img, (int(bbox_track[0]), int(bbox_track[1])), (int(bbox_track[2]), int(bbox_track[3])), (0,255,0), 3)
                #cv2.putText(det_img, 'Person', (bbox_track[0]+5, bbox_track[1]+20), 0,0.6, (255,255,255),thickness=2)
            
            #computing tracked stats
            if len(img_iou)>0:
                img_iou_mean = sum(img_iou)/len(img_iou)
                iou_array.append(img_iou_mean)
                text = 'Frame {}: Mean IoU is {}%, {} lost'.format(a, round((img_iou_mean *100),2), len(gt_blist))
            else:
                text = 'Frame {}: Mean IoU is {}%, {} lost)'.format(a, 0, len(gt_blist))
            print(text)
            cv2.putText(det_img, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
        else:
            overlay = det_img.copy()
            for gt_box in gt_result:
                gt_box = [gt_box[1], gt_box[2], gt_box[1]+gt_box[3], gt_box[2]+gt_box[4]]
                cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255,255,255), -1)
            alpha = 0.3
            det_img =cv2.addWeighted(overlay, alpha, det_img, 1 - alpha, 0)
            text = 'Frame {}: Mean IoU is {}%, no track)'.format(a, 0)
            print(text)
            cv2.putText(det_img, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
        out.write(det_img)
    print('Overall mean IoU is {} %'.format(round((sum(iou_array)/len(iou_array))*100,2)))
    out.release()

if __name__ == '__main__':
	main()