#Turning xml annotation to <frame> <id> <x> <y>
#output file <frame> <id> <x> <y> <w> <h>
#w h being how big the point is
import xml.etree.ElementTree as ET
from collections import defaultdict


def main():
    tree = ET.parse('../data/cell_tracking/C2C12/annotation/Human exp1_F0001 Data.xml')
    root = tree.getroot()

    if root.tag and root.attrib:
        print(',meow')
    print('Root is', root.tag, 'attrib',root.attrib)
    #first extract xml file by id <id> <frame> <x> <y>
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

    #<frame><id><x><y>
    frame_id =[]
    for key in dict_by_id.keys():
        for item in dict_by_id.get(key):
            #frame number starts at 1
            #print(item)
            frame_id.append([int(item[0])+1, key, int(float(item[1]))-2, int(float(item[2]))-2])
    print(frame_id)
    # process the file to be a gt dictionary with key being frame number
    gt = defaultdict(list)
    for key in dict_by_id.keys():
        for item in dict_by_id.get(key):
            gt[int(item[0])+1].append([int(key), int(float(item[1])), int(float(item[2]))])
    print(gt)

    #width and height of the dot
    w, h = 1, 1
    with open('gt.txt', 'w') as file:
        for item in frame_id:
            item ='{}, {}, {}, {}, {}, {}'.format(item[0], item[1], item[2], item[3], 5, 5)
            file.write('%s\n' % str(item))

if __name__ == '__main__':
    main()