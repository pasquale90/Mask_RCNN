import os
import cv2
import json

def read_json(filepath):
    # read file
    with open(filepath, 'r') as f:
        data=json.loads(f.read())
    return data



def parse_json(d, depth=0):
    unq_keys=[]
    depths=[]
    values=[]
    
    dict2list=[]
    for key, value in d.items():
        if isinstance(value, dict):
            parse_json(value, depth+1)
        else:
            if not(key in unq_keys):
                print(f'{key} doesn t exist in keylist {unq_keys}')
                unq_keys.append(key)
                depths.append(depth)
                values.append(value)
            #print('\t' * depth, f'{key}: {value}')
            #line='\t' * depth, f'{key}: {value}'
            #dict2list.append(line)
    return unq_keys,depths,values
            
prototype_path="/data/CoRoSect/10.code/maskRCNN/Mask_RCNN_matterport/mask_rcnn/datasets/balloon/balloon"
train_splitf=os.path.join(prototype_path,'train')
val_splitf=os.path.join(prototype_path,'val')

train_imgs=[]
for file in os.listdir(train_splitf):
    if file.split('.')[1]=='jpg':
        print(f'Appending image file {file}')
        train_imgs.append(file)
    else:
        print(f'Annotation file : {file}')
        annotationf=os.path.join(train_splitf,file)
        print(annotationf)

annots = read_json(annotationf)
#print(json.dumps(annots, indent=2, sort_keys=True))

#a=list_stuff(annots)

keys,depths,values=parse_json(annots)
print(keys)
    
    #break