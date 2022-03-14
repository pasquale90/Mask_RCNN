import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

#nadir="/data/CoRoSect/10.code/maskRCNN/Mask_RCNN_matterport/mask_rcnn/datasets/Nasekomo_insects"
#nadir="../../mask_rcnn/datasets/Nasekomo_insects"
#nadir="/home/melissap/Desktop/CoRoSect/10.code/maskRCNN/mask_rcnn/datasets/Nasekomo_insects"
nadir="../../datasets/Nasekomo_insects"

annotation_file=os.path.join(nadir,"phase1_Nasekomo_3609.txt")
annotation_folder=annotation_file.split(".txt")[0]

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


''' replace config and utils with new files'''
'''
from mrcnn.config import Config
from mrcnn import model as modellib, utils
'''
sys.path.append(os.path.join(ROOT_DIR,"mrcnn")) 
print(sys.path)
from mrcnn.insects_config import Config
from mrcnn import insects_utils as utils

from mrcnn import model as modellib




# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
############################################################
#  Configurations
############################################################
class InsectsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "insects"
        
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1#4 # background + 4 lifecycle stages - eggs,larvae,pupae,adults

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = InsectsConfig()
config.display()



def create_path(path):
    try:
        os.makedirs(path)
        print("folder '{}' created ".format(path))
    except FileExistsError:
        print("folder {} already exists".format(path))
        
def define_class(sf):
    sf=int(sf)
    if (sf<=4):
        insect_class='eggs'
        class_id=1
    elif (sf>4 and sf<=19):
        insect_class='larvae'
        class_id=2
    elif (sf>19 and sf<=37):
        insect_class='pupae'
        class_id=3
    elif (sf>=40):
        insect_class='adult'
        class_id=4
    return "insects",1#insect_class,class_id

def get_image_dim(impath):
    im=cv2.imread(impath)
    w,h=im.shape[1],im.shape[0]
    return w,h
    
def parse_IT_annotation_file(filename):
    #flines=[]
    with open(filename) as file:
        lines = file.readlines()
    
    for line in lines:
        #print(line)
        xs,ys=[],[]
        new_line=''
        fname=line.split("|")[0].split(".jpg")[0]+".txt"
        annotation_subfile=os.path.join(annotation_folder,fname)
        #print(annotation_subfile)
        xys=line.split("|")[3]
        for c,s in enumerate(xys.split(",")[0:-1]):
            annot=s.split(":")[1]
            new_line+=annot+","
            if(c%2):
                ys.append(annot)
                #print("y",c,"=",annot)
            elif (c%2==0):
                xs.append(annot)
                #print("x",c,"=",annot)
            #print(s.split(":")[1])
        #print(f' new line = {new_line}' )
        #save_line_to_file(annotation_subfile,new_line)
        with open(annotation_subfile, mode='a', encoding='utf-8') as f:
            #for lines in data:
            f.write('\n')
            f.write(new_line)

def im_is_in_split(split,img):
    #images are named as : 1_010001 up to 1_360001 - 1_[01]0001 the first 30 will be used for training
    split_id=int(img.split("_")[1][:2])

    #print(f'split_id {split_id}')
    if split_id<30:
        if split=="train":
            return True
        elif split=="valid":
            return False
    elif split_id>=30:
        if split=="train":
            return False
        elif split=="valid":
            return True
        
        
        
def parse_polygons_from_annotation_file(annotation_filepath):
    with open(annotation_filepath) as file:
        lines = file.readlines()
    
    polygons=[]

    for line in lines:
        xs,ys=[],[]        
        
        if (not line.isspace()):
            #try:
            line=line.split(",")[:8]
            if len(line)==8:
                for c,xys in enumerate(line):
                    #print(f'xys[{c}] = {xys}')
                    if(c%2):
                        ys.append(int(xys)-1)
                        #print("y",c,"=",int(xys))
                    elif (c%2==0):
                        xs.append(int(xys)-1)
                        #print("x",c,"=",int(xys))
            #except ValueError:
            #    continue
            dict_annotation={"name":"polygon","all_points_x":xs,"all_points_y":ys}
            polygons.append(dict_annotation)
    return polygons


#at First, parse the ImageTagger annotation file
if (os.path.isdir(annotation_folder)):
    print(f'annotation_folder {annotation_folder} exists')
else:
    create_path(annotation_folder)
    print(f'annotation_folder : {annotation_folder} does not exist. \nParsing of annotation file : {annotation_file}')
    parse_IT_annotation_file(annotation_file)


############################################################
#  Dataset
############################################################

class InsectsDataset(utils.Dataset):

    def load_subset(self, nadir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        """
        self.add_class("insects", 1, "eggs")
        self.add_class("insects", 2, "larvae")
        self.add_class("insects", 3, "pupae")
        self.add_class("insects", 4, "adults")
        """
        self.add_class("insects",1,"insects")
        
        # Add images
        for sf in os.listdir(nadir):
            currf=os.path.join(nadir,sf,'trainvalid')
            #print(sf)
            if (os.path.isdir(currf)):
                insect_class,class_id=define_class(sf)
        
                for im in os.listdir(currf):
                    if (im_is_in_split(subset,im)):
                        image_id= im 
                        #print(im)
                        image_path=os.path.join(currf,im)
                        #print(image_path)
            
                        width,height=get_image_dim(image_path)
            
            
                        #get mask_annotations
                        try:
                            annotation_filepath=os.path.join(annotation_folder,im.split(".jpg")[0]+'.txt')
                            #print(f'annotation_filepath = {annotation_filepath}')
                            polygons=parse_polygons_from_annotation_file(annotation_filepath)
                            num_ids=[class_id for i in range(len(polygons))]
                        except FileNotFoundError:
                            #remove file from annotation 
                            continue   
            
            

                        self.add_image(insect_class,
                            image_id=image_id,  # use file name as a unique image id
                            path=image_path,
                            width=width, height=height,
                            polygons=polygons)#,
                            #num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "insects":
            print(super(self.__class__, self).load_mask(image_id))
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        #return mask, info['class_ids'].astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "insects":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = InsectsDataset()
    dataset_train.load_subset(nadir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = InsectsDataset()
    dataset_val.load_subset(nadir, "valid")
    dataset_val.prepare()
    
    
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
    
    
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/insects/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = InsectsConfig()
    else:
        class InferenceConfig(InsectsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
