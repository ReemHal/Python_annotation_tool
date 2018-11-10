
''' This is a simple annotation tool that allows users to annotate images by labeling superpixels
    created in the image.
    Created by: Reem Al-Halimi
    On: March/9/2017
'''

# import the necessary packages
from skimage.segmentation import slic
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from pylab import *
import pickle
from scipy.sparse import csr_matrix
from matplotlib.widgets import Slider, Button, RadioButtons
import time

# Create an interactive plotting session
plt.ion()

# initialize environment settings
val_inc = 10
axis_color='lightgoldenrodyellow'
initNumSegments = 100
sigmaVal = 5
SLICO = False # SLIC-zero is the version of SLIC that automatically calculates the best parameters. Only user input is numSegments
annot_dir=""
default_annot_dir = "/home/robot/scripts/annot_images/"
annotation_extension = "_annot"
move_to_next_object= u' ' # space bar
move_to_next_image= u'right'
move_to_prev_image= u'left'
end_labelling_session_without_saving= u'ctrl+q'
save_and_quit= u'escape'  # escape key
quit_request= False
quit_and_save_request= False
IMAGE_FILE_ERROR_1 = "Image file Path is Invalid (read_image module):"
NO_IMAGES_IN_FILE = "Image directory has to contain images"
error_msg_dict={"IMAGE_FILE_ERROR_1":IMAGE_FILE_ERROR_1,
                "NO_IMAGES_IN_FILE": NO_IMAGES_IN_FILE}

# Get list of labels
tomato = 'tomato'
stem= 'stem'
leaf = 'leaf'
label_list = [tomato, stem, leaf]

#initialize global variables that are referenced in event functions
image_axes=0
segment_dict= {}
annot_mask =[]
curr_image_file= []
clicked_region_indeces= []
segments= []
image= []
segmented_image=[]
next_item=False
init_item_number = 1
item_number= init_item_number
annotation_dict= {}
debugging= False
debugging2 = False
kMeanSuperpixels = 'kMeanSuperpixels'
numSegments= initNumSegments
read_image_gen=None
reset_dict= True  # used to prevent the slider from resetting the annotation dictionary and annotation mask when slider is not changed manually
compare_annot_mask= [] # remove later. Not needed@

# @profile
def sliders_on_changed(val):
    global image, numSegments, image_axes, annot_mask, item_number, segmented_image
    global reset_dict, segments, currentLabel, curr_image_file

    if debugging2: print "into sliders on changed. reset_dict= ", reset_dict
    if reset_dict:
        numSegments= val
        annot_mask= np.zeros(image.shape[:2], dtype="uint8")
        item_number= init_item_number
        reset_annotation_dict(annotation_dict, curr_image_file, currentLabel)
        # Superimpose the superpixel boundaries on the  image array
        (segmented_image, segments)= get_segmented_image(image, val)
        image_axes.set_title(get_image_title(curr_image_file, currentLabel, numSegments))
    reset_dict= True

# Event handler for mouse click events
def onclick(event):

    global annot_mask, clicked_region_indeces, segments
    global segmented_image, item_number, debugging, image_axes

    if debugging:
       print "Type:", type(event.x), type(event.y), type(event.xdata), type(event.ydata)
       print 'button=', event.button, \
            'x=',  event.x, 'y=',  event.y, 'xdata=', event.xdata, 'ydata=', event.ydata, 'item_number=', item_number
    # make sure the mouse button was clicked within the image boundaries (e.g. not on the slider or radio buttons)
    if (event.inaxes == image_axes):
        if (type(event.xdata) == float64 and \
            type(event.ydata) == float64):
            if (event.button == 1):
                if debugging: print 'annotating item_number', item_number
                annot_mask = colorSegment(segmented_image, segments, annot_mask, item_number, clicked_region_indeces, event.ydata, event.xdata)
            elif (event.button==3):
                annot_mask = undo_colorSegment(segmented_image, segments, annot_mask, clicked_region_indeces, event.ydata, event.xdata)

# Event handler for keyboard presses
def on_key(event):
    global debugging, annot_dir, item_number, segment_dict
    global quit_request, quit_and_save_request, move_to_next_object, end_labelling_session_without_saving, save_and_quit
    global annot_mask, segments, image, curr_image_file, currentLabel, annotation_dict,\
           segmented_image,  segType, numSegments, read_image_gen, image_axes, annot_dir, label_radios

    if debugging:
        print 'you pressed', event.key, event.xdata, event.ydata
        print 'item_number', item_number
    if (event.key == move_to_next_object):
        item_number += 1
        if debugging: print 'updated to item_number', item_number
    elif (event.key == move_to_next_image) or (event.key == move_to_prev_image):
        if (event.key == move_to_prev_image):
            new_inc = -1
            if debugging: print "Trying to look for  prev image"
        else:
            new_inc = 1
        more_images= True
        # make sure there are more images to display
        try:
            return_res = get_next_image(read_image_gen,  annot_dir, new_inc= new_inc)
        except StopIteration:
            more_images = False
            if debugging: print "No more images to navigate to"
            pass
        if more_images:
            # Update the annotation dictionary to include changes done to the annotation mask
            annotation_dict = update_annot_dict(curr_image_file, currentLabel= currentLabel, annot_mask= annot_mask, \
                segType= segType, segParameters={'numSegments' : numSegments})
            # Save current image
            save_image(annot_dir, curr_image_file, annotation_dict)
            reset_label_radios(label_radios)
            segment_dict = {}
            item_number = 1
            (curr_image_file, image, currentLabel, segType, numSegments, annot_mask, annotation_dict, segParameters,\
                   init_annot_mask) = return_res
            if debugging: print "displaying next image ", curr_image_file, "..."
            #Display the image with the segment boundaries and annotated segments highlighted
            (segmented_image, segments, item_number)= display_image(image, image_axes, numSegments, annot_mask,\
                   curr_image_file, currentLabel)
            if debugging: print "Image ", curr_image_file, "DISPLAYED"
    elif (event.key == save_and_quit):
        quit_and_save_request= True
    elif (event.key == end_labelling_session_without_saving):
        quit_request= True

# Event handler for the Save button
def save_button_on_clicked(mouse_event):
    global curr_image_file, currentLabel, annot_mask, segType, numSegments, annot_dir, annotation_dict
    # this dictionary update should be done earlier when user move on to another category or image @
    annotation_dict = update_annot_dict(curr_image_file, currentLabel= currentLabel, annot_mask= annot_mask, \
        segType= segType, segParameters={'numSegments' : numSegments})
    save_image(annot_dir, curr_image_file, annotation_dict)

# Event handler for label radio buttons
# @profile
def label_radios_on_clicked(label):
    global annotation_dict, segParameters, segmented_image, segments, image_axes, image
    global curr_image_file, currentLabel, annot_mask, segType, numSegments, init_annot_mask, item_number
    global compare_annot_mask # remove later @

    # Update the annotation dictionary to include changes done to the annotation mask
    annotation_dict = update_annot_dict(curr_image_file, currentLabel= currentLabel, annot_mask= annot_mask, \
        segType= segType, segParameters={'numSegments' : numSegments})
    if currentLabel == tomato: compare_annot_mask= annot_mask
    # switch to new label
    currentLabel = label
    item_number = 1
    segType= kMeanSuperpixels
    annot_mask= np.zeros(image.shape[:2], dtype="uint8")
    init_annot_mask = annot_mask
    # Get the annotation mask of the new label from the annotation dictionary
    if (annotation_dict[curr_image_file].has_key(currentLabel)):
        # Retrieve the annotation mask and relevant information for the current label and image
        (annot_mask, segParameters, segType) = get_annot_info(curr_image_file, annotation_dict, currentLabel)
        init_annot_mask = annot_mask
        if (segParameters.has_key('numSegments')):
            numSegments= segParameters['numSegments']
    (segmented_image, segments, item_number) = display_image(image, image_axes, numSegments, annot_mask, curr_image_file, currentLabel)

def reset_label_radios(label_radios):
    label_radios.set_active(0)

# Event handler for the Remove All button
def removeAll_button_on_clicked(mouse_event):
    global image, curr_image_file, currentLabel, annotation_dict, annot_mask, item_number, init_item_number
    if debugging2: print "into removeAll_button_on_clicked"
    numSegs_slider.reset()
    annot_mask= np.zeros(image.shape[:2], dtype="uint8")
    item_number= init_item_number
    reset_annotation_dict(annotation_dict, curr_image_file, currentLabel)
    # Superimpose the superpixel boundaries on the  image array
    (segmented_image, segments)= get_segmented_image(image, numSegments)
    image_axes.set_title(get_image_title(curr_image_file, currentLabel, numSegments))

# Given a pixel location (x, y), return the indeces of all pixels within the same superpixel as (x, y)
# @profile
def getSegment(segments, x, y):
    # Get number of segment that was clicked
    clicked_segVal= segments[int(round(x)), int(round(y))]
    # get the indeces of all pixels that have the same segment (i.e. superpixel) number as (x, y)
    clicked_region_indeces= segments == clicked_segVal

    return clicked_region_indeces

# Highlight the segment that was clicked
# @profile
def colorSegment(image, segments, annot_mask, item_number=0, clicked_region_indeces=[], x=-1, y=-1):
    # Get indeces ot the clicked superpixel
    if debugging: "item number:", item_number
    # Are we higlighting a region where the user clicked on pixel (x, y)?
    if (x>0) and (y>0):
        clicked_region_indeces= getSegment(segments, x, y)
        annot_mask[clicked_region_indeces]= item_number

    res_image= cv2.bitwise_and(image, image, mask = flip_image(annot_mask))
    image_axes.cla()
    image_axes.imshow(res_image, cmap='jet', alpha=1)

    return(annot_mask)

# Undo segment highlight on the current segment
# @profile
def undo_colorSegment(image, segments, annot_mask, clicked_region_indeces, x, y):
        # Get indeces ot the clicked superpixel
        clicked_region_indeces= getSegment(segments, x, y)
        annot_mask[clicked_region_indeces]= 0
        res_image= cv2.bitwise_and(image, image, mask = flip_image(annot_mask))
        image_axes.cla()
        image_axes.imshow(res_image, cmap='jet', alpha=1)

        return(annot_mask)

def flip_image(image):
    return (image == 0).astype('uint8')

# Save the annotaiton data for current image in image_file_name
# @profile
def save_image(annot_dir, image_file_name, annotation_dict):

    annot_file = ""
    if not(annot_dir):
        annot_dir= default_annot_dir

    # get proper annotation file name for the current image
    annot_file = get_annot_file_name(annot_dir, image_file_name, create_dir= True)

    if debugging:
        print "annotation dir:", annot_dir
        print "annotation file", annot_file

    # save the annotations dictionary
    with open(annot_file, 'wb') as f:
        pickle.dump({image_file_name: annotation_dict[image_file_name]}, f, pickle.HIGHEST_PROTOCOL)

def get_arguments(ap):
    # construct the argument parser and parse the arguments
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    ap.add_argument("-a", "--annot", required = False, help = "Path to the annotations directory")
    args = vars(ap.parse_args())
    image_files= args["image"]
    if (args["annot"]):
        annot_dir = args["annot"]
    else:
        annot_dir= None
    return image_files, annot_dir

def error_msg(msg_key, msg_var, Error_type):
    print "ERROR: ", error_msg_dict[msg_key], msg_var
    raise Error_type

# @profile
# Given the absolute path and file name of an image file, read the file and return the image
def read_image(image_files):
    global image_names_list
    #check that the image path is valid
    if not os.path.exists(image_files):
        error_msg("IMAGE_FILE_ERROR_1", image_files, SystemExit)

    # check whether we have a directory of images or a single image file
    if (os.path.isdir(image_files)):
        initial_image_names_list = os.listdir(image_files)
        # Remove directory names from image names list
        image_names_list = [image_name for image_name in initial_image_names_list \
                            if not(os.path.isdir(image_files + '/'+ image_name))]
        num_images = len(image_names_list)
        curr_index= 0
        if num_images > 0:
            while True:
                if debugging: print "index:", curr_index, "current image in read_image:", image_files + '/'+ image_names_list[curr_index]
                # Make sure this is an actual file not a dir
                if not(os.path.isdir(image_files + '/'+ image_names_list[curr_index])):
                    image = cv2.imread(image_files + '/'+ image_names_list[curr_index])
                    curr_file = image_names_list[curr_index]
                    new_increment = yield (image_files, curr_file, image)
                    val = yield new_increment
                    if debugging: print "new increment:", new_increment
                    if new_increment is None:
                        curr_index +=1
                    else:
                        # using the modulo % operator allows us to circulate through the images.
                        # If we reach the last image, the "next" image becomes the first.
                        # similarly, going "back" from the first image takes us the last image.
                        curr_index = (curr_index + new_increment) % num_images
                        if curr_index < 0:
                            curr_index = num_images - curr_index
        else:
            error_msg("NO_IMAGES_IN_FILE", image_files, SystemExit)
    else:
        # split the path from the file name for consistency with the case where image_files contains a dir name
        image_path, curr_image_file = os.path.split(image_files)
        image = cv2.imread(image_files)

        yield (image_path, curr_image_file, image)

# Given the absolute path and file names of annotations file,
# get the annotation dictionary containing information about each label and the segments
# in the image that were labelled with that label
# @profile
def load_annotation_dict(annot_file):
    global debugging
    annotation_dict={}

    # load annotations file if Given
    if (annot_file) and (os.path.isfile(annot_file)):
        if debugging: print "annotations file to load dictionary from:", annot_file
        with open(annot_file, 'rb') as f:
            annotation_dict = pickle.load(f)

    return annotation_dict

# generate the name of the file where the annotations dictionary will be saved for an image given its
# full path name in image_file_name and the directory where all the annotations are to be saved
# if the user supplied annot_dir is a file name, then that is returned instead.
# @profile
def get_annot_file_name(annot_dir, image_file_name, create_dir=False):

    # check if the annot_dir is a directory
    if (annot_dir) and (os.path.isdir(annot_dir)):
        # Find the directory name for the current image
        image_path, image_name = os.path.split(image_file_name)
        annot_dir_for_image= annot_dir.rstrip("/") + "/" + image_path.strip("/").replace("/","_")

        # create the directory if it does not exists and create_dir flag is true
        if not os.path.exists(annot_dir_for_image):
            if create_dir:
                # Create the directory
                os.makedirs(annot_dir_for_image)
                annot_file = annot_dir_for_image+ "/" + image_name.replace(".","_") + annotation_extension
            else:
                # Throw and exception
                print "image file", image_file_name, " has no annotation file "
                annot_file = None
        else:
            annot_file = annot_dir_for_image+ "/" + image_name.replace(".","_") + annotation_extension
            if debugging: print "annotations file:", annot_file
    else:
            annot_file = annot_dir

    return annot_file

# given an image return the superpixels mask
# @profile
def get_superpixels(image, numSegments, sigmaVal, SLICO):
    global segment_dict

    #create superpixels: apply SLIC and extract (approximately) the supplied number of segments
    if debugging: print "numSegments=%d, sigma=%d", numSegments, sigmaVal
    if bool(segment_dict) and bool(segment_dict.has_key(numSegments)):
        segments= segment_dict[numSegments]
    else:
        start= time.time()
        segments = slic(img_as_float(image), n_segments= numSegments, sigma= sigmaVal, slic_zero= SLICO)
        end= time.time()
        segment_dict[numSegments] = segments
        if debugging: print "SLIC time=", end-start

    return  segments

# @profile
def create_segmented_image(image, segments):
    segmented_image = mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)

    return segmented_image

# @profile
def get_annotated_objects():
    annotated_image=[]
    if (annot_mask.size>0):
        annotated_image = create_segmented_image(image, annot_mask)
    return annotated_image

# Encoporate previously annotated areas into the displayed image
# @profile
def addPreviousAnnot(image, annot_mask, image_axes):
    res_image= cv2.bitwise_and(image, image, mask = flip_image(annot_mask))
    image_axes.cla()
    image_axes.imshow(res_image, cmap='jet', alpha=1)
    if debugging2: print "Prev annot shown"

# Generate image with superpixels
# @profile
def get_segmented_image(image, numSegments):

    if debugging:
        print "numSegments:", numSegments
    # get the superpixels mask with the number of segments as set for the current label
    segments = get_superpixels(image, numSegments, sigmaVal, SLICO)

    segmented_image = create_segmented_image(image, segments)
    image_axes.cla()
    image_axes.imshow(segmented_image)
    if debugging2: print "segmented img shown"
    return segmented_image, segments

# Initialize the annotation dictionary
# @profile
def initialize_dict(annotation_dict, curr_image_file, currentLabel, annot_mask=[], segParameters={}, segType=False):

    if (not(annotation_dict)):
        annotation_dict={}
    if (not(annotation_dict[curr_image_file])):
        annotation_dict[curr_image_file] = {}
    if (not(annotation_dict[curr_image_file].get(currentLabel))):
        annotation_dict[curr_image_file][currentLabel]= {}
        annotation_dict[curr_image_file][currentLabel]['segParameters']= segParameters
        annotation_dict[curr_image_file][currentLabel]['segType']= segType
        annotation_dict[curr_image_file][currentLabel]['annot_mask']= annot_mask

    return annotation_dict

# remove the annotation mask and attributes associated with the current label for the given image
# @profile
def reset_annotation_dict(annotation_dict, curr_image_file, currentLabel):
        if ((annotation_dict) and (annotation_dict[curr_image_file]) and  (annotation_dict[curr_image_file].get(currentLabel))):
            # Empty current Label mask and attributes
            del(annotation_dict[curr_image_file][currentLabel])

        return annotation_dict

# Given an annotation dictionary, an image file, and a label, get the relevant info fromt he dictionary
# Returns the annotation mask, the segmentation method used (segType), and the relevaant parameters
# for that method (segParameters)
# @profile
def get_annot_info(curr_image_file, annotation_dict, currentLabel):

    global debugging

    if (bool(annotation_dict[curr_image_file].get(currentLabel)) and \
       (bool(annotation_dict[curr_image_file][currentLabel] != {}))):
       if (bool(annotation_dict[curr_image_file][currentLabel].has_key('annot_mask'))):
           if debugging2: print "dictionary is not empty and there is an annotation mask"
           annot_mask = annotation_dict[curr_image_file][currentLabel]['annot_mask']
           # unpack from a compact representation into a numpy array
           annot_mask = annot_mask.toarray()
       if (bool(annotation_dict[curr_image_file][currentLabel].has_key('segParameters'))):
           if debugging: print "getting mask parameters"
           segParameters= annotation_dict[curr_image_file][currentLabel]['segParameters']
       if (bool(annotation_dict[curr_image_file][currentLabel].has_key('segType'))):
           if debugging: print "getting mask parameters"
           segType= annotation_dict[curr_image_file][currentLabel]['segType']

    return (annot_mask, segParameters, segType)

# Add the annotation mask for currentLabel to the dictionary
# @profile
def update_annot_dict(curr_image_file, currentLabel, annot_mask, segType, segParameters):
    # add the curent label and its annotation mask only if there are annotations in the mask
    if (np.all(annot_mask == 0)):
        if debugging2: print "popping ", currentLabel, " as it contains no annotations"
        annotation_dict[curr_image_file].pop(currentLabel, None)
    else:
        if debugging2: print "saving current annotation"
        annotation_dict[curr_image_file][currentLabel] = {}
        # add a compact form of the annotation mask
        annotation_dict[curr_image_file][currentLabel]['annot_mask']  = csr_matrix(annot_mask)
        annotation_dict[curr_image_file][currentLabel]['segParameters'] = segParameters
        annotation_dict[curr_image_file][currentLabel]['segType'] = segType

    return annotation_dict

def get_image_title(curr_image_file, currentLabel, numSegments):
    return ("%s\n %s -- %d segments" % (curr_image_file, currentLabel, numSegments))

class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """ "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)

# Given an instantiated read_image_gen generator which goes over all the files in
# the image directory, get_next_image gets the image's annotaton file if there is one,
#  creates the proper anootation mask, and initializes the number of superpixels to use
# (numSegments), the label to start with, and the segmentation type (currently only superpixels)
# input: instatiated read_image_gen, [optional] fig handle
# output: curr_image_file: the full path of the  image to be displayed next,
#         image: the image array to be displayed next,
#         currentLabel, segType, numSegments, annot_mask: the first label, segmentation type, and number of superpixels, respectively.
#         annotation_dict: the array containing annotations saved so far for all labels for the image.
#         segParameters: a dictionary of all parameters needed for the currentLabel and segmentation method
#                       (for superpixels it is only numSegment)
#         init_annot_mask: the initial value of the annotation mask. This is used when resetting the image for the current label.

# @profile
def get_next_image(read_image_gen, annot_dir, new_inc= None, fig=None):

    # load the image array  and convert it to a floating point data type
    return_res = [None]*9

    try:
        # New_inc reflects where the user wants to navigate to: +1 for next image forward, -1 means previous image
        if not(new_inc is None):
            read_image_gen.send(new_inc)
        next_image_res = next(read_image_gen)


    except StopIteration:
        next_image_res = return_res

        # if we navigated to a new image
    if (next_image_res != return_res):
        (curr_image_path, image_file_name, image) = next_image_res
        # Full path to the current image
        curr_image_file = curr_image_path + '/' + image_file_name
        if debugging: print "image_file in get_next_image:", curr_image_file
        # Full path to the annotations file
        annot_file = get_annot_file_name(annot_dir, curr_image_file, create_dir= False)

        ###############################################
        # Label superpixels for each label
        currentLabel= label_list[0]
        segType= kMeanSuperpixels
        numSegments= initNumSegments
        segParameters= {}

        #initialize annotation mask
        annot_mask= np.zeros(image.shape[:2], dtype="uint8")
        init_annot_mask = annot_mask

        # Load the annotation dictionary if an annotation file was given
        annotation_dict = load_annotation_dict(annot_file)
        if (not(annotation_dict.get(curr_image_file))):
            annotation_dict[curr_image_file]= {}
        if (not(annotation_dict[curr_image_file].get(currentLabel))):
            if debugging: print "label is not in dictionary...initializing it to empty"
            annotation_dict= initialize_dict(annotation_dict, curr_image_file, currentLabel, numSegments, segType)
        else:
            # Retrieve the annotation mask and relevant information for the current label and image
            (annot_mask, segParameters, segType) = get_annot_info(curr_image_file, annotation_dict, currentLabel)
            init_annot_mask = annot_mask
            if (segParameters.get('numSegments')):
                numSegments= segParameters['numSegments']

        if fig:
            # Show the new image
            # update the image label
            pass

        return_res = (curr_image_file, image, currentLabel, segType, numSegments, annot_mask, annotation_dict, segParameters,\
                init_annot_mask)

    return  return_res

# Display the image with the segment boundaries and annotated segments highlighted
# @profile
def display_image(image, image_axes, numSegments, annot_mask, curr_image_file, currentLabel):
    global reset_dict

    # Superimpose the superpixel boundaries on the  image array
    if debugging2: print "into display_image"
    (segmented_image, segments)= get_segmented_image(image, numSegments)

    # Take previously annotated superpixels into account when displaying the initial image
    if (annot_mask.size > 0):
        if debugging2: print "adding prev annotation"
        item_number= np.max(annot_mask) + 1
        addPreviousAnnot(segmented_image, annot_mask, image_axes)

    image_axes.set_title(get_image_title(curr_image_file, currentLabel, numSegments))

    # Warning: the sliders_on_changed handler rediplays the image
    # reset_dict is used to prevent the slider from resetting the dictionary and the annotation mask
    # set slider value to current numSegments
    reset_dict = False
    numSegs_slider.set_val(numSegments)
    return (segmented_image, segments, item_number)

#######################################################

if '__name__ == __main__':

    # Get image file and annotation file info from the argument list
    (image_files, annot_dir)= get_arguments(argparse.ArgumentParser())

     # process images
    read_image_gen = read_image(image_files)
    return_res = [None]*9

    try:
        return_res = get_next_image(read_image_gen, annot_dir)
    except StopIteration:
        print "Error: No images to display!"
        raise

    (curr_image_file, image, currentLabel, segType, numSegments, annot_mask, annotation_dict, segParameters,\
            init_annot_mask) = return_res
    # setup the figure area
    matplotlib.rcParams['toolbar']= 'None'
    fig = plt.figure()
    image_axes= fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    #create the slider. The slider allows the user to change the desired number of segments (i.e. superpixels)
    numSegs_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03], facecolor=axis_color)
    numSegs_slider_ax.set_xticklabels([], fontdict=None, minor=False)
    numSegs_slider    = DiscreteSlider(numSegs_slider_ax, 'Segs', 0, 1000, increment= val_inc, valfmt="%0.0f", valinit=numSegments,\
        visible= True, dragging= True, edgecolor='black', color='blue')
    numSegs_slider.on_changed(sliders_on_changed)

    # Add a button for resetting the parameters
    removeAll_button_ax = fig.add_axes([0.8, 0.025, 0.15, 0.04])
    removeAll_button = Button(removeAll_button_ax, 'Remove All', color=axis_color, hovercolor='0.975')
    removeAll_button.on_clicked(removeAll_button_on_clicked)

    #add a save button for saving the current annotatation for the current number of segmetns and label
    save_button_ax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
    save_button = Button(save_button_ax, 'Save', color=axis_color, hovercolor='0.975')
    save_button.on_clicked(save_button_on_clicked)

    # Add a set of radio buttons for changing labels
    # this must include exiting current labelled array if label is different from current @
    label_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], axisbg=axis_color)
    label_radios = RadioButtons(label_radios_ax, label_list, active=0)
    label_radios.on_clicked(label_radios_on_clicked)

    # create the event mouse click and keyboard press handles
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_keypress = fig.canvas.mpl_connect('key_press_event', on_key)

    #Display the image with the segment boundaries and annotated segments highlighted
    (segmented_image, segments, item_number)= display_image(image, image_axes, numSegments, annot_mask, curr_image_file, currentLabel)

    plt.axis('image')
    plt.axis("off")
    plt.show()

    while not(quit_request or quit_and_save_request):
        plt.pause(0.01)

    if (quit_and_save_request):
        # Check if anything is labelled before trying to save
        if (annot_mask.all != 0):
            # this dictionary update should be done earlier when user move on to another category or image @
            annotation_dict = update_annot_dict(curr_image_file, currentLabel= currentLabel, annot_mask= annot_mask, \
                segType= segType, segParameters={'numSegments' : numSegments})
            save_image(annot_dir, curr_image_file, annotation_dict)
