from utils.util_textract import *

import os

# Extract external table frame
######
path = 'LocalData/'
os.chdir(path)
list_file = [x for x in os.listdir() if 'No' in x]

output_dir = 'output_tablecrop'
if os.path.isdir(output_dir) is False:
    os.mkdir(output_dir)

for i, file_name in enumerate(list_file):
    pano_num = check_filenum(file_name)  # Remove both heads
    img = cv2.imread('./{}'.format(file_name))
    table_extract(img, pano_num, output_dir)


# Extract tablecell
#####
path = 'C:/Users/neilc/Projects/Dental_Panorama/LocalData/'
os.chdir(path)
dir_data = './output_tablecrop/'
dir_output = 'output_cellparsed'

list_file = [x for x in os.listdir(dir_data)]

if os.path.isdir(dir_output) is False:
    os.mkdir(dir_output)

for i, file_name in enumerate(list_file):
    print(file_name)
    pano_num = file_name[:8]
    img = cv2.imread(dir_data + file_name)
    img_f, k_length = outline_parser(img)
    cnt = find_contours(img_f)
    _, bounding_boxes = sort_contours(cnt)
    print(len(bounding_boxes))

    dir_single = dir_output + '/' + pano_num
    if os.path.isdir(dir_single) is False:
        os.mkdir(dir_single)

    intable_parser(img, bounding_boxes, pano_num, dir_single, k_length)

