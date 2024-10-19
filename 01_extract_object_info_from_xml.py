import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et

# load all xml files and store in a list
xmlfiles = glob('./datasets/container/*.xml')
#data cleaning. replace \\ with /
replace_text = lambda x: x.replace('\\', '/')
xmlfiles = list(map(replace_text, xmlfiles))

#print(xmlfiles)

# step-2: read xml files
# from each xml file we need to extract
# filename, size(with, height), object(name, xmin, xmax, ymin, ymax)

def extract_text(filename):
	tree = et.parse(filename)
	root = tree.getroot()

	# extract filename
	image_name = root.find('filename').text
	# extract width and height
	width = root.find('size').find('width').text
	height = root.find('size').find('height').text

	objs = root.findall('object')
	parser = []

	for obj in objs:
		name = obj.find('name').text
		bndbox = obj.find('bndbox')
		xmin = bndbox.find('xmin').text
		xmax = bndbox.find('xmax').text
		ymin = bndbox.find('ymin').text
		ymax = bndbox.find('ymax').text
		parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])

	return parser

parser_all = list(map(extract_text, xmlfiles))

data = reduce(lambda x, y: x + y, parser_all)

df = pd.DataFrame(data, columns = ['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

# type conversion
cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)

# center x, center y
df['center_x'] = ((df['xmax'] + df['xmin']) / 2) / df['width']
df['center_y'] = ((df['ymax'] + df['ymin']) / 2) / df['height']

# w
df['w'] = (df['xmax'] - df['xmin']) / df['width']
# h
df['h'] = (df['ymax'] - df['ymin']) / df['height']

images = df['filename'].unique()

# 80% train and 20% test
img_df = pd.DataFrame(images, columns = ['filename'])
img_train = tuple(img_df.sample(frac = 0.8)['filename']) # shuffle and pick 80% of images
img_test = tuple(img_df.query(f'filename not in {img_train}')['filename']) # take rest 20% images

train_df = df.query(f'filename in {img_train}')
test_df = df.query(f'filename in {img_test}')

# label encoding
def label_encoding(x):
	labels = {"dent": 0, "hole": 1, "rust": 2}
	return labels[x]

train_df['id'] = train_df['name'].apply(label_encoding)
test_df['id'] = test_df['name'].apply(label_encoding)



import os
from shutil import move

train_folder = 'datasets/container/train'
test_folder = 'datasets/container/test'

os.mkdir(train_folder)
os.mkdir(test_folder)

cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

def save_data(filename, folder_path, group_obj):
	# move image
	scr = os.path.join('./datasets/container', filename)
	dst = os.path.join(folder_path, filename)
	move(scr, dst) # move img to the destnation folder

	#save the labels
	text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
	group_obj.get_group(filename).set_index('filename').to_csv(text_filename, sep = ' ', index = False, header = False)

filename_series_train = pd.Series(groupby_obj_train.groups.keys())
filename_series_train.apply(save_data, args = (train_folder, groupby_obj_train))

filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_test.apply(save_data, args = (test_folder, groupby_obj_test))
