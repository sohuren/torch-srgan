import os, json, argparse
from threading import Thread
from Queue import Queue

import numpy as np 
from scipy.misc import imread, imresize, imsave
from PIL import Image
import h5py
import time

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',default='/data/imagenet_val')
parser.add_argument('--val_dir',default='/data/Set14')
parser.add_argument('--output_file',default='/data/imagenet-val-192.h5')

parser.add_argument('--height',type=int,default=480) # we need to adjust it now
parser.add_argument('--width',type=int,default=640) # we need to adjust it now
parser.add_argument('--scale',type=int,default=4)
parser.add_argument('--max_images',type=int,default=-1)
parser.add_argument('--num_workers',type=int,default=2)
parser.add_argument('--include_val',type=int,default=1)
args = parser.parse_args()

def add_data(h5_file, image_dir, prefix, args):
	image_list = []
	image_extensions = {'.jpg','.jpeg','.JPG','.JPEG','.png','.PNG','.bmp'} # for the glass now
	for filename in os.listdir(image_dir):
		ext = os.path.splitext(filename)[1]
		name = os.path.splitext(filename)[0]
		if ext in image_extensions and name.endswith('_g'):
			image_list.append(os.path.join(image_dir,filename))
	if args.max_images > 0:
		num_images = args.max_images
	else:
		num_images = len(image_list)

	dset_data_name = prefix + '_data'
	dset_data_size = (num_images, 1, args.height, args.width)
	imgs_dset_data = h5_file.create_dataset(dset_data_name,dset_data_size,np.uint8)

	dset_label_name = prefix + '_label'
	dset_label_size = (num_images, 1, args.height, args.width)
	imgs_dset_label = h5_file.create_dataset(dset_label_name,dset_label_size,np.uint8)


	input_queue = Queue()
	output_queue = Queue()

	def read_worker():
		while True:
			idx, filename = input_queue.get()
		
			try:
				
				# read the image with glass
				img_glass = imread(filename)				
				H, W = img_glass.shape[0], img_glass.shape[1]
				data = Image.fromarray(np.uint8(img_glass))
				# scale the short edge to arg.width or arg.height
				if H <= W:
					if H < args.height:
						data = data.resize((W * args.height / H , args.height), Image.ANTIALIAS)
				else:
					if W < args.width:
						data = data.resize((args.width, H * args.width / W), Image.ANTIALIAS)
				# center crop
				W, H   = data.size
				left   = (W - args.width ) / 2
				top    = (H - args.height) / 2
				right  = (W + args.width ) / 2
				bottom = (H + args.height) / 2
				
				data = data.crop((left, top, right, bottom))
				data = data.resize((args.width, args.height),Image.ANTIALIAS)
			
	
				#W1, H1 = data.size
				#print(W1)
				#print(H1)	
				# read the image without glasses
				img_noglass = imread(filename.replace('_g',''))
				H, W = img_noglass.shape[0], img_noglass.shape[1]
				label = Image.fromarray(np.uint8(img_noglass))
				# scale the short edge to arg.width or arg.height
				if H <= W:
					if H < args.height:
						label = label.resize((W * args.height / H , args.height), Image.ANTIALIAS)
				else:
					if W < args.width:
						label = label.resize((args.width, H * args.width / W), Image.ANTIALIAS)
				# center crop
				W, H   = label.size
				left   = (W - args.width ) / 2
				top    = (H - args.height) / 2
				right  = (W + args.width ) / 2
				bottom = (H + args.height) / 2
				
				label = label.crop((left, top, right, bottom))
				#label = label.resize((args.width/args.scale, args.height/args.scale),Image.ANTIALIAS)
				
				#W1, H1 = label.size
				#print(W1)
				#print(H1)


			except (ValueError, IndexError) as e:
				print filename
				print img.shape, img.dtype
				print e

			input_queue.task_done()
			output_queue.put((idx, np.asarray(data), np.asarray(label)))

	def write_work():
		num_written = 0
		tic = time.time()
		while True:
			idx, data, label = output_queue.get()
			# Grayscale image; it is H x W so broadcasting to C x H x W will just copy
  		  	# grayscale values into all channels.
			imgs_dset_label[idx] = label
			imgs_dset_data[idx] = data
			output_queue.task_done()
			num_written = num_written + 1
			if num_written % 100 == 0:
				print 'Copied %d / %d images, times: %4fsec' % (num_written, num_images, time.time() - tic)
				tic = time.time()

	for i in xrange(args.num_workers):
		t = Thread(target=read_worker)
		t.daemon = True
		t.start()

	t = Thread(target=write_work)
	t.daemon = True
	t.start()

	for idx, filename in enumerate(image_list):
		if args.max_images > 0 and idx >= args.max_images: break
		input_queue.put((idx, filename))
	input_queue.join()
	output_queue.join()




if __name__ == '__main__':
	print(args)
	with h5py.File(args.output_file,'w') as f:
		if args.include_val:
			add_data(f, args.val_dir, 'val', args)
			print('val set done!')
		add_data(f, args.train_dir, 'train', args)
		print('train set done!')
