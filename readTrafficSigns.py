# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
	    X1.append(row[3])
            X2.append(row[5])
            Y1.append(row[4])
            Y2.append(row[6])
        gtFile.close()
    return images, labels,X1,X2,Y1,Y2

def magic_function(directory):
    #directory is the one we are doing work in 

    print('YOU HAVE CHOOSEN A VERY DRASTIC FUNCTION , YOU NEED TO BE PATIENT FOR THE TIME BEING .ALTHOUGH IT WILL GIVE YOU THE UPDATES BUT STILL TAKE A LONG TIME (FAR MORE WHAT YOU WOULD EXPECT).\n WANT TO KNOW WHY !!! GO READ THE CODE :)')

    print('Calling Function To Read Training Dataset')
    image,lable,X1,X2,Y1,Y2=readTrafficSigns(directory +'/GTSRB/Final_Training/Images')
    print('Reading Complete')

    print('Calling For Loop To Reshape And Creating The Tensors')
    image=np.asarray(image)
    r1=tf.pack(image[0])
    r1=tf.image.crop_to_bounding_box(r1,int(Y1[0]),int(X1[0]),int(Y2[0])-int(Y1[0]),int(X2[0])-int(X1[0]))
    r1=tf.cast(r1,tf.uint8)
    X_train=tf.pack([tf.image.resize_images(r1,48,48)])
    X_train=tf.cast(X_train,tf.uint8)
    for i in range(1,image.shape[0]):
	   r=tf.pack(image[i])
	   r=tf.image.crop_to_bounding_box(r,int(Y1[i]),int(X1[i]),int(Y2[i])-int(Y1[i]),int(X2[i])-int(X1[i]))
	   r=tf.image.resize_images(r,48,48)
	   r=tf.cast(r,tf.uint8)
	   X_train=tf.concat(0,[X_train,tf.pack([r])])  #This is the image vector which is a 4D tensor of shape(no_train_images,48,48,3)
    print('For Loop Ended')

    print('Running The Session')
    with tf.Session() as sess:
	   npimage=sess.run(X_train)
	   #npimage_=sess.run(X_test)
    npimage=npimage.astype(np.uint8)
    #npimage_=npimage_.astype(np.uint8)
    print('Session Ended')


    print('For Loop for Making a list for Binary ')
    lable=np.asarray(lable)
    lable=np.reshape(lable,(len(lable),1))
    lable=lable.astype(np.uint8)

    Final_binary_help=[]
    Final_binary_help2=[]
    for i in range(30000):
	   Final_binary_help.extend(lable[i])
	   Final_binary_help.extend(np.reshape(np.reshape(npimage[i],(48*48,3)).T,48*48*3))
    
    for i in range(30000,39209):
	   Final_binary_help2.extend(lable[i])
	   Final_binary_help2.extend(np.reshape(np.reshape(npimage[i],(48*48,3)).T,48*48*3))



    # convert it into a binary format and then enjoy the code 
    print('Creating Bianry File :)')
    f=open(directory+'/'+"data_batch_1.bin","wb")
    #Final_binary_help=Final_binary_help.astype(np.uint8)
    newFileByteArray = bytearray(Final_binary_help)
    f.write(newFileByteArray)
    f.close()
    f=open(directory+'/'+"test_batch.bin","wb")
    #Final_binary_help2=Final_binary_help2.astype(np.uint8)
    newFileByteArray = bytearray(Final_binary_help2)
    f.write(newFileByteArray)
    f.close()
    print('Finished :)')




