# GuessWho
**CNN classification playground**

Inspired by draw guess game and The Simpson characters recognization by Akexabdre Attia.<br/>
I trained on ~100 pictures I got from google imgae of real bottle, and ~100 pircture of everything else. For test set, I invited friends to send me over their hand draw bottle,~20 images.<br/>
The perdictions were all right, and in handdraw folder there are a few more result images, if you want to look at them;)<br/>
I am aware the structure of bottle is relatively simple, which is the reason I pick it as the first place.
![input](https://github.com/minibutterbread/guesswho/blob/master/handdraw/IMG_0195.jpg)

![result](https://github.com/minibutterbread/guesswho/blob/master/handdraw/IMG_0196.jpg)



I decide to use more complex image, <del>and for the reason of easy assessing, selfie is my choice.<br/></del>**change to cifar10**
**Task**<br/>
<del>Two person(female/male) to distinguish</del>

<del>50 image per class</del>

The result from tf.ipynb showed the train_acc and val_acc fluctuate a lot. The main reason that I suspect is the size of my dataset.<br/>
I also used VGG16 with extra dense layer as our classifer, but the train_acc and val_acc also fluctuate a lot.<br/>
Ideally, we want see a relative smooth increasing or decresing trend for first 5 epochs of each case(structure) listed, without fine-tuning on each layer/block. so I decided switch my dataset to cifar10 for more intuitive understanding on CNN structure.   Dataset will only be a part of cifar10, for speeding up computation time.<br/>

1.one hidden layer, fully connected model

2.cnn with 3conv layer,one pooling

3.more conv layer cnn

4.batch norm layer

5.resNet-like structure

6.Look at wrong predicted images. Plot image with high confidence score, but actually belong to another class. Plot images with probabilty close to 0.5.

7.training images contains individual person as main parts, use sigmoid as muti-label output, see the result of test image with two person init.

8.dataset inbalence: double the size of one class, see weather simply copy image of the other class to make up the size difference work or not.
