# GuessWho
**CNN classification playground**

Inspired by draw guess game and The Simpson characters recognization by Akexabdre Attia.<br/>
I trained on ~100 pictures I got from google imgae of real bottle, and ~100 pircture of everything else. For test set, I invited friends to send me over their hand draw bottle,~20 images.<br/>
The perdictions were all right, and in handdraw folder there are a few more result images, if you want to look at them;)<br/>
I am aware the structure of bottle is relatively simple, which is the reason I pick it as the first place.
![input](https://github.com/minibutterbread/guesswho/blob/master/handdraw/IMG_0195.jpg)

![result](https://github.com/minibutterbread/guesswho/blob/master/handdraw/IMG_0196.jpg)



I decide to use more complex image, </del>and for the reason of easy assessing, selfie is my choice.<br/></del>
**Task**<br/>
<del>Two person(female/male) to distinguish</del>

<del>50 image per class</del>

The result from tf.ipynb showed the train_acc and val_acc fluctuate a lot. The main reason that I suspect is the size of my dataset.</del>
Ideally, we want see a relative smooth increasing or decresing trend for first 5 epochs of each case listed. so I decided to switch to cifar10.</del>

1.one hidden layer, fully connected model

2.cnn with 3conv layer,one pooling

3.more conv layer cnn

4.batch norm layer

5.resNet-like structure


6.training images contains individual person as main parts, use sigmoid as muti-label output, see the result of test image with two person init.

7.dataset inbalence: double the size of one class, see weather simply copy image of the other class to make up the size difference work or not.
