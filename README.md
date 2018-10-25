# GuessWho
**CNN classification playground**

Inspired by draw guess game and The Simpson characters recognization by Akexabdre Attia

![input](https://github.com/minibutterbread/guesswho/blob/master/handdraw/IMG_0195.jpg=100x100)
![result](https://github.com/minibutterbread/guesswho/handdraw/IMG_0196.jpg)





Two person(female/male) to distinguish

50 image per class

1.one hidden layer, fully connected model

2.cnn with 3conv layer,one pooling

3.more conv layer cnn

4.batch norm layer

5.resNet-like structure


6.training images contains indidual person as main parts, use sigmoid as muti-label output, see the result of test image with two person init.

7.dataset inbalence: double the size of one class, see weather simply copy image of the other class to make up the size difference work or not.
