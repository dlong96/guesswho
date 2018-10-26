import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from generate_xml import write_xml

from graphics import *

image = None
tl_list = []
br_list = []
object_list = []
obj=''
# constants
image_folder =r'C:\Users\Desktop\video\frames'
savedir = r'C:\Users\Desktop\video'



def line_select_callback(clk, rls):
	global tl_list
	global br_list
	global object_list
	tl_list.append((int(clk.xdata), int(clk.ydata)))
	br_list.append((int(rls.xdata), int(rls.ydata)))



def onkeypress(event):
	global object_list
	global tl_list
	global br_list
	global image

	if event.key == 'q':
		write_xml(image_folder, img, object_list, tl_list, br_list, savedir)
		tl_list = []
		br_list = []
		object_list = []
		image = None


	elif event.key == 'r':
		tl_list.pop()
		br_list.pop()


	elif event.key=='a':
		win = GraphWin('label',300,50)
		instructions = Text(Point(100, 10),
						 "which obj are you labelling?")
		instructions.draw(win)

		entry = Entry(Point(70,30),10)
		entry.draw(win)
		win.getMouse()
		obj = entry.getText()
		object_list.append(obj)
		print(object_list)
		win.getMouse()
		win.close()



def toggle_selector(event):
	toggle_selector.RS.set_active(True)





if __name__ == '__main__':


	for n, image_file in enumerate(os.scandir(image_folder)):


		img = image_file
		fig, ax = plt.subplots(1, figsize=(10.5, 8))
		mngr = plt.get_current_fig_manager()
		mngr.window.setGeometry(250, 40, 800, 600)
		image = cv2.imread(image_file.path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ax.imshow(image)

		toggle_selector.RS = RectangleSelector(
			ax, line_select_callback,
			drawtype='box', useblit=True,
			button=[1], minspanx=5, minspany=5,
			spancoords='pixels', interactive=True,
		)


		bbox = plt.connect('key_press_event', toggle_selector)
		key = plt.connect('key_press_event', onkeypress)


		plt.tight_layout()
		plt.show()
		plt.close(fig)
