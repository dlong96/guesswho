
import time
import cv2
import os


def video_to_frames(input_loc, output_loc):

    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    count = 0
    crop_no=0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        count = count + 1
        # Write the results back to output location.
        if count % 5 == 0:  #modular
            crop_no +=1
            cv2.imwrite(output_loc + "/%#05d.jpg" % (crop_no), frame)

        else:
            pass
        # If there are no more frames left
        #if (count > (video_length-1)):
        if (count > 500):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % crop_no)
            break

if __name__=='__main__':
    input_loc=r"C:\Users\video_to_frame\LHVU0996.MP4"
    output_loc=r"C:\Users\video_to_frame\frames"
    video_to_frames(input_loc, output_loc)
