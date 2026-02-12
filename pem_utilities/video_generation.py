"""
Created on Sun Jan 28 10:00:00 2024

@author: asligar
"""

import cv2
import os


def generate_video_from_images(image_list,
                               output_name='animation.avi',
                               fps=30,
                               output_dir='./output/',
                               clean_up=True):
    """
    This function generates a video from a list of images.

    Parameters:
    ------------
    image_list : list
        A list of image file paths to be used in the video.

    output_name : str, optional
        The name of the output video file. Defaults to 'animation.avi'.

    fps : int, optional
        The frames per second of the output video. Defaults to 30.

    output_dir : str, optional
        The directory where the output video will be saved. Defaults to './output/'.

    clean_up : bool, optional
        If True, the images used in the video will be deleted after the video is created. Defaults to True.

    Returns:
    --------
    str
        The file path of the output video.
    """

    # Check if output directory exists, if not, create it
    os.makedirs(output_dir, exist_ok=True)

    # If there are less than 2 images, do not create a video
    if len(image_list) < 2:
        print("not making video")
        return

    # Define the codec using VideoWriter_fourcc() and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Read the first image from image_list to get the frame size
    frame = cv2.imread(image_list[0])
    height, width, layers = frame.shape

    # Initialize the VideoWriter object
    video = cv2.VideoWriter(os.path.join(output_dir, output_name), fourcc, float(fps), (width, height))
    print('\nRendering Video')

    # Write each frame to the video
    for n in range(len(image_list)):
        video.write(cv2.imread(image_list[n]))

    # Release the VideoWriter and destroy all windows
    cv2.destroyAllWindows()
    video.release()

    # If clean_up is True, delete all images in image_list
    if clean_up:
        for each in image_list:
            if os.path.isfile(each):
                os.remove(each)

    # Print the output file path
    out_filename = f'{output_dir}{output_name}'
    print(f'\nDone: {out_filename}')

    # Return the output file path
    return out_filename
def generate_video_from_multiple_videos(video_list,
                                        output_name='multi_animation.avi',
                                        output_dir='./output/',
                                        clean_up=False):
    """
    :param video_list:
    :param output_name:
    :param fps:
    :param output_dir:
    :param clean_up:
    :return:
    """

    # Check if output directory exists, if not, create it

    os.makedirs(output_dir, exist_ok=True)
    # If there are less than 2 videos, do not create a video
    if len(video_list) < 2:
        print("not making video")
        return
    # Define the codec using VideoWriter_fourcc() and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Read the first video from video_list to get the frame size
    video = cv2.VideoCapture(video_list[0])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get fps of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    # Initialize the VideoWriter object
    video_out = cv2.VideoWriter(os.path.join(output_dir,output_name), fourcc, float(fps), (width, height))
    print('\nRendering Video')
    # Write each frame to the video
    for n in range(len(video_list)):
        video = cv2.VideoCapture(video_list[n])
        while True:
            ret, frame = video.read()
            if not ret:
                break
            video_out.write(frame)
    video_out.release()

    # print the output file path
    out_filename = os.path.join(output_dir,output_name)
    print(f'\nDone: {out_filename}')