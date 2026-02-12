import glob
import numpy as np
import os


import re
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from pem_utilities.video_generation import generate_video_from_images

fps = 25
dynamic_range_db = 80  # Dynamic range in dB for visualization
root_dir = r'C:\tmp\output2'
path_to_frames = os.path.join(root_dir,'Capture_frames')
path_to_results = os.path.join(root_dir,'data')
output_path = root_dir

generate_scenario_video = False
generate_results_video = True

def get_frame_file_paths(path_to_frames):
    """
    Get a sorted list of all PNG file paths in the specified directory.

    Args:
        path_to_frames (str): The directory path where frame images are stored.
    Returns:
        list: A sorted list of file paths to PNG images.
    """
    frame_file_paths = glob.glob(os.path.join(path_to_frames, '*.png'))
    frame_file_paths.sort()
    return frame_file_paths

def get_results_file_path(path_to_result):
    """
    get all the npy files in teh directory and sort them
    Args:
        path_to_result (str): The directory path where result files are stored.
        Returns:
            list: A sorted list of file paths to npy files.
    """
    result_file_paths = glob.glob(os.path.join(path_to_result, '*.npy'))
    # these files have a name format like debug_response_awr_1642_timeidx_0.npy
    # sort by the index value at the end
    result_file_paths.sort(key=lambda x: int(x.split('timeidx_')[-1].split('.')[0]))

    return result_file_paths




# get all the paths of frame images (png files, generated in another rendering tool)
paths = get_frame_file_paths(path_to_frames)

# get all the paths of results npy files
results_paths = get_results_file_path(path_to_results)
# check on of the results and the total number of results
print(f'number of result files: {len(results_paths)}')
print(f'example result file path: {results_paths[0]}')  
# load one of the results to check

example_result = np.load(results_paths[0])
# print(f'example result data: {example_result}') 
print(f'result data shape: {example_result.shape}')
results_shape = example_result.shape
print(f'Results Length: {len(results_paths)}')
print(f' Frame Paths Length: {len(paths)}')
# check length of frame paths and make sure they are equal to results_path, if not trim the longer one
if len(paths) != len(results_paths):
    min_length = min(len(paths), len(results_paths))
    paths = paths[:min_length]
    results_paths = results_paths[:min_length]

    print(f'Trimmed paths to minimum length: {min_length}')


if generate_scenario_video:
    # generate video from images of the physical scenario. These are generated in another rendering tool.
    generate_video_from_images(paths,
                                output_name='scene_isaacsim.avi',
                                fps=fps,
                                output_dir=output_path,
                                clean_up=False)






def reduce_data(file_path):
    data = np.load(file_path)
    data = data[0,0]
    np.save(file_path, data) #trying to clean up data
    return data
def process_frame(file_path):
    # Load data: [Tx, Rx, Doppler, Range]
    data = np.load(file_path)
    # if data shape is not 2d, reduce and save
    if data.ndim != 2:
        data = reduce_data(file_path)

    # Combine channels: Non-coherent integration (average magnitude)
    # Shape becomes (Doppler, Range)
    # combined = np.mean(np.abs(data), axis=(0, 1))
    
    # Convert to dB
    # Add a small epsilon to avoid log(0)
    combined_db = 20 * np.log10(np.abs(data) + 1e-9)
    
    return combined_db

def export_images(files, output_file, fps=10):

    all_saved_files = []
    files = results_paths
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Processing...")

    # Setup video writer
    # We need to determine frame size from the first plot
    # Let's generate one frame to get dimensions
    
    # Setup plot style for "artistic" look
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    # canvas = FigureCanvas(fig)
    
    # Process first frame to set up limits
    first_frame = process_frame(files[0])
    
    # Determine dynamic range for consistent visualization
    # We can use a fixed dynamic range, e.g., 60 dB
    vmax = np.max(first_frame)
    vmin = vmax - dynamic_range_db
    
    im = ax.imshow(first_frame, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
    ax.axis('off') # Turn off axes for artistic look

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    for i, file_path in tqdm(enumerate(files)):
        print(file_path)
        frame_data = process_frame(file_path)
        im.set_data(frame_data)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        png_output = os.path.join(os.path.dirname(output_file), f"{base_name}_{i:04d}.png")
        
        plt.savefig(png_output, bbox_inches='tight', pad_inches=0)
        all_saved_files.append(png_output)
    plt.close(fig)
    return all_saved_files


# create results video
output_file = os.path.join(output_path, 'results_visualization.avi')
if generate_results_video:
    # create_video(results_paths, output_file)
    images_paths= export_images(results_paths, output_file)
    #generate video from images of the physical scenario. These are generated in another rendering tool.
    generate_video_from_images(images_paths,
                                output_name='results_visualization.avi',
                                fps=fps,
                                output_dir=output_path,
                                clean_up=True)




