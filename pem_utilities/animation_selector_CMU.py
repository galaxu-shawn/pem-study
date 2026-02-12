"""
CMU Motion Capture Database Animation Selector

This module provides functionality to select and retrieve animation files from the 
Carnegie Mellon University (CMU) Motion Capture Database. It categorizes animations 
by activity type and allows for random or targeted selection of motion capture data.

The CMU database contains thousands of motion capture recordings of human subjects 
performing various activities. This module parses the database index, categorizes 
activities based on keywords, and provides an interface to select specific animations 
or random animations within activity categories.

Dependencies:
    - numpy: For random selection functionality
    - csv: For parsing the CMU database index file
    - os: For file path operations
    - pathlib: For modern path handling

Author: ANSYS, Inc.
Contact: arien.sligar@ansys.com for CMU database access
"""

import os
import csv
import numpy as np

from pathlib import Path


def select_CMU_animation(file_name=None, activity_type=None):
    """
    Select a CMU motion capture animation file by name or activity type.
    
    This function provides access to the CMU Motion Capture Database by either
    selecting a specific animation file or randomly choosing from animations
    categorized by activity type. It searches for both compressed (.daez) and
    uncompressed (.dae) versions of the animation files.
    
    Parameters
    ----------
    file_name : str, optional
        Specific animation file to select in format "##_###" (e.g., "01_01").
        This corresponds to the CMU ID number from the database. Do not include
        file extension - the function will automatically search for .daez 
        (compressed) files first, then .dae files. Default is None.
        
    activity_type : str, optional
        Activity category for random selection when file_name is None.
        Available activity types include:
        - 'walking': Walking, stepping, limping, marching, pacing
        - 'sitting': Sitting motions
        - 'getting_up': Getting up from seated position
        - 'dancing': Dance movements
        - 'jumping': Jumping and hopping motions
        - 'running': Running and jogging
        - 'kicking': Kicking motions
        - 'standing': Standing poses and movements
        - 'punching': Boxing and punching motions
        - 'climbing': Climbing movements
        - 'picking up': Picking up objects
        - 'sport': Sports activities (basketball, soccer, golf, etc.)
        - 'swimming': Swimming motions
        - 'animal_behavior': Animal-like movements
        - 'waving': Waving gestures
        - 'eating_drinking': Eating and drinking motions
        - 'misc': Miscellaneous activities not fitting other categories
        
        If None, random selection from all available animations. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - filename (str): Absolute path to the selected animation file
        - description (dict): Dictionary containing animation metadata with keys:
            - 'subject_number': CMU subject number
            - 'instance': Animation instance number
            - 'description': Primary description from CMU database
            - 'description_2': Secondary description from CMU database
            - 'activity': Categorized activity type
            
    Raises
    ------
    FileNotFoundError
        If the CMU database index file is not found
        
    Notes
    -----
    The function prioritizes activity categorization in the order listed in the
    activity_type parameter description. Some animations may fit multiple
    categories but will be assigned to the first matching category.
    
    The CMU database must be downloaded separately. Contact arien.sligar@ansys.com
    for download instructions if the database is not found.
    
    Examples
    --------
    Select a specific animation:
    >>> filename, desc = select_CMU_animation(file_name="01_01")
    
    Select a random walking animation:
    >>> filename, desc = select_CMU_animation(activity_type="walking")
    
    Select any random animation:
    >>> filename, desc = select_CMU_animation()
    """

    # Resolve paths dynamically relative to the current script location
    # This ensures the function works regardless of where it's called from
    script_path = Path(__file__).resolve()
    model_path = os.path.join(script_path.parent.parent, 'models')

    # Construct path to CMU database directory and index file
    base_path = os.path.join(model_path, 'CMU_Database')
    database_file = 'cmu-mocap-index-simple.csv'
    database_file = os.path.join(base_path, database_file)
    
    # Verify database exists, provide helpful error message if not
    if not os.path.exists(database_file):
        print('ERROR: CMU Database does not exists, contact arien.sligar@ansys.com for download link')

    # Initialize containers for activity categorization
    all_activities = []  # Track all unique activity types found
    data_dict = {}       # Store animation metadata keyed by filename
    
    # Parse the CMU database index CSV file
    with open(database_file, newline='') as csvfile:
        description_database = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        next(description_database)  # Skip header row
        
        for i, line in enumerate(description_database):
            temp_dict = {}
            
            # Skip empty lines
            if any(x.strip() for x in line):
                # Extract subject number and instance from filename (format: ##_###)
                sub_number = line[0].split('_')[0]
                temp_dict['subject_number'] = sub_number
                temp_dict['instance'] = line[0].split('_')[1]
                
                # Store primary description
                description = line[1]
                temp_dict['description'] = description
                
                # Clean and process secondary description
                description2 = line[2]
                description2 = description2.replace('Subject #', '')
                if description2[0] == '0':
                    description2 = description2.replace('0', '')
                description2 = description2.replace(sub_number, '')
                description2 = description2.lstrip()
                description2 = description2.rstrip()
                description2 = description2.replace(')', '')
                description2 = description2.replace('(', '')
                temp_dict['description_2'] = description2

                # Default activity category
                temp_dict['activity'] = 'misc'
                
                # Combine both descriptions for keyword matching
                full_desc = description + " " + description2
                
                # Categorize activities based on keywords in descriptions
                # Priority order matters - first match wins
                if ('walk' in full_desc.lower() or
                        'step' in full_desc.lower() or
                        'limping' in full_desc.lower() or
                        'march' in full_desc.lower() or
                        'pacing' in full_desc.lower() or
                        'mope' in full_desc.lower()
                ):
                    temp_dict['activity'] = 'walking'
                elif 'sit' in full_desc.lower():
                    temp_dict['activity'] = 'sitting'
                elif ('get up' in full_desc.lower() or
                      'getting up' in full_desc.lower()
                ):
                    temp_dict['activity'] = 'getting_up'
                elif ('dance' in full_desc.lower() or 'dancing' in full_desc.lower()):
                    temp_dict['activity'] = 'dancing'
                elif ('jump' in full_desc.lower() or 'hop' in full_desc.lower()):
                    temp_dict['activity'] = 'jumping'
                elif ('run' in full_desc.lower() or 'jog' in full_desc.lower()):
                    temp_dict['activity'] = 'running'
                elif 'kick' in full_desc.lower():
                    temp_dict['activity'] = 'kicking'
                elif 'stand' in full_desc.lower():
                    temp_dict['activity'] = 'standing'
                elif ('boxing' in full_desc.lower() or
                      'punch' in full_desc.lower()
                ):
                    temp_dict['activity'] = 'punching'
                elif 'climb' in full_desc.lower():
                    temp_dict['activity'] = 'climbing'
                elif 'pick' in full_desc.lower():
                    temp_dict['activity'] = 'picking up'
                elif ('basketball' in full_desc.lower() or
                      'soccer' in full_desc.lower() or
                      'ball' in full_desc.lower() or
                      'sport' in full_desc.lower() or
                      'golf' in full_desc.lower() or
                      'catch' in full_desc.lower() or
                      'throw' in full_desc.lower() or
                      'yoga' in full_desc.lower() or
                      'flip' in full_desc.lower()
                ):
                    temp_dict['activity'] = 'sport'
                elif 'swim' in full_desc.lower():
                    temp_dict['activity'] = 'swimming'
                elif 'animal' in full_desc.lower():
                    temp_dict['activity'] = 'animal_behavior'
                elif 'wave' in full_desc.lower():
                    temp_dict['activity'] = 'waving'
                elif ('eat' in full_desc.lower() or
                      'drink' in full_desc.lower()):
                    temp_dict['activity'] = 'eating_drinking'
                
                # Track all activity types and store animation data
                all_activities.append(temp_dict['activity'])
                data_dict[line[0]] = temp_dict

    # Create unique list of all activity types found in database
    all_activities = list(set(all_activities))
    
    # Organize animations by activity type for efficient random selection
    data_dict_by_activity = {}
    for activity in all_activities:
        temp_dict = {}
        for inst in data_dict:
            if activity in data_dict[inst]['activity']:
                temp_dict[inst] = data_dict[inst]
        data_dict_by_activity[activity] = temp_dict

    # Handle specific file selection
    if file_name is not None:
        # Look for compressed version first, then uncompressed
        if os.path.exists(os.path.join(base_path, file_name) + '.daez'):
            file_name = os.path.join(base_path, file_name) + '.daez'
        elif os.path.exists(f'{file_name}.daez'):
            file_name = f'{file_name}.daez'
    else:
        # Handle random selection based on activity type or completely random
        tries = 1
        
        if activity_type is None:
            # No activity specified - select randomly from all animations
            file_name = np.random.choice(list(data_dict.keys()))
        elif activity_type in all_activities:
            # Valid activity type - select randomly from that category
            file_name = np.random.choice(list(data_dict_by_activity[activity_type].keys()))
        else:
            # Invalid activity type - fall back to random selection with warning
            print(f'Activity Type {activity_type} not found, making random selection')
            file_name = np.random.choice(list(data_dict.keys()))
        
        # Attempt to find the actual file (try multiple variations and locations)
        while tries < 5:
            # Check for compressed version in base path
            if os.path.exists(os.path.join(base_path, file_name) + '.daez'):
                file_name = os.path.join(base_path, file_name) + '.daez'
                break
            # Check for compressed version in current directory
            elif os.path.exists(f'{file_name}.daez'):
                file_name = f'{file_name}.daez'
                break
            # Check for uncompressed version in base path
            elif os.path.exists(f'{base_path}{file_name}.dae'):
                file_name = f'{base_path}{file_name}.dae'
                break
            tries += 1

    # Ensure absolute path for return value
    file_name = os.path.abspath(file_name)
    
    # Extract filename without extension to lookup metadata
    name_only = os.path.splitext(os.path.basename(file_name))[0]
    
    # Retrieve animation description/metadata if available
    data_description = ''
    if name_only in data_dict.keys():
        data_description = data_dict[name_only]
        
    return os.path.abspath(file_name), data_description
