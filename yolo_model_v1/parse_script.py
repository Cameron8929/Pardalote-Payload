# Split between train and val folders
# src code obtained from https://www.ejtech.io/learn/train-yolo-models
from pathlib import Path
import random
import os
import sys
import shutil

def find_project_boxes():
    """Find project_boxes folder starting from current directory"""
    current_dir = Path.cwd()
    
    # Search in current directory and all subdirectories
    for path in current_dir.rglob("project_boxes"):
        if path.is_dir():
            print(f"Found project_boxes at: {path}")
            return str(path)
    
    # If not found, search in parent directories
    parent = current_dir.parent
    while parent != parent.parent:  # Stop at root
        project_boxes = parent / "project_boxes"
        if project_boxes.exists() and project_boxes.is_dir():
            print(f"Found project_boxes at: {project_boxes}")
            return str(project_boxes)
        parent = parent.parent
    
    return None

def main_parse_script():
    
  # Auto-find the project_boxes folder
  data_path = find_project_boxes()
  if not data_path:
      print("Could not find 'project_boxes' folder. Please make sure it exists in the current directory or its subdirectories.")
      sys.exit(0)

  # Set train percentage (you can change this value)
  train_percent = 0.8

  # Check for valid entries
  if not os.path.isdir(data_path):
      print('Directory specified by auto-detection not found. Please check that project_boxes folder exists.')
      sys.exit(0)

  if train_percent < 0.01 or train_percent > 0.99:
      print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
      sys.exit(0)

  val_percent = 1 - train_percent

  # Define path to input dataset
  input_image_path = os.path.join(data_path, 'images')
  input_label_path = os.path.join(data_path, 'labels')

  # Define paths to image and annotation folders
  cwd = os.getcwd()
  train_img_path = os.path.join(cwd, 'data/train/images')
  train_txt_path = os.path.join(cwd, 'data/train/labels')
  val_img_path = os.path.join(cwd, 'data/validation/images')
  val_txt_path = os.path.join(cwd, 'data/validation/labels')

  # Create folders if they don't already exist
  for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
      if not os.path.exists(dir_path):
          os.makedirs(dir_path)
          print(f'Created folder at {dir_path}.')

  # Get list of all images and annotation files
  img_file_list = [path for path in Path(input_image_path).rglob('*')]
  txt_file_list = [path for path in Path(input_label_path).rglob('*')]
  print(f'Number of image files: {len(img_file_list)}')
  print(f'Number of annotation files: {len(txt_file_list)}')

  # Determine number of files to move to each folder
  file_num = len(img_file_list)
  train_num = int(file_num * train_percent)
  val_num = file_num - train_num
  print('Images moving to train: %d' % train_num)
  print('Images moving to validation: %d' % val_num)

  # Select files randomly and copy them to train or val folders
  for i, set_num in enumerate([train_num, val_num]):
      for ii in range(set_num):
          img_path = random.choice(img_file_list)
          img_fn = img_path.name
          base_fn = img_path.stem
          txt_fn = base_fn + '.txt'
          txt_path = os.path.join(input_label_path, txt_fn)
          if i == 0:  # Copy first set of files to train folders
              new_img_path, new_txt_path = train_img_path, train_txt_path
          elif i == 1:  # Copy second set of files to the validation folders
              new_img_path, new_txt_path = val_img_path, val_txt_path
          shutil.copy(img_path, os.path.join(new_img_path, img_fn))
          # os.rename(img_path, os.path.join(new_img_path,img_fn))
          if os.path.exists(txt_path):  # If txt path does not exist, this is a background image, so skip txt file
              shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))
              # os.rename(txt_path,os.path.join(new_txt_path,txt_fn))
          img_file_list.remove(img_path)