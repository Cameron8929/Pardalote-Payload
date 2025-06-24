# Training the Model with new data

The set of instructions is provided in order to make changes to the src files used to create the synthetic data, change the amount of image data. To use the pretrained model without altering anything then skip to section *Using the Pretrained Model.*

## Prerequisites

Before starting the following should be installed:

* OpenCV (pip3 install opencv-python)
* NumPy (pip3 install numpy)
* YOLOv11 (pip3 install ultralytics)
* Label Studio (pip3 install label-studio)

## Creating the Synthetic Thermal Imagery Data

Use the following command to begin the program:

```
python3 main.py
```

Run the program twice, first selecting 0 and then selecting 1 after using label studio to draw bounding boxes around the data.

### Annotating the Thermal data Using Label Studio

After selecting 0 in the previous step the thermal data will be located in the folder *thermal_with_gradient_spots*. In total the training and validation will comprise of 149 images. These values were chosen to approximately match a 70/20/10 train/val/test split. If the number of total images (including test) is increased then the variables *v1* and *v2* can replace the the hardcoded values. The images in *thermal_with_gradient_spots* should be uploaded to label studio and labelled. Firstly start label studio with the following command:

```
label-studio
```

Then log on and access http://localhost:8080. Sign up with an email address. Create a project and upload the images to that created project under the *Data Import* tab. If the amount of images is greater than 100 they must be uploaded in batches of 100. The final setup setting is to define the class labels in the *labelling setup* tab. Click the *Object Detection with Bounding Boxes* option. Then delete the default labels and enter the desired labels for the dataset. Each new label should be seperated by a new line. 

For more information see https://labelstud.io/guide/quick_start for information about using label studio. For information about how to use label studio to annotate the nozzle class onto the images then see here https://www.ejtech.io/learn/train-yolo-models.

After all images are labelled then download the images from label studio. The images will come as one zip. Download, unzip and then rename the folder to *project_boxes*.

### Parsing the data

The data will need to be parsed into a form that YOLOv11 can recognise. After running the command 1 in the *main.py* file, the file structure should look like the following:

```
data/
├── train/
│   ├── images/
│   └── labels/
└── validation/
    ├── images/
    └── labels/
```

On completion the number of images moved to training and validation will show in the terminal:

```
Number of image files: 149
Number of annotation files: 149
Images moving to train: 119
Images moving to validation: 30
```

### Creating the .yaml file

Next, create the .yaml configuration file. Since we're only detecting one object type (nozzles), set *nc = 1*, and *names = ["nozzle"]*:

```
path: ./data
train: train/images
val: validation/images
nc: 1
names: ["nozzle"]
```

### Training the Model

At this point the model can be trained. The number of epochs and image size can be set here by changing *epochs* and *imgsz*. To train the model then use the following command:

```
yolo detect train data=data.yaml model=yolo11s.pt epochs=30 imgsz=640
```

The results of the model and weights will be located in the **run/detect/trainX.**

# Using the Pretrained Model

The model trained above will be located here:

```
runs/detect/
├── train1/
    ├── weights/
    	└── best.pt/

```

Where train1 will be the model just trained. If this is not the first time training, then it will correspond to the amount of times trained:

```
yolo predict model=runs/detect/train1/weights/best.pt source=test_images/image_one.png conf=0.01 save=True
```

To use the pretrained model to test *image_one* run:

```
yolo predict model=runs/detect/with_validation/weights/best.pt source=test_images/thermal_satellite_0000.png conf=0.01 save=True
```

The prediction of the model will be located in the **run/detect/predictX.** Where X is the number of times YOLOv11 has been used for prediction.
