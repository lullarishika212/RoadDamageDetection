# RoadDamageDetection

## Dataset:
### <a href="https://crddc2022.sekilab.global/"> Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022)</a> 

- The [article](https://www.researchgate.net/publication/363668453_RDD2022_A_multi-national_image_dataset_for_automatic_Road_Damage_Detection) providing detailed statistics and other information for data released through CRDDC'2022 can be accessed [here](https://www.researchgate.net/publication/363668453_RDD2022_A_multi-national_image_dataset_for_automatic_Road_Damage_Detection)!

- [RDD2022.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip)
  - `RDD2022.zip` contains train and test data from six countries: **Japan,  India, Czech Republic, Norway, United States, and China.** 
  - Images (.jpg) and annotations (.xml) are provided for the train set. The format of annotations is the same as pascalVOC.
  - Only images are provided for test data.  

### Damage Categories to be considered
{1 : 'cracks', 2: 'pothole'}

## Technologies Used

- Deep Learning Models
  - YOLOv5
  - Faster R-CNN
  - SSD (Single Shot Multicast Detector)
- Deployment 
  - Streamlit
  
## Dependencies

- tensorflow
- keras
- pytorch
- object_detection
- Pillow
- opencv-python
- PyYAML

## Local Setup

1. The 'model training' directory contains the .ipynb files to train models. Train the models.
(Recommeded: Google colab to train the model)

2. After training, export the models:

- For YOLOv5:
  Download the best.pt/last.pt file which contains the best weights/weights computed in the last epoch
  
- For Faster R-CNN and SSD:
  Download the exported_models directory created after training. It contains the saved_model and checkpoints created while training for respective models.

3. Create a directory named 'models'
- Place the best.pt/last.pt file in it for YOLOv5
- Place the exported_model directories for Faster R-CNN and SSD in it.

4. Setup streamlit appliaction in your local system
- install streamlit using command in Windows cmd or Linux/Mac terminal<br>
  `pip install streamlit` 
- Run the 'app.py' file with command <br>
  `streamlit run app.py`
  
## Graphical User Interface
<br>

<img src=".result/op ui 1.png" width="30%"/>
The Graphical User Interace(GUI) for the model is created using Streamlit which provides the option of uploading the image to the user.
<img src=".result/op ui 2.1.png" width="30%"/>
<br>
The user then clicks on the "Detect" button to run the inferences on the uploaded image and detect road damages with their class probabilities.
<img src=".result/op ui final result.png" width="30%"/>
