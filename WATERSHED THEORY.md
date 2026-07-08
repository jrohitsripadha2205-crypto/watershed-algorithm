**Watershed Algorithm for Image Segmentation**



The Watershed Algorithm is an image segmentation technique that treats a grayscale image like a topographic landscape and separates different objects by finding boundaries (watershed lines) between them.



Input Image - upload an image

&#x20;     │

&#x20;     ▼

Convert to Grayscale - convert it to black and white 

&#x20;     │

&#x20;     ▼

Noise Removal - This smooths the image (Apply filters such as:- Gaussian Blur ,Median Blur)

&#x20;     │

&#x20;     ▼

Thresholding - Convert the image into a binary image.

&#x20;     │

&#x20;     ▼

Morphological Operations - Use operations like: Opening (remove small noise), Closing (fill small holes)

&#x20;     │

&#x20;     ▼

Find Sure Background - Dilate the image. The expanded area becomes the definite background.

&#x20;     │

&#x20;     ▼

Distance Transform - Threshold the distance transform to obtain the sure foreground.

&#x20;     │

&#x20;     ▼

Find Sure Foreground - Pixels near the centre of objects receive higher values.



&#x20;     │

&#x20;     ▼

Unknown Region - Assign labels to each object.

&#x20;     │

&#x20;     ▼

Marker Labelling - When two floods meet → a boundary (watershed line) is created.

&#x20;     │

&#x20;     ▼

Apply Watershed 

&#x20;     │

&#x20;     ▼

Segmented Image - Each object gets a different label. Object boundaries are clearly separated.



**PARAMETERS:-**



1\. Gaussian Blur Kernel Size:- Before applying watershed, the image is smoothed using a Gaussian filter to remove noise.



2\. Threshold Value :- Thresholding converts a grayscale image into a binary image.



3\. Adaptive Threshold Parameters :- Instead of one threshold, different regions receive different threshold



4\. Morphological Kernel Size :-

Used during



Erosion

Dilation

Opening

Closing



5\. Number of Morphological Iterations:- Controls how many times dilation or erosion is repeated.



6\. Distance Transform Type:- 



DIST\_L1



Uses Manhattan distance.



Distance



|x| + |y|



Fast



Less accurate.



DIST\_L2



Uses Euclidean distance.



Distance



√(x²+y²)



Most commonly used.



DIST\_C



Chessboard distance.



Rarely used.



7\. Distance Mask Size :- Controls neighbourhood size during distance transform.



8\. Distance Threshold :- Determines the sure foreground.



9\. Marker Labels :- Assigns unique numbers.



10\. Connectivity:- Determines which neighbouring pixels belong to the same object.



11\. Marker Type:- Markers can be ( Manual, Automatic )



12\. Watershed Boundary Value:- Marks watershed lines	Output boundary pixels are labelled -1


**LIBRARIES:-** 



OpenCV → Core image processing and Watershed algorithm.

NumPy → Image array manipulation and mathematical operations.

PyQt6 → Multi-tab graphical user interface.

Pillow → Display OpenCV images in the GUI.

Matplotlib → Histograms and visualization of parameter changes.



APPLICATION LAYOUT:- 



\------------------------------------------------------------

&#x20;          Interactive Watershed Segmentation Tool



&#x20;Load Image      Save Output      Reset



\------------------------------------------------------------



Original Image          Segmented Image



\------------------------------------------------------------



Tabs



General

Pre-processing

Thresholding

Morphology

Distance Transform

Markers

Watershed

Analysis



**TOOLS:-** 


| Component            | Recommendation              |

| -------------------- | --------------------------- |

| Programming Language | Python                      |

| GUI                  | OpenCV + PyQt6 (or PySide6) |

| Image Processing     | OpenCV                      |

| Numerical Operations | NumPy                       |

| Charts/Histogram     | Matplotlib                  |

| Optional Analysis    | scikit-image                |



**CODE:-** 



def load\_image():

&#x20;   global original



&#x20;   file\_path = filedialog.askopenfilename(

&#x20;       filetypes=\[("Images", "\*.png \*.jpg \*.jpeg \*.bmp")]

&#x20;   )



&#x20;   if file\_path == "":

&#x20;       return



&#x20;   original = cv2.imread(file\_path)

&#x20;   process\_image()







