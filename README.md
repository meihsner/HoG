# Histogram of Oriented Gradients
The HoG (Histogram of Oriented Gradients) project aims to recreate the title function in a Python environment based on the article:
https://learnopencv.com/histogram-of-oriented-gradients/. The only public library that offers the presented method is the *skimage*.
The main assumption of the project was to learn the function of reading the feature descriptor by recreating it and own idea for the visualization of the result.

# Libraries:
- *OpenCV* - image reading and processing (4.5.3.56),
- *NumPy* - mathematical operations (1.21.2),
- *matplotlib* - data visualization (3.4.3),
- *sklrearn* - feature vector testing using Linear Support Vector Classification (1.0),
- other built-in libraries, such as: os, math, copy.

# Sample results of the developed algorithm:
The function has been limited to the aspect ratio presented in the article and to the creation of 8x8 histograms.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91888660/136373541-d200e33c-c4a7-45d2-8ed6-e35df971abbe.png" width="200" />
  <img src="https://user-images.githubusercontent.com/91888660/136373544-b0d59181-8255-43ba-9c34-016f861b8bdb.png" width="200" />
  <img src="https://user-images.githubusercontent.com/91888660/136373545-23f38494-db6e-48ad-91d8-cf8ec0337f39.png" height="405"/>
</p>
