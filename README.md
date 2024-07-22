# Image Annotation Tool

## Overview

This project is an image annotation tool built with Python and Tkinter. It allows users to annotate images with rectangles, freehand drawings, and labels. Users can also choose colors for annotations, undo actions, and save both the annotated image and the annotations.

## Features

- **Open Image**: Load an image to annotate.
- **Draw Rectangles**: Annotate objects with rectangles.
- **Freehand Drawing**: Draw freehand annotations on the image.
- **Add Labels**: Add text labels to annotations.
- **Color Selection**: Choose colors for annotations and labels.
- **Undo/Redo**: Revert actions and manage annotations.
- **Zoom and Scroll**: Zoom in/out and scroll the image.
- **Save Annotations**: Save annotations in a text file.
- **Save Image**: Save the annotated image.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/image-annotation-tool.git
   cd image-annotation-tool
Install Dependencies:
Ensure you have Python installed, then install the required libraries:
bash
Copy code
pip install pillow opencv-python numpy
Usage
Run the Application:

bash
Copy code
python annotation_tool.py
Open an Image:
Click "File" -> "Open" to select an image file.

Annotate the Image:

Draw rectangles or freehand shapes.
Add labels using the text entry field and "Add Label" button.
Choose colors for different elements using the color buttons.
Save Annotations:
Click "File" -> "Save Annotations" to save annotation data.

Save Annotated Image:
Click "File" -> "Save Image" to save the image with annotations.

Undo Actions:
Click "File" -> "Undo" to revert the last action.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Tkinter for the GUI
Pillow for image processing
OpenCV for image handling


### Example Content for LICENSE

If you choose the MIT License, your `LICENSE` file should contain:

```plaintext
MIT License

Copyright (c) [year] [your name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
