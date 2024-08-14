import tkinter as tk
import cv2
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageTk, ImageDraw
import os
import json
from lxml import etree
import threading
import yaml
import subprocess
from shutil import copy
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm  # For progress bar

class YOLOv5Tool:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced YOLOv5 Training and Inference Tool")
        self.root.geometry("1200x800")

        self.create_menu()
        self.create_widgets()
        self.class_list = []
        self.annotations = []
        self.bbox = None
        self.selected_bbox = None
        self.start_x, self.start_y = None, None
        self.scale_factor = 1.0
        self.axis_lines = []

        self.data_dir = Path('./data_images')
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.inference_dir = Path('./inference_images')
        self.output_dir = Path('./output')
        self.model_dir = Path('./models')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_config = 'yolov5s.yaml'  # Default configuration
        self.confidence_threshold = 0.25  # Default confidence threshold for inference
        self.iou_threshold = 0.45  # Default IoU threshold for inference

    def create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Images", command=self.load_images)
        filemenu.add_command(label="Load Inference Images", command=self.load_inference_images)
        filemenu.add_command(label="Import Annotations", command=self.import_annotations)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        exportmenu = tk.Menu(menubar, tearoff=0)
        exportmenu.add_command(label="Save Annotations", command=self.save_annotations)
        exportmenu.add_command(label="Export as Pascal VOC", command=self.save_as_voc)
        exportmenu.add_command(label="Export as COCO", command=self.save_as_coco)
        menubar.add_cascade(label="Export", menu=exportmenu)

        trainmenu = tk.Menu(menubar, tearoff=0)
        trainmenu.add_command(label="Select YOLO Model", command=self.select_yolo_model)
        trainmenu.add_command(label="Train YOLO Model", command=self.train_yolo)
        trainmenu.add_command(label="Validate YOLO Model", command=self.validate_yolo)
        trainmenu.add_command(label="Load Model", command=self.load_model)
        menubar.add_cascade(label="Train/Validate", menu=trainmenu)

        inferencemenu = tk.Menu(menubar, tearoff=0)
        inferencemenu.add_command(label="Run Inference", command=self.run_inference)
        inferencemenu.add_command(label="Advanced Inference Options", command=self.advanced_inference_options)
        menubar.add_cascade(label="Inference", menu=inferencemenu)

        exportonnxmenu = tk.Menu(menubar, tearoff=0)
        exportonnxmenu.add_command(label="Export to ONNX", command=self.export_onnx)
        menubar.add_cascade(label="Export ONNX", menu=exportonnxmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="How to Use", command=self.show_help)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.frame, padding=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        self.load_button = ttk.Button(self.sidebar, text="Load Images", command=self.load_images)
        self.load_button.pack(fill=tk.X, pady=5)

        self.save_button = ttk.Button(self.sidebar, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(fill=tk.X, pady=5)

        self.export_voc_button = ttk.Button(self.sidebar, text="Export as Pascal VOC", command=self.save_as_voc)
        self.export_voc_button.pack(fill=tk.X, pady=5)

        self.export_coco_button = ttk.Button(self.sidebar, text="Export as COCO", command=self.save_as_coco)
        self.export_coco_button.pack(fill=tk.X, pady=5)

        self.train_button = ttk.Button(self.sidebar, text="Train YOLO Model", command=self.train_yolo)
        self.train_button.pack(fill=tk.X, pady=5)

        self.validate_button = ttk.Button(self.sidebar, text="Validate YOLO Model", command=self.validate_yolo)
        self.validate_button.pack(fill=tk.X, pady=5)

        self.inference_button = ttk.Button(self.sidebar, text="Run Inference", command=self.run_inference)
        self.inference_button.pack(fill=tk.X, pady=5)

        self.export_onnx_button = ttk.Button(self.sidebar, text="Export to ONNX", command=self.export_onnx)
        self.export_onnx_button.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(self.sidebar, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        self.status = tk.StringVar()
        self.status.set("Welcome to YOLOv5 Training and Inference Tool")
        self.statusbar = ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.log_box = tk.Text(self.root, height=10, state='disabled', bg='lightgray')
        self.log_box.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.select_bbox)
        self.canvas.bind("<Motion>", self.show_axis_lines)

    def log(self, message):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, message + '\n')
        self.log_box.config(state='disabled')
        self.log_box.see(tk.END)

    def show_help(self):
        help_text = (
            "YOLOv5 Training and Inference Tool:\n\n"
            "1. Load Images: Load images for annotation and model training.\n"
            "2. Save Annotations: Save the bounding box annotations.\n"
            "3. Export Annotations: Export annotations in Pascal VOC or COCO format.\n"
            "4. Select YOLO Model: Choose a YOLO model configuration (e.g., yolov5s, yolov5m).\n"
            "5. Train YOLO Model: Train the YOLO model with the annotated images.\n"
            "6. Validate YOLO Model: Validate the model on a separate validation dataset.\n"
            "7. Run Inference: Perform inference on new images.\n"
            "8. Export to ONNX: Export the trained model to ONNX format.\n"
        )
        messagebox.showinfo("How to Use", help_text)

    def load_images(self):
        try:
            file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if file_paths:
                self.image_paths = list(file_paths)
                self.image_index = 0
                self.preprocess_and_display_image(self.image_paths[self.image_index])
                self.status.set("Loaded image: " + os.path.basename(self.image_paths[self.image_index]))
                self.log("Loaded image: " + os.path.basename(self.image_paths[self.image_index]))
        except Exception as e:
            self.log(f"Error loading images: {e}")
            messagebox.showerror("Error", f"Error loading images: {e}")

    def load_inference_images(self):
        try:
            file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if file_paths:
                self.inference_image_paths = list(file_paths)
                self.log(f"Loaded {len(self.inference_image_paths)} images for inference.")
                self.status.set(f"Loaded {len(self.inference_image_paths)} images for inference.")
        except Exception as e:
            self.log(f"Error loading inference images: {e}")
            messagebox.showerror("Error", f"Error loading inference images: {e}")

    def import_annotations(self):
        try:
            annotation_file = filedialog.askopenfilename(filetypes=[("Annotation files", "*.txt;*.xml;*.json")])
            if annotation_file:
                self.load_annotations(annotation_file)
                self.log(f"Imported annotations from {annotation_file}.")
                self.status.set(f"Imported annotations from {annotation_file}.")
        except Exception as e:
            self.log(f"Error importing annotations: {e}")
            messagebox.showerror("Error", f"Error importing annotations: {e}")

    def preprocess_image(self, image_path):
        # Load image
        img = Image.open(image_path)
    
        # Ensure image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
        # Convert PIL image to OpenCV format for advanced preprocessing
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
        # Resize image while maintaining aspect ratio, with padding if necessary
        desired_size = 640
        old_size = img_cv.shape[:2]  # old_size is in (height, width) format
    
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
    
        img_cv = cv2.resize(img_cv, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
    
        color = [128, 128, 128]
        img_cv = cv2.copyMakeBorder(img_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
        # Mosaic Augmentation (advanced, optional)
        if np.random.rand() > 0.5:
            img_cv = self.mosaic_augmentation(img_cv)
    
        # Random Cutout (data augmentation)
        if np.random.rand() > 0.5:
            img_cv = self.cutout_augmentation(img_cv)
    
        # Random flipping (data augmentation)
        if np.random.rand() > 0.5:
            img_cv = cv2.flip(img_cv, 1)  # Horizontal flip
    
        # Random rotation (data augmentation)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((desired_size / 2, desired_size / 2), angle, 1)
            img_cv = cv2.warpAffine(img_cv, M, (desired_size, desired_size), borderValue=color)
    
        # AutoAugment/RandAugment (optional, dynamic data augmentation)
        if np.random.rand() > 0.5:
            img_cv = self.apply_autoaugment(img_cv)
    
        # Random brightness and contrast adjustment (data augmentation)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast control
            beta = np.random.uniform(-10, 10)    # Brightness control
            img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
    
        # Random Gaussian blur (optional, data augmentation)
        if np.random.rand() > 0.5:
            ksize = np.random.choice([3, 5])  # Kernel size
            img_cv = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)
    
        # Ensure pixel values are within [0, 255]
        img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
    
        # Convert back to PIL Image for compatibility with GUI
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
        return img
    
    def mosaic_augmentation(self, img, dataset=None):
        try:
            # If no dataset is provided, create one with the current image duplicated
            if dataset is None:
                dataset = [img] * 4  # Create a list of 4 copies of the image
            
            # Rest of the mosaic augmentation code...
            h, w = img.shape[:2]
            target_size = (w, h)
    
            # Randomly select 3 additional images from the dataset
            imgs = [img] + random.sample(dataset, 3)
            
            # Resize all images to the target size
            imgs_resized = [cv2.resize(i, target_size, interpolation=cv2.INTER_LINEAR) for i in imgs]
    
            # Combine images into a mosaic
            top = np.hstack((imgs_resized[0], imgs_resized[1]))
            bottom = np.hstack((imgs_resized[2], imgs_resized[3]))
            mosaic_img = np.vstack((top, bottom))
    
            # Final resize to fit the desired output size
            mosaic_img = cv2.resize(mosaic_img, (w, h))
    
            return mosaic_img
        except Exception as e:
            print(f"Error in mosaic_augmentation: {e}")
            return img  # Return the original image if there's an error

    
            return mosaic_img
        except Exception as e:
            print(f"Error in mosaic_augmentation: {e}")
            return img  # Return the original image if there's an error
    
    def cutout_augmentation(self, img):
        try:
            h, w, _ = img.shape
    
            # Apply multiple random cutouts
            for _ in range(np.random.randint(1, 5)):  # Apply 1 to 5 cutouts
                mask_size = np.random.randint(h // 8, h // 4)  # Mask size between 1/8th and 1/4th of the image height
                mask_x = np.random.randint(0, w)
                mask_y = np.random.randint(0, h)
    
                x1 = max(0, mask_x - mask_size // 2)
                y1 = max(0, mask_y - mask_size // 2)
                x2 = min(w, mask_x + mask_size // 2)
                y2 = min(h, mask_y + mask_size // 2)
    
                img[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1, 3))  # Fill with random colors
    
            return img
        except Exception as e:
            print(f"Error in cutout_augmentation: {e}")
            return img  # Return the original image if there's an error
    
    def apply_autoaugment(self, img):
        try:
            augmentations = [
                lambda x: cv2.flip(x, 1),  # Horizontal flip
                lambda x: cv2.GaussianBlur(x, (3, 3), 0),  # Gaussian blur
                lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR),  # HSV shift
                lambda x: cv2.addWeighted(x, 4, cv2.GaussianBlur(x, (0, 0), 10), -4, 128),  # Unsharp masking
                lambda x: cv2.add(x, np.random.uniform(-10, 10, x.shape)),  # Random brightness
                lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),  # Rotate 90 degrees
            ]
    
            # Apply a random number of augmentations (1-3)
            for aug in random.sample(augmentations, k=random.randint(1, 3)):
                img = aug(img)
    
            # Ensure pixel values are within [0, 255]
            img = np.clip(img, 0, 255).astype(np.uint8)
    
            return img
        except Exception as e:
            print(f"Error in apply_autoaugment: {e}")
            return img  # Return the original image if there's an error

    def preprocess_and_display_image(self, file_path):
        try:
            # Log the start of the process
            self.log(f"Starting image preprocessing for: {file_path}")
    
            # Preprocess the image
            self.img = self.preprocess_image(file_path)
    
            # Get the target display size while maintaining aspect ratio
            display_width = self.canvas.winfo_width()
            display_height = self.canvas.winfo_height()
    
            # Calculate the scale factor while maintaining aspect ratio
            scale_factor = min(display_width / self.img.width, display_height / self.img.height)
    
            # Resize the image with proper resampling method for high quality
            new_width = int(self.img.width * scale_factor)
            new_height = int(self.img.height * scale_factor)
            self.img_resized = self.img.resize((new_width, new_height), Image.LANCZOS)
    
            # Check if padding is needed to maintain aspect ratio and center the image
            pad_x = (display_width - new_width) // 2
            pad_y = (display_height - new_height) // 2
    
            # Create a blank image with padding if necessary
            img_padded = Image.new('RGB', (display_width, display_height), (128, 128, 128))
            img_padded.paste(self.img_resized, (pad_x, pad_y))
    
            # Convert to Tkinter-compatible format
            self.img_tk = ImageTk.PhotoImage(img_padded)
    
            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
    
            # Store the file path for future reference
            self.file_path = Path(file_path)
    
            # Log successful processing
            self.log(f"Image successfully preprocessed and displayed: {file_path}")
    
        except FileNotFoundError:
            self.log(f"File not found: {file_path}")
            messagebox.showerror("Error", f"File not found: {file_path}")
        except IOError:
            self.log(f"IO error while processing the file: {file_path}")
            messagebox.showerror("Error", f"IO error while processing the file: {file_path}")
        except Exception as e:
            self.log(f"Unexpected error displaying image: {str(e)}")
            messagebox.showerror("Error", f"Unexpected error displaying image: {str(e)}")


    def on_click(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.selected_bbox:
            self.selected_bbox = None
            self.update_image()

    def on_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        if self.start_x and self.start_y and not self.selected_bbox:
            self.update_image()
        elif self.selected_bbox:
            self.move_bbox(event.x - self.start_x, event.y - self.start_y)
            self.start_x, self.start_y = event.x, event.y

    def on_release(self, event):
        if not self.selected_bbox:
            self.end_x, self.end_y = event.x, event.y
            self.bbox = (self.start_x, self.start_y, self.end_x, self.end_y)
            self.save_annotation()
        self.selected_bbox = None

    def update_image(self):
        img_copy = self.img_resized.copy()
        draw = ImageDraw.Draw(img_copy)
        if self.start_x and self.start_y and self.end_x and self.end_y:
            draw.rectangle((self.start_x, self.start_y, self.end_x, self.end_y), outline="red")
        for label_id, bbox in self.annotations:
            draw.rectangle(bbox, outline="blue")
            label = self.class_list[int(label_id)]
            draw.text((bbox[0], bbox[1]), label, fill="blue")
        self.img_tk = ImageTk.PhotoImage(img_copy)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def show_axis_lines(self, event):
        for line in self.axis_lines:
            self.canvas.delete(line)
        self.axis_lines = [
            self.canvas.create_line(event.x, 0, event.x, self.canvas.winfo_height(), fill='#006400', dash=(2, 2)),  # Deep green color
            self.canvas.create_line(0, event.y, self.canvas.winfo_width(), event.y, fill='#006400', dash=(2, 2))   # Deep green color
        ]

    

    def save_annotation(self):
        try:
            label = simpledialog.askstring("Input", "Enter class label:")
            if label is not None:
                if label not in self.class_list:
                    self.class_list.append(label)
                    self.log(f"New label '{label}' assigned ID {self.class_list.index(label)}")

                label_id = self.class_list.index(label)
                self.annotations.append((label_id, self.bbox))

                self.status.set(f"Annotation added: {label} (ID: {label_id}) at {self.bbox}")
                self.log(f"Annotation added: {label} (ID: {label_id}) at {self.bbox}")
                self.save_annotations_to_file()
        except Exception as e:
            self.log(f"Error saving annotation: {e}")
            messagebox.showerror("Error", f"Error saving annotation: {e}")

    def save_annotations(self):
        try:
            self.save_annotations_to_file()
            self.status.set("Annotations saved.")
            self.log("Annotations saved.")
        except Exception as e:
            self.log(f"Error saving annotations: {e}")
            messagebox.showerror("Error", f"Error saving annotations: {e}")

    def save_annotations_to_file(self):
        try:
            annotation_path = self.file_path.with_suffix('.txt')
            with open(annotation_path, 'w') as f:
                for label_id, bbox in self.annotations:
                    x_center = (bbox[0] + bbox[2]) / 2 / self.img.width
                    y_center = (bbox[1] + bbox[3]) / 2 / self.img.height
                    width = abs(bbox[0] - bbox[2]) / self.img.width
                    height = abs(bbox[1] - bbox[3]) / self.img.height
                    f.write(f"{label_id} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            self.log(f"Error saving annotations to file: {e}")
            messagebox.showerror("Error", f"Error saving annotations to file: {e}")

    def load_annotations(self, annotation_file=None):
        self.annotations = []
        if annotation_file:
            ext = Path(annotation_file).suffix
            if ext == ".txt":
                with open(annotation_file, "r") as f:
                    for line in f.readlines():
                        label_id, x_center, y_center, width, height = line.strip().split()
                        x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
                        bbox = (x_center * self.img.width - width * self.img.width / 2,
                                y_center * self.img.height - height * self.img.height / 2,
                                x_center * self.img.width + width * self.img.width / 2,
                                y_center * self.img.height + height * self.img.height / 2)
                        self.annotations.append((label_id, bbox))
            elif ext == ".xml":
                # Add XML parsing code here (Pascal VOC format)
                pass
            elif ext == ".json":
                # Add JSON parsing code here (COCO format)
                pass
            self.update_image()
            self.log("Annotations loaded.")

    def save_as_voc(self):
        try:
            annotation = etree.Element("annotation")
            folder = etree.SubElement(annotation, "folder")
            folder.text = "VOC2012"
            filename_elem = etree.SubElement(annotation, "filename")
            filename_elem.text = os.path.basename(self.file_path)

            source = etree.SubElement(annotation, "source")
            database = etree.SubElement(source, "database")
            database.text = "The VOC2007 Database"
            annotation_text = etree.SubElement(source, "annotation")
            annotation_text.text = "PASCAL VOC2007"
            image = etree.SubElement(source, "image")
            image.text = "flickr"

            size = etree.SubElement(annotation, "size")
            width_elem = etree.SubElement(size, "width")
            width_elem.text = str(self.img.width)
            height_elem = etree.SubElement(size, "height")
            height_elem.text = str(self.img.height)
            depth_elem = etree.SubElement(size, "depth")
            depth_elem.text = "3"

            segmented = etree.SubElement(annotation, "segmented")
            segmented.text = "0"

            for label_id, bbox in self.annotations:
                obj = etree.SubElement(annotation, "object")
                name = etree.SubElement(obj, "name")
                name.text = self.class_list[int(label_id)]
                pose = etree.SubElement(obj, "pose")
                pose.text = "Unspecified"
                truncated = etree.SubElement(obj, "truncated")
                truncated.text = "0"
                difficult = etree.SubElement(obj, "difficult")
                difficult.text = "0"
                bndbox = etree.SubElement(obj, "bndbox")
                xmin = etree.SubElement(bndbox, "xmin")
                ymin = etree.SubElement(bndbox, "ymin")
                xmax = etree.SubElement(bndbox, "xmax")
                ymax = etree.SubElement(bndbox, "ymax")
                xmin.text = str(int(bbox[0]))
                ymin.text = str(int(bbox[1]))
                xmax.text = str(int(bbox[2]))
                ymax.text = str(int(bbox[3]))

            tree = etree.ElementTree(annotation)
            tree.write(self.file_path.with_suffix('.xml'), pretty_print=True)
            self.status.set("Annotations exported as Pascal VOC.")
            self.log("Annotations exported as Pascal VOC.")
        except Exception as e:
            self.log(f"Error exporting as Pascal VOC: {e}")
            messagebox.showerror("Error", f"Error exporting as Pascal VOC: {e}")

    def save_as_coco(self):
        try:
            annotations = []
            for label_id, bbox in self.annotations:
                annotations.append({
                    "label_id": label_id,
                    "bbox": bbox
                })
            with open(self.file_path.with_suffix('.json'), "w") as f:
                json.dump(annotations, f, indent=4)
            self.status.set("Annotations exported as COCO.")
            self.log("Annotations exported as COCO.")
        except Exception as e:
            self.log(f"Error exporting as COCO: {e}")
            messagebox.showerror("Error", f"Error exporting as COCO: {e}")

    def select_yolo_model(self):
        model_choices = ["yolov5s.yaml", "yolov5m.yaml", "yolov5l.yaml", "yolov5x.yaml"]
        self.yolo_config = simpledialog.askstring("Select YOLO Model", f"Choose YOLO model:\n{model_choices}",
                                                  initialvalue=self.yolo_config)
        if self.yolo_config not in model_choices:
            self.yolo_config = "yolov5s.yaml"  # Default to yolov5s if an invalid choice is made
        self.log(f"Selected YOLO model: {self.yolo_config}")

    def train_yolo(self):
        try:
            epochs = simpledialog.askinteger("Input", "Enter number of epochs:")
            batch_size = simpledialog.askinteger("Input", "Enter batch size:")

            # Prepare dataset for YOLO training
            self.prepare_dataset()

            # Create YAML configuration for YOLO training
            data_config = {
                'train': str(self.train_dir),
                'val': str(self.val_dir),
                'nc': len(self.class_list),
                'names': self.class_list
            }

            with open(self.data_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_config, f)

            self.status.set("Training started.")
            self.log("Training started.")
            self.progress.start()

            # Run training in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_training, args=(epochs, batch_size)).start()
        except Exception as e:
            self.log(f"Error starting training: {e}")
            messagebox.showerror("Error", f"Error starting training: {e}")

    def prepare_dataset(self):
        try:
            train_folder = self.train_dir
            val_folder = self.val_dir
    
            # Clear train and val folders if they already exist
            for folder in [train_folder, val_folder]:
                if folder.exists():
                    for file in folder.iterdir():
                        file.unlink()
    
            # Support multiple image formats
            image_files = list(self.data_dir.glob('*.jpg')) + \
                          list(self.data_dir.glob('*.jpeg')) + \
                          list(self.data_dir.glob('*.png'))
            annotation_files = list(self.data_dir.glob('*.txt'))
    
            # Ensure images and annotations are paired correctly
            paired_files = [(img, img.with_suffix('.txt')) for img in image_files if img.with_suffix('.txt') in annotation_files]
    
            if not paired_files:
                raise ValueError("No matching image-annotation pairs found in the dataset.")
    
            # Shuffle the paired files before splitting
            random.shuffle(paired_files)
    
            # Split images and annotations into train and val sets (80% train, 20% val)
            split_index = int(0.8 * len(paired_files))
            train_pairs = paired_files[:split_index]
            val_pairs = paired_files[split_index:]
    
            # Copy images and annotations to respective folders with progress bar
            self.log("Copying training data...")
            for image, annotation in tqdm(train_pairs, desc="Training data"):
                shutil.copy(image, train_folder / image.name)
                shutil.copy(annotation, train_folder / annotation.name)
    
            self.log("Copying validation data...")
            for image, annotation in tqdm(val_pairs, desc="Validation data"):
                shutil.copy(image, val_folder / image.name)
                shutil.copy(annotation, val_folder / annotation.name)
    
            self.log(f"Dataset prepared: {len(train_pairs)} train images, {len(val_pairs)} val images")
    
        except ValueError as ve:
            self.log(f"Dataset preparation error: {ve}")
            messagebox.showerror("Error", f"Dataset preparation error: {ve}")
        except Exception as e:
            self.log(f"Error preparing dataset: {e}")
            messagebox.showerror("Error", f"Error preparing dataset: {e}")


    def run_training(self, epochs, batch_size):
        try:
            # Actual YOLO training command
            command = [
                "python", "train.py",
                "--data", str(self.data_dir / 'data.yaml'),
                "--cfg", self.yolo_config,  # Selected YOLO model configuration
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--name", "custom_yolov5",  # Name of the training run
                "--project", str(self.model_dir)  # Save the model in the models directory
            ]
            subprocess.run(command, check=True)
            self.status.set("Training completed.")
            self.log("Training completed.")
            self.progress.stop()
        except subprocess.CalledProcessError as e:
            self.log(f"Error during training: {e}")
            messagebox.showerror("Error", f"Error during training: {e}")
            self.progress.stop()

    def validate_yolo(self):
        try:
            # Validate YOLO model
            command = [
                "python", "val.py",
                "--weights", str(self.model_dir / "custom_yolov5/weights/best.pt"),  # Path to the best model weights
                "--data", str(self.data_dir / 'data.yaml'),
                "--img", "640"  # Image size
            ]
            self.status.set("Validation started.")
            self.log("Validation started.")
            self.progress.start()
            subprocess.run(command, check=True)
            self.status.set("Validation completed.")
            self.log("Validation completed.")
            self.progress.stop()
        except subprocess.CalledProcessError as e:
            self.log(f"Error during validation: {e}")
            messagebox.showerror("Error", f"Error during validation: {e}")
            self.progress.stop()

    def load_model(self):
        try:
            model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
            if model_path:
                self.selected_model_path = model_path
                self.log(f"Loaded model from {model_path}")
                self.status.set(f"Loaded model from {model_path}")
        except Exception as e:
            self.log(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Error loading model: {e}")

    def run_inference(self):
        try:
            if not hasattr(self, 'inference_image_paths') or not self.inference_image_paths:
                messagebox.showwarning("No Images", "Please load images for inference.")
                return

            if not hasattr(self, 'selected_model_path'):
                self.selected_model_path = str(self.model_dir / "custom_yolov5/weights/best.pt")

            # Inference command
            command = [
                "python", "detect.py",
                "--weights", self.selected_model_path,  # Path to the selected model weights
                "--source", str(self.inference_dir),  # Directory with images for inference
                "--conf", str(self.confidence_threshold),  # Confidence threshold
                "--iou", str(self.iou_threshold),  # IoU threshold
                "--img", "640",  # Image size
                "--save-txt", "--save-conf",  # Save results to text files and include confidence scores
                "--project", str(self.output_dir)  # Save results in the output directory
            ]

            # Copy images to inference directory
            for img_path in self.inference_image_paths:
                copy(img_path, self.inference_dir / Path(img_path).name)

            self.status.set("Inference started.")
            self.log("Inference started.")
            self.progress.start()
            subprocess.run(command, check=True)
            self.status.set("Inference completed. Check the output directory for results.")
            self.log("Inference completed. Check the output directory for results.")
            self.progress.stop()
        except subprocess.CalledProcessError as e:
            self.log(f"Error during inference: {e}")
            messagebox.showerror("Error", f"Error during inference: {e}")
            self.progress.stop()

    def advanced_inference_options(self):
        try:
            self.confidence_threshold = simpledialog.askfloat("Confidence Threshold", "Enter confidence threshold:",
                                                              initialvalue=self.confidence_threshold)
            self.iou_threshold = simpledialog.askfloat("IoU Threshold", "Enter IoU threshold:",
                                                       initialvalue=self.iou_threshold)
            self.log(f"Set confidence threshold to {self.confidence_threshold} and IoU threshold to {self.iou_threshold}")
        except Exception as e:
            self.log(f"Error setting advanced inference options: {e}")
            messagebox.showerror("Error", f"Error setting advanced inference options: {e}")

    def export_onnx(self):
        try:
            # ONNX export command
            command = [
                "python", "export.py",
                "--weights", str(self.model_dir / "custom_yolov5/weights/best.pt"),  # Path to the best model weights
                "--img", "640",  # Image size
                "--batch", "1",  # Batch size
                "--device", "cpu",  # Exporting on CPU
                "--include", "onnx"  # Export to ONNX format
            ]
            subprocess.run(command, check=True)
            self.status.set("Model exported to ONNX.")
            self.log("Model exported to ONNX.")
        except subprocess.CalledProcessError as e:
            self.log(f"Error exporting to ONNX: {e}")
            messagebox.showerror("Error", f"Error exporting to ONNX: {e}")

    def select_bbox(self, event):
        for idx, (label_id, bbox) in enumerate(self.annotations):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self.selected_bbox = idx
                self.start_x, self.start_y = event.x, event.y
                break
        self.update_image()

    def move_bbox(self, dx, dy):
        if self.selected_bbox is not None:
            label_id, bbox = self.annotations[self.selected_bbox]
            new_bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
            self.annotations[self.selected_bbox] = (label_id, new_bbox)
            self.update_image()

    def resize_bbox(self, dx, dy):
        if self.selected_bbox is not None:
            label_id, bbox = self.annotations[self.selected_bbox]
            new_bbox = (bbox[0], bbox[1], bbox[2] + dx, bbox[3] + dy)
            self.annotations[self.selected_bbox] = (label_id, new_bbox)
            self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv5Tool(root)
    root.mainloop()
