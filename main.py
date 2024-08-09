import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import os
import json
from lxml import etree
import threading
import yaml
import glob
from functools import reduce
import pandas as pd
from shutil import copy
from pathlib import Path


class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Object Detection Labeling Tool")
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Images", command=self.load_images)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        exportmenu = tk.Menu(menubar, tearoff=0)
        exportmenu.add_command(label="Save Annotations", command=self.save_annotations)
        exportmenu.add_command(label="Export as Pascal VOC", command=self.save_as_voc)
        exportmenu.add_command(label="Export as COCO", command=self.save_as_coco)
        menubar.add_cascade(label="Export", menu=exportmenu)

        trainmenu = tk.Menu(menubar, tearoff=0)
        trainmenu.add_command(label="Train YOLO Model", command=self.train_yolo)
        trainmenu.add_command(label="Train from Folder", command=self.select_folder_for_training)
        menubar.add_cascade(label="Train", menu=trainmenu)

        exportonnxmenu = tk.Menu(menubar, tearoff=0)
        exportonnxmenu.add_command(label="Export to ONNX", command=self.export_onnx)
        menubar.add_cascade(label="Export ONNX", menu=exportonnxmenu)

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

        self.export_onnx_button = ttk.Button(self.sidebar, text="Export to ONNX", command=self.export_onnx)
        self.export_onnx_button.pack(fill=tk.X, pady=5)

        self.status = tk.StringVar()
        self.status.set("Welcome to the Advanced Object Detection Labeling Tool")
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

    def load_images(self):
        try:
            file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
            if file_paths:
                self.image_paths = list(file_paths)
                self.image_index = 0
                self.display_image(self.image_paths[self.image_index])
                self.status.set("Loaded image: " + os.path.basename(self.image_paths[self.image_index]))
                self.log("Loaded image: " + os.path.basename(self.image_paths[self.image_index]))
        except Exception as e:
            self.log(f"Error loading images: {e}")
            messagebox.showerror("Error", f"Error loading images: {e}")

    def display_image(self, file_path):
        try:
            self.img = Image.open(file_path)
            self.scale_factor = min(self.canvas.winfo_width() / self.img.width, self.canvas.winfo_height() / self.img.height)
            self.img_resized = self.img.resize((int(self.img.width * self.scale_factor), int(self.img.height * self.scale_factor)), Image.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(self.img_resized)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.file_path = Path(file_path)
            self.load_annotations()
        except Exception as e:
            self.log(f"Error displaying image: {e}")
            messagebox.showerror("Error", f"Error displaying image: {e}")

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
        for label, bbox in self.annotations:
            draw.rectangle(bbox, outline="blue")
            draw.text((bbox[0], bbox[1]), label, fill="blue")
        self.img_tk = ImageTk.PhotoImage(img_copy)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def show_axis_lines(self, event):
        for line in self.axis_lines:
            self.canvas.delete(line)
        self.axis_lines = [
            self.canvas.create_line(event.x, 0, event.x, self.canvas.winfo_height(), fill='gray', dash=(2, 2)),
            self.canvas.create_line(0, event.y, self.canvas.winfo_width(), event.y, fill='gray', dash=(2, 2))
        ]

    def save_annotation(self):
        try:
            label = simpledialog.askstring("Input", "Enter class label:")
            if label is not None:
                self.annotations.append((label, self.bbox))
                if label not in self.class_list:
                    self.class_list.append(label)
                self.status.set(f"Annotation added: {label} at {self.bbox}")
                self.log(f"Annotation added: {label} at {self.bbox}")
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
                for label, bbox in self.annotations:
                    x_center = (bbox[0] + bbox[2]) / 2 / self.img.width
                    y_center = (bbox[1] + bbox[3]) / 2 / self.img.height
                    width = abs(bbox[0] - bbox[2]) / self.img.width
                    height = abs(bbox[1] - bbox[3]) / self.img.height
                    f.write(f"{self.class_list.index(label)} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            self.log(f"Error saving annotations to file: {e}")
            messagebox.showerror("Error", f"Error saving annotations to file: {e}")

    def load_annotations(self):
        self.annotations = []
        annotation_file = self.file_path.with_suffix('.txt')
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                for line in f.readlines():
                    label, x_center, y_center, width, height = line.strip().split()
                    x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
                    bbox = (x_center * self.img.width - width * self.img.width / 2,
                            y_center * self.img.height - height * self.img.height / 2,
                            x_center * self.img.width + width * self.img.width / 2,
                            y_center * self.img.height + height * self.img.height / 2)
                    self.annotations.append((label, bbox))
                    if label not in self.class_list:
                        self.class_list.append(label)
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

            for label, bbox in self.annotations:
                obj = etree.SubElement(annotation, "object")
                name = etree.SubElement(obj, "name")
                name.text = label
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
            for label, bbox in self.annotations:
                annotations.append({
                    "label": label,
                    "bbox": bbox
                })
            with open(self.file_path.with_suffix('.json'), "w") as f:
                json.dump(annotations, f, indent=4)
            self.status.set("Annotations exported as COCO.")
            self.log("Annotations exported as COCO.")
        except Exception as e:
            self.log(f"Error exporting as COCO: {e}")
            messagebox.showerror("Error", f"Error exporting as COCO: {e}")

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

            # Split images and annotations into train and val sets (80% train, 20% val)
            image_files = list(self.data_dir.glob('*.jpg'))
            annotation_files = list(self.data_dir.glob('*.txt'))
            image_files.sort()
            annotation_files.sort()

            split_index = int(0.8 * len(image_files))
            train_images = image_files[:split_index]
            val_images = image_files[split_index:]
            train_annotations = annotation_files[:split_index]
            val_annotations = annotation_files[split_index:]

            # Copy images and annotations to respective folders
            for image, annotation in zip(train_images, train_annotations):
                copy(image, train_folder / image.name)
                copy(annotation, train_folder / annotation.name)

            for image, annotation in zip(val_images, val_annotations):
                copy(image, val_folder / image.name)
                copy(annotation, val_folder / annotation.name)

            self.log(f"Dataset prepared: {len(train_images)} train images, {len(val_images)} val images")
        except Exception as e:
            self.log(f"Error preparing dataset: {e}")
            messagebox.showerror("Error", f"Error preparing dataset: {e}")

    def run_training(self, epochs, batch_size):
        try:
            # Training command placeholder
            self.log(f"Training for {epochs} epochs with batch size {batch_size}. Training command should be implemented here.")
            self.status.set("Training completed.")
            self.log("Training completed.")
        except Exception as e:
            self.log(f"Error during training: {e}")
            messagebox.showerror("Error", f"Error during training: {e}")

    def export_onnx(self):
        try:
            # ONNX export command placeholder
            self.log("ONNX export logic should be implemented here.")
            self.status.set("Model exported to ONNX.")
            self.log("Model exported to ONNX.")
        except Exception as e:
            self.log(f"Error exporting to ONNX: {e}")
            messagebox.showerror("Error", f"Error exporting to ONNX: {e}")

    def select_folder_for_training(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.log(f"Selected folder for training: {folder_path}")
            self.train_yolo_from_folder(folder_path)

    def train_yolo_from_folder(self, folder_path):
        try:
            epochs = simpledialog.askinteger("Input", "Enter number of epochs:")
            batch_size = simpledialog.askinteger("Input", "Enter batch size:")

            # Create YAML configuration for YOLO training
            data_config = {
                'train': folder_path,
                'val': folder_path,
                'nc': len(self.class_list),
                'names': self.class_list
            }

            with open('data.yaml', 'w') as f:
                yaml.dump(data_config, f)

            self.status.set("Training started with selected folder.")
            self.log("Training started with selected folder.")

            # Run training in a separate thread to keep the GUI responsive
            threading.Thread(target=self.run_training, args=(epochs, batch_size)).start()
        except Exception as e:
            self.log(f"Error starting training from folder: {e}")
            messagebox.showerror("Error", f"Error starting training from folder: {e}")

    def select_bbox(self, event):
        for idx, (label, bbox) in enumerate(self.annotations):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self.selected_bbox = idx
                self.start_x, self.start_y = event.x, event.y
                break
        self.update_image()

    def move_bbox(self, dx, dy):
        if self.selected_bbox is not None:
            label, bbox = self.annotations[self.selected_bbox]
            new_bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
            self.annotations[self.selected_bbox] = (label, new_bbox)
            self.update_image()

    def resize_bbox(self, dx, dy):
        if self.selected_bbox is not None:
            label, bbox = self.annotations[self.selected_bbox]
            new_bbox = (bbox[0], bbox[1], bbox[2] + dx, bbox[3] + dy)
            self.annotations[self.selected_bbox] = (label, new_bbox)
            self.update_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()
