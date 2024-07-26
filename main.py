import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, simpledialog
from PIL import Image, ImageTk, ImageDraw
import json
import os
import threading

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(root, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.scroll_y = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<ButtonPress-3>", self.start_move_image)
        self.canvas.bind("<B3-Motion>", self.move_image)
        self.canvas.bind("<ButtonRelease-3>", self.end_move_image)
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.polygon_points = []
        self.drawing = False
        self.freehand_id = None
        self.image = None
        self.tk_image = None
        self.original_image = None
        self.image_path = None
        self.labels = []
        self.freehand_labels = []
        self.font_color = "black"
        self.rect_color = "red"
        self.polygon_color = "green"
        self.circle_color = "yellow"
        self.keypoint_color = "purple"
        self.scale = 1.0

        self.actions = []
        self.redo_actions = []
        self.annotation_mode = "rectangle"  # Default annotation mode

        # Autosave
        self.autosave_interval = 300  # 5 minutes
        self.autosave_path = "autosave.json"
        self.start_autosave()

        # Create menus
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        self.file_menu.add_command(label="Load Annotations", command=self.load_annotations)
        self.file_menu.add_command(label="Save Project", command=self.save_project)
        self.file_menu.add_command(label="Load Project", command=self.load_project)
        self.file_menu.add_command(label="Save Image", command=self.save_image)
        self.file_menu.add_command(label="Exit", command=root.quit)

        # Annotation mode menu
        self.mode_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Mode", menu=self.mode_menu)
        self.mode_menu.add_command(label="Rectangle", command=lambda: self.set_annotation_mode("rectangle"))
        self.mode_menu.add_command(label="Polygon", command=lambda: self.set_annotation_mode("polygon"))
        self.mode_menu.add_command(label="Circle", command=lambda: self.set_annotation_mode("circle"))
        self.mode_menu.add_command(label="Keypoints", command=lambda: self.set_annotation_mode("keypoints"))

        # Image preprocessing menu
        self.preprocess_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Preprocess", menu=self.preprocess_menu)
        self.preprocess_menu.add_command(label="Rotate 90°", command=lambda: self.augment_image("rotate_90"))
        self.preprocess_menu.add_command(label="Rotate 180°", command=lambda: self.augment_image("rotate_180"))
        self.preprocess_menu.add_command(label="Flip Horizontal", command=lambda: self.augment_image("flip_horizontal"))
        self.preprocess_menu.add_command(label="Flip Vertical", command=lambda: self.augment_image("flip_vertical"))
        self.preprocess_menu.add_command(label="Scale 1.5x", command=lambda: self.augment_image("scale_1.5"))
        self.preprocess_menu.add_command(label="Scale 0.5x", command=lambda: self.augment_image("scale_0.5"))
        self.preprocess_menu.add_command(label="Crop", command=self.crop_image)
        self.preprocess_menu.add_command(label="Resize", command=self.resize_image)

        # Color selection buttons
        self.color_frame = tk.Frame(root)
        self.color_frame.pack(side=tk.BOTTOM, pady=5)
        self.font_color_button = tk.Button(self.color_frame, text="Font Color", command=self.choose_font_color)
        self.font_color_button.pack(side=tk.LEFT, padx=5)
        self.rect_color_button = tk.Button(self.color_frame, text="Rect Color", command=self.choose_rect_color)
        self.rect_color_button.pack(side=tk.LEFT, padx=5)
        self.polygon_color_button = tk.Button(self.color_frame, text="Polygon Color", command=self.choose_polygon_color)
        self.polygon_color_button.pack(side=tk.LEFT, padx=5)
        self.circle_color_button = tk.Button(self.color_frame, text="Circle Color", command=self.choose_circle_color)
        self.circle_color_button.pack(side=tk.LEFT, padx=5)
        self.keypoint_color_button = tk.Button(self.color_frame, text="Keypoint Color", command=self.choose_keypoint_color)
        self.keypoint_color_button.pack(side=tk.LEFT, padx=5)

        # Label entry and add button
        self.label_entry = tk.Entry(root, width=20)
        self.label_entry.pack(side=tk.BOTTOM, pady=5)
        self.label_button = tk.Button(root, text="Add Label", command=self.add_label)
        self.label_button.pack(side=tk.BOTTOM, pady=5)

        # Toggle for freehand drawing
        self.freehand_mode = False
        self.freehand_button = tk.Button(root, text="Toggle Freehand Mode", command=self.toggle_freehand_mode)
        self.freehand_button.pack(side=tk.BOTTOM, pady=5)

        # Store freehand drawings separately
        self.freehand_drawings = []

        # Actions listbox
        self.action_frame = tk.Frame(root)
        self.action_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.action_listbox = tk.Listbox(self.action_frame, height=20)
        self.action_listbox.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)
        self.action_listbox.insert(tk.END, "Actions:")

        # Annotation list
        self.annotation_listbox = tk.Listbox(self.action_frame, height=20)
        self.annotation_listbox.pack(side=tk.TOP, fill=tk.Y, padx=5, pady=5)
        self.annotation_listbox.insert(tk.END, "Annotations:")

        # Variables for moving the image
        self.image_id = None
        self.move_image_start_x = None
        self.move_image_start_y = None

    def log_action(self, action):
        self.actions.append(action)
        self.redo_actions.clear()
        self.action_listbox.insert(tk.END, action)

    def undo(self, event=None):
        if not self.actions:
            return
        action = self.actions.pop()
        self.redo_actions.append(action)
        self.action_listbox.insert(tk.END, f"Undo: {action}")
        # Implement specific undo logic based on action type
        # ...

    def redo(self, event=None):
        if not self.redo_actions:
            return
        action = self.redo_actions.pop()
        self.actions.append(action)
        self.action_listbox.insert(tk.END, f"Redo: {action}")
        # Implement specific redo logic based on action type
        # ...

    def start_autosave(self):
        self.autosave()
        self.root.after(self.autosave_interval * 1000, self.start_autosave)

    def autosave(self):
        if not self.image:
            return
        project = {
            "image": self.image_path,
            "annotations": []
        }
        for rect_id, label in self.labels:
            coords = self.canvas.coords(rect_id)
            project["annotations"].append({
                'label': label,
                'bbox': (coords[0], coords[1], coords[2], coords[3])
            })
        for line_id, label in self.freehand_labels:
            coords = self.canvas.coords(line_id)
            project["annotations"].append({
                'label': label,
                'freehand': coords
            })
        with open(self.autosave_path, "w") as f:
            json.dump(project, f, indent=4)
        self.log_action("Autosaved project")

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
        self.image_path = file_path
        self.image = Image.open(file_path)
        self.original_image = self.image.copy()  # Keep a copy of the original image

        self.fit_image_to_canvas()

    def fit_image_to_canvas(self):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Resize the image to fit within the canvas while maintaining the aspect ratio
        image_width, image_height = self.image.size
        aspect_ratio = image_width / image_height

        if image_width > canvas_width or image_height > canvas_height:
            if aspect_ratio > 1:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(self.image)
        if self.image_id:
            self.canvas.delete(self.image_id)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.rect = None
        self.labels = []
        self.freehand_drawings = []
        self.freehand_labels = []  # Clear previous freehand labels
        self.log_action("Opened image")

    def start_move_image(self, event):
        self.move_image_start_x = event.x
        self.move_image_start_y = event.y

    def move_image(self, event):
        if not self.image_id:
            return
        dx = event.x - self.move_image_start_x
        dy = event.y - self.move_image_start_y
        self.canvas.move(self.image_id, dx, dy)
        self.move_image_start_x = event.x
        self.move_image_start_y = event.y

    def end_move_image(self, event):
        self.move_image_start_x = None
        self.move_image_start_y = None

    def on_click(self, event):
        if self.freehand_mode:
            return

        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.annotation_mode == "rectangle":
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                        outline=self.rect_color)
            self.log_action(f"Started rectangle at ({self.start_x}, {self.start_y})")
        elif self.annotation_mode == "polygon":
            self.polygon_points.append((self.start_x, self.start_y))
            if len(self.polygon_points) > 1:
                self.canvas.create_line(self.polygon_points[-2][0], self.polygon_points[-2][1],
                                        self.polygon_points[-1][0], self.polygon_points[-1][1], fill=self.polygon_color)
            self.log_action(f"Added polygon point ({self.start_x}, {self.start_y})")
        elif self.annotation_mode == "circle":
            self.circle_id = self.canvas.create_oval(self.start_x, self.start_y, self.start_x, self.start_y,
                                                     outline=self.circle_color)
            self.log_action(f"Started circle at ({self.start_x}, {self.start_y})")
        elif self.annotation_mode == "keypoints":
            self.keypoint_id = self.canvas.create_oval(self.start_x - 3, self.start_y - 3, self.start_x + 3, self.start_y + 3,
                                                       fill=self.keypoint_color, outline=self.keypoint_color)
            self.log_action(f"Added keypoint at ({self.start_x}, {self.start_y})")

    def on_drag(self, event):
        if self.freehand_mode:
            return

        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        if self.annotation_mode == "rectangle":
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
            self.log_action(f"Updated rectangle to ({cur_x}, {cur_y})")
        elif self.annotation_mode == "circle":
            self.canvas.coords(self.circle_id, self.start_x, self.start_y, cur_x, cur_y)
            self.log_action(f"Updated circle to ({cur_x}, {cur_y})")

    def on_release(self, event):
        if self.freehand_mode:
            return

        if self.annotation_mode == "polygon" and len(self.polygon_points) > 2:
            self.canvas.create_line(self.polygon_points[-1][0], self.polygon_points[-1][1],
                                    self.polygon_points[0][0], self.polygon_points[0][1], fill=self.polygon_color)
            self.log_action(f"Closed polygon with {len(self.polygon_points)} points")
            self.polygon_points.clear()

    def set_annotation_mode(self, mode):
        self.annotation_mode = mode
        self.log_action(f"Set annotation mode to {mode}")

    def choose_font_color(self):
        color = colorchooser.askcolor(title="Choose Font Color")[1]
        if color:
            self.font_color = color
            self.log_action(f"Changed font color to {color}")

    def choose_rect_color(self):
        color = colorchooser.askcolor(title="Choose Rectangle Color")[1]
        if color:
            self.rect_color = color
            self.log_action(f"Changed rectangle color to {color}")

    def choose_polygon_color(self):
        color = colorchooser.askcolor(title="Choose Polygon Color")[1]
        if color:
            self.polygon_color = color
            self.log_action(f"Changed polygon color to {color}")

    def choose_circle_color(self):
        color = colorchooser.askcolor(title="Choose Circle Color")[1]
        if color:
            self.circle_color = color
            self.log_action(f"Changed circle color to {color}")

    def choose_keypoint_color(self):
        color = colorchooser.askcolor(title="Choose Keypoint Color")[1]
        if color:
            self.keypoint_color = color
            self.log_action(f"Changed keypoint color to {color}")

    def toggle_freehand_mode(self):
        self.freehand_mode = not self.freehand_mode
        self.canvas.config(cursor="cross" if self.freehand_mode else "arrow")
        self.log_action(f"Toggled freehand mode to {'on' if self.freehand_mode else 'off'}")

    def add_label(self):
        if self.freehand_mode:
            return

        label = self.label_entry.get()
        if not label:
            messagebox.showwarning("Warning", "Please enter a label.")
            return

        if self.annotation_mode == "rectangle" and self.rect_id:
            self.labels.append((self.rect_id, label))
            self.label_entry.delete(0, tk.END)
            self.canvas.create_text(self.start_x, self.start_y - 10, text=label, fill=self.font_color)
            self.log_action(f"Added label '{label}' to rectangle")
        elif self.annotation_mode == "circle" and self.circle_id:
            self.labels.append((self.circle_id, label))
            self.label_entry.delete(0, tk.END)
            self.canvas.create_text(self.start_x, self.start_y - 10, text=label, fill=self.font_color)
            self.log_action(f"Added label '{label}' to circle")
        elif self.annotation_mode == "keypoints" and self.keypoint_id:
            self.labels.append((self.keypoint_id, label))
            self.label_entry.delete(0, tk.END)
            self.canvas.create_text(self.start_x, self.start_y - 10, text=label, fill=self.font_color)
            self.log_action(f"Added label '{label}' to keypoint")

    def save_annotations(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to save annotations.")
            return
        annotations = []
        for rect_id, label in self.labels:
            coords = self.canvas.coords(rect_id)
            annotations.append({
                'label': label,
                'bbox': (coords[0], coords[1], coords[2], coords[3])
            })
        for line_id, label in self.freehand_labels:
            coords = self.canvas.coords(line_id)
            annotations.append({
                'label': label,
                'freehand': coords
            })
        # Save annotations (e.g., to a file)
        with open("annotations.json", "w") as f:
            json.dump(annotations, f, indent=4)
        messagebox.showinfo("Info", "Annotations saved.")
        self.log_action("Saved annotations")

    def load_annotations(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to load annotations.")
            return
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        with open(file_path, "r") as f:
            annotations = json.load(f)
        for ann in annotations:
            if 'bbox' in ann:
                rect_id = self.canvas.create_rectangle(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3],
                                                       outline=self.rect_color)
                self.labels.append((rect_id, ann['label']))
                self.canvas.create_text(ann['bbox'][0], ann['bbox'][1] - 10, text=ann['label'], fill=self.font_color)
            elif 'freehand' in ann:
                line_id = self.canvas.create_line(ann['freehand'], fill=self.freehand_color, width=2)
                self.freehand_labels.append((line_id, ann['label']))
        messagebox.showinfo("Info", "Annotations loaded.")
        self.log_action("Loaded annotations")

    def save_project(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to save project.")
            return
        project = {
            "image": self.image_path,
            "annotations": []
        }
        for rect_id, label in self.labels:
            coords = self.canvas.coords(rect_id)
            project["annotations"].append({
                'label': label,
                'bbox': (coords[0], coords[1], coords[2], coords[3])
            })
        for line_id, label in self.freehand_labels:
            coords = self.canvas.coords(line_id)
            project["annotations"].append({
                'label': label,
                'freehand': coords
            })
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        with open(file_path, "w") as f:
            json.dump(project, f, indent=4)
        messagebox.showinfo("Info", "Project saved.")
        self.log_action("Saved project")

    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        with open(file_path, "r") as f:
            project = json.load(f)
        self.image_path = project["image"]
        self.open_image_file(self.image_path)
        annotations = project["annotations"]
        for ann in annotations:
            if 'bbox' in ann:
                rect_id = self.canvas.create_rectangle(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3],
                                                       outline=self.rect_color)
                self.labels.append((rect_id, ann['label']))
                self.canvas.create_text(ann['bbox'][0], ann['bbox'][1] - 10, text=ann['label'], fill=self.font_color)
            elif 'freehand' in ann:
                line_id = self.canvas.create_line(ann['freehand'], fill=self.freehand_color, width=2)
                self.freehand_labels.append((line_id, ann['label']))
        messagebox.showinfo("Info", "Project loaded.")
        self.log_action("Loaded project")

    def open_image_file(self, image_path):
        self.image = Image.open(image_path)
        self.original_image = self.image.copy()  # Keep a copy of the original image

        self.fit_image_to_canvas()

    def save_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to save.")
            return

        # Draw annotations on the image
        draw = ImageDraw.Draw(self.image)
        for rect_id, label in self.labels:
            coords = self.canvas.coords(rect_id)
            draw.rectangle(coords, outline=self.rect_color, width=2)
            draw.text((coords[0], coords[1] - 10), label, fill=self.font_color)

        # Draw freehand drawings
        for line_id, label in self.freehand_labels:
            coords = self.canvas.coords(line_id)
            draw.line(coords, fill=self.freehand_color, width=2)
            x, y = coords[0:2]
            draw.text((x, y - 10), label, fill=self.font_color)

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if not file_path:
            return
        self.image.save(file_path)
        messagebox.showinfo("Info", "Image saved with annotations.")
        self.log_action("Saved image with annotations")

    def augment_image(self, method):
        if not self.image:
            messagebox.showwarning("Warning", "No image to augment.")
            return

        if method == "rotate_90":
            self.image = self.image.rotate(90, expand=True)
        elif method == "rotate_180":
            self.image = self.image.rotate(180, expand=True)
        elif method == "flip_horizontal":
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        elif method == "flip_vertical":
            self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        elif method == "scale_1.5":
            new_size = (int(self.image.width * 1.5), int(self.image.height * 1.5))
            self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        elif method == "scale_0.5":
            new_size = (int(self.image.width * 0.5), int(self.image.height * 0.5))
            self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)

        self.fit_image_to_canvas()
        self.log_action(f"Applied augmentation: {method}")

    def crop_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to crop.")
            return

        crop_box = simpledialog.askstring("Crop Image", "Enter crop box as 'x1,y1,x2,y2':")
        if crop_box:
            try:
                x1, y1, x2, y2 = map(int, crop_box.split(','))
                self.image = self.image.crop((x1, y1, x2, y2))
                self.fit_image_to_canvas()
                self.log_action(f"Cropped image with box ({x1}, {y1}, {x2}, {y2})")
            except ValueError:
                messagebox.showwarning("Warning", "Invalid crop box format.")

    def resize_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to resize.")
            return

        new_size = simpledialog.askstring("Resize Image", "Enter new size as 'width,height':")
        if new_size:
            try:
                width, height = map(int, new_size.split(','))
                self.image = self.image.resize((width, height), Image.Resampling.LANCZOS)
                self.fit_image_to_canvas()
                self.log_action(f"Resized image to ({width}, {height})")
            except ValueError:
                messagebox.showwarning("Warning", "Invalid size format.")

    def on_zoom(self, event):
        factor = 1.1
        if event.delta < 0:
            factor = 1 / factor
        self.scale *= factor
        self.canvas.scale("all", event.x, event.y, factor, factor)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))
        self.log_action(f"Zoomed {'in' if event.delta > 0 else 'out'} to scale {self.scale}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
