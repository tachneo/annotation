import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")

        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.scroll_y = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<B1-Motion>", self.on_freehand_draw, add="+")
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_zoom)

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.drawing = False
        self.freehand_id = None
        self.image = None
        self.tk_image = None
        self.original_image = None
        self.labels = []
        self.freehand_labels = []
        self.font_color = "black"
        self.rect_color = "red"
        self.freehand_color = "blue"
        self.scale = 1.0

        self.undo_stack = []
        self.redo_stack = []

        # Create menus
        self.menubar = tk.Menu(root)
        self.root.config(menu=self.menubar)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        self.file_menu.add_command(label="Save Image", command=self.save_image)
        self.file_menu.add_command(label="Exit", command=root.quit)

        # Tool frame
        self.tool_frame = tk.Frame(self.main_frame, padx=5, pady=5)
        self.tool_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Color selection buttons
        self.font_color_button = tk.Button(self.tool_frame, text="Font Color", command=self.choose_font_color)
        self.font_color_button.pack(fill=tk.X, pady=2)
        self.rect_color_button = tk.Button(self.tool_frame, text="Rect Color", command=self.choose_rect_color)
        self.rect_color_button.pack(fill=tk.X, pady=2)
        self.freehand_color_button = tk.Button(self.tool_frame, text="Freehand Color", command=self.choose_freehand_color)
        self.freehand_color_button.pack(fill=tk.X, pady=2)

        # Label entry and add button
        self.label_entry = tk.Entry(self.tool_frame, width=20)
        self.label_entry.pack(fill=tk.X, pady=2)
        self.label_button = tk.Button(self.tool_frame, text="Add Label", command=self.add_label)
        self.label_button.pack(fill=tk.X, pady=2)

        # Toggle for freehand drawing
        self.freehand_button = tk.Button(self.tool_frame, text="Toggle Freehand Mode", command=self.toggle_freehand_mode)
        self.freehand_button.pack(fill=tk.X, pady=2)

        # Zoom controls
        self.zoom_in_button = tk.Button(self.tool_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(fill=tk.X, pady=2)
        self.zoom_out_button = tk.Button(self.tool_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(fill=tk.X, pady=2)

        # Rotation controls
        self.rotate_left_button = tk.Button(self.tool_frame, text="Rotate Left", command=self.rotate_left)
        self.rotate_left_button.pack(fill=tk.X, pady=2)
        self.rotate_right_button = tk.Button(self.tool_frame, text="Rotate Right", command=self.rotate_right)
        self.rotate_right_button.pack(fill=tk.X, pady=2)

        # Bindings for undo and redo
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
        self.image = Image.open(file_path)
        self.original_image = self.image.copy()  # Keep a copy of the original image
        self.fit_image_to_canvas()
        self.rect = None
        self.labels = []
        self.freehand_drawings = []
        self.freehand_labels = []  # Clear previous freehand labels
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.save_canvas_state()

    def fit_image_to_canvas(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.image.size

        scale = min(canvas_width / img_width, canvas_height / img_height)
        self.scale = scale
        self.display_image(scale)

    def display_image(self, scale):
        resized_image = self.image.resize((int(self.image.width * scale), int(self.image.height * scale)), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("image")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="image")
        self.canvas.tag_lower("image")  # Ensure the image is at the bottom layer
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_click(self, event):
        if self.freehand_mode:
            return
        if self.rect:
            self.canvas.delete(self.rect)
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                   outline=self.rect_color, tags="annotation")

    def on_drag(self, event):
        if self.freehand_mode:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_freehand_draw(self, event):
        if not self.freehand_mode:
            return
        if not self.drawing:
            self.drawing = True
            self.prev_x = self.canvas.canvasx(event.x)
            self.prev_y = self.canvas.canvasy(event.y)
        else:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            line_id = self.canvas.create_line(self.prev_x, self.prev_y, cur_x, cur_y, fill=self.freehand_color, width=2, tags="annotation")
            self.freehand_drawings.append(line_id)
            self.prev_x = cur_x
            self.prev_y = cur_y

    def on_freehand_draw_release(self, event):
        self.drawing = False

    def on_release(self, event):
        if self.freehand_mode and self.freehand_drawings:
            self.save_canvas_state()
        elif self.rect_id:
            self.save_canvas_state()

    def add_label(self):
        if not self.freehand_mode and self.rect_id:
            label = self.label_entry.get()
            if label:
                self.labels.append((self.rect_id, label))
                self.label_entry.delete(0, tk.END)
                self.canvas.create_text(self.start_x, self.start_y - 10, text=label, fill=self.font_color, tags="annotation")
                self.save_canvas_state()
            else:
                messagebox.showwarning("Warning", "Please enter a label.")
        elif self.freehand_mode:
            # Optionally handle labeling freehand drawings
            label = self.label_entry.get()
            if label:
                self.freehand_labels.append((self.freehand_drawings[-1], label))  # Label the last freehand drawing
                self.label_entry.delete(0, tk.END)
                # Create label for the last freehand drawing (simplified approach)
                x, y = self.canvas.coords(self.freehand_drawings[-1])[0:2]
                self.canvas.create_text(x, y - 10, text=label, fill=self.font_color, tags="annotation")
                self.save_canvas_state()
            else:
                messagebox.showwarning("Warning", "Please enter a label.")

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
        with open("annotations.txt", "w") as f:
            for ann in annotations:
                if 'bbox' in ann:
                    f.write(f"{ann['label']} {ann['bbox']}\n")
                else:
                    f.write(f"{ann['label']} {ann['freehand']}\n")
        messagebox.showinfo("Info", "Annotations saved.")

    def save_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "No image to save.")
            return

        # Draw annotations on the image
        annotated_image = self.original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
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
        annotated_image.save(file_path)
        messagebox.showinfo("Info", "Image saved with annotations.")

    def choose_font_color(self):
        color = colorchooser.askcolor(title="Choose Font Color")[1]
        if color:
            self.font_color = color

    def choose_rect_color(self):
        color = colorchooser.askcolor(title="Choose Rectangle Color")[1]
        if color:
            self.rect_color = color

    def choose_freehand_color(self):
        color = colorchooser.askcolor(title="Choose Freehand Color")[1]
        if color:
            self.freehand_color = color

    def toggle_freehand_mode(self):
        self.freehand_mode = not self.freehand_mode
        self.canvas.config(cursor="cross" if self.freehand_mode else "arrow")

    def on_zoom(self, event):
        factor = 1.1
        if event.delta < 0:
            factor = 1 / factor
        self.scale *= factor
        self.display_image(self.scale)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def zoom_in(self):
        self.scale *= 1.1
        self.display_image(self.scale)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def zoom_out(self):
        self.scale /= 1.1
        self.display_image(self.scale)
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

    def rotate_left(self):
        if self.image:
            self.image = self.image.rotate(90, expand=True)
            self.fit_image_to_canvas()
            self.save_canvas_state()

    def rotate_right(self):
        if self.image:
            self.image = self.image.rotate(-90, expand=True)
            self.fit_image_to_canvas()
            self.save_canvas_state()

    def save_canvas_state(self):
        items = self.canvas.find_all()
        state = []
        for item in items:
            item_type = self.canvas.type(item)
            coords = self.canvas.coords(item)
            tags = self.canvas.itemcget(item, "tags")
            state_item = {"type": item_type, "coords": coords, "tags": tags}
            if item_type == "rectangle" or item_type == "line":
                state_item["outline"] = self.canvas.itemcget(item, "outline")
                state_item["width"] = self.canvas.itemcget(item, "width")
            if item_type == "text":
                state_item["text"] = self.canvas.itemcget(item, "text")
                state_item["fill"] = self.canvas.itemcget(item, "fill")
            state.append(state_item)
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def load_canvas_state(self, state):
        self.canvas.delete("all")
        for item in state:
            item_type = item["type"]
            coords = item["coords"]
            tags = item["tags"]
            if item_type == "image":
                self.canvas.create_image(coords, anchor=tk.NW, image=self.tk_image, tags=tags)
            elif item_type == "rectangle":
                self.canvas.create_rectangle(coords, outline=item["outline"], width=item["width"], tags=tags)
            elif item_type == "line":
                self.canvas.create_line(coords, fill=item["outline"], width=item["width"], tags=tags)
            elif item_type == "text":
                self.canvas.create_text(coords, text=item["text"], fill=item["fill"], tags=tags)

    def undo(self, event=None):
        if len(self.undo_stack) > 1:
            state = self.undo_stack.pop()
            self.redo_stack.append(state)
            self.load_canvas_state(self.undo_stack[-1])

    def redo(self, event=None):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.load_canvas_state(state)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
