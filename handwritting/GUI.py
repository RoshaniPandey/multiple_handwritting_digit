from tkinter import *
import numpy as np
from PIL import ImageGrab, ImageOps, Image
import os
from Prediction import predict

window = Tk()
window.title("Handwritten Digit Recognition")
l1 = Label()

def preprocess_image(img):
    """ Convert image to grayscale, resize, invert colors, normalize, and flatten. """
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (white digit on black background)
    
    # Crop bounding box of the digit
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Resize while maintaining aspect ratio
    img = img.resize((20, 20), Image.Resampling.LANCZOS)

    # Create a blank 28x28 canvas
    new_img = Image.new('L', (28, 28), 0)  # Black background
    new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))  # Center the digit

    # Convert to numpy array
    img_array = np.asarray(new_img, dtype=np.float32)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Flatten the image (1D array of 784 elements)
    return img_array.reshape(1, 784)

def MyProject():
    global l1

    widget = cv
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Capture the drawn digit
    img = ImageGrab.grab().crop((x, y, x1, y1))

    # Preprocess image
    vec = preprocess_image(img)

    # Check if trained weights exist
    if not os.path.exists("Theta1.txt") or not os.path.exists("Theta2.txt"):
        l1 = Label(window, text="Error: Model not trained!", font=('Algerian', 20), fg="red")
        l1.place(x=200, y=420)
        return

    # Load trained weights
    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')

    # Predict digit
    pred = predict(Theta1, Theta2, vec)

    # Display prediction
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Algerian', 20))
    l1.place(x=230, y=420)

lastx, lasty = None, None

def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()

def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=True, splinesteps=12)
    lastx, lasty = x, y

# GUI Elements
L1 = Label(window, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10)

b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
