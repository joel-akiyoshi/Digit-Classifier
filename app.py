import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model


class App:
    '''
    Tkinter window based application to facilitate drawing & prediction of handwritten digits.
    '''
    def __init__(self, root):
        self.root = root
        self.root.configure(bg="lightblue")

        # initialize cnn_model
        self.model = load_model("mnist_cnn_model.keras")

        # make a canvas for drawing
        self.canvas = tk.Canvas(root, width=800, height=800, bg='white')
        self.canvas.bind("<B1-Motion>", self.draw_canvas)

        # Image object for drawing (canvas_image), ImageDraw object to modify Image
        self.canvas_image = Image.new("L", (800, 800), (0))
        self.draw_image = ImageDraw.Draw(self.canvas_image)

        # buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict, width=10, height=2, font=("Georgia", 20), bg="#348feb")
        self.clear_button = tk.Button(root, text="Clear", command=self.clear, width=10, height=2, font=("Georgia", 20), bg="#f79eb6")

        # labels
        self.instruction = tk.Label(root, text="Draw a Digit Below:", font=("Georgia", 40), justify="center", bg="lightblue")
        self.result_label = tk.Label(root, text="", font=("Georgia", 40), justify="center", bg="lightblue")
        self.graph_label = tk.Label(root, text="Probability Graph:", font=("Georgia", 30), justify="center", bg="lightblue")

        # graph
        self.bar_fig = plt.figure(figsize=(16, 8))
        plt.xlabel("Digit")
        plt.ylabel("Probability")
        plt.xticks(range(10))
        self.bar_canvas = FigureCanvasTkAgg(figure=self.bar_fig, master=root)
        self.bar_canvas.draw()

        # grid positioning
        self.instruction.grid(row=0, column=0, columnspan=2, pady=50)
        self.graph_label.grid(row=0, column=2, pady=50)

        self.canvas.grid(row=1, column=0, columnspan=2, padx = (50, 25))
        self.bar_canvas.get_tk_widget().grid(row=1, column=2, padx=(25, 50))

        self.predict_button.grid(row=2, column=0, pady=50, padx=50, sticky="e")
        self.clear_button.grid(row=2, column=1, pady=50, padx=50, sticky="w")
        self.result_label.grid(row=2, column=2)


    def draw_canvas(self, event):
        '''
        Called when user begins drawing. Modifies self.canvas and self.canvas_image.
        '''
        x, y = event.x, event.y
        r = 30

        # modifies visible canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')

        # modifies canvas_image (what will be processed)
        self.draw_image.ellipse([x-r, y-r, x+r, y+r], fill='white')


    def clear(self):
        '''
        Clear all drawings, plots, and result labels. Clear plot.
        '''
        self.canvas.delete("all")
        self.canvas_image = Image.new("L", (800, 800), (0))
        self.draw_image = ImageDraw.Draw(self.canvas_image)
        self.result_label.config(text="")
        self.bar_fig.clear()
        plt.xlabel("Digit")
        plt.ylabel("Probability")
        plt.xticks(range(10))
        self.bar_canvas.draw()


    def predict(self):
        '''
        Resize and scale down canvas_image, predict digit via CNN.
        '''
        self.canvas_image = self.canvas_image.resize((28, 28), Image.LANCZOS)
        self.canvas_image.save("drawn.png")
        
        # Convert to numpy array, normalize data
        img_array = np.array(self.canvas_image).reshape(1, 28, 28, 1).astype('float32') / 255

        # Model prediction (list of 10D vectors)
        prediction = self.model.predict(img_array)

        # 10D vector for each num
        probabilities = prediction[0]
        nums = range(10)

        # Plot probabilities
        plt.bar(nums, probabilities, color="green")
        self.bar_canvas.draw()

        # Display max
        result_text = f"Predicted: {np.argmax(probabilities)}\n"
        self.result_label.config(text=result_text)



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
