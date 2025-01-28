import tkinter as tk
from model import ANN, CNN
from gui import DigitalCanvas
from helper_funcs import load_model, predict_digit, preprocess_img


def main():
    root = tk.Tk()
    root.title("Draw a Digit")

    digit_canvas = DigitalCanvas(root)

    model = CNN(num_class=10)
    # model = ANN(28*28, 10)
    model = load_model(model, "model/cnn_model.pth")

    def predict():
        img_tensor = preprocess_img(digit_canvas.image)
        prediction = predict_digit(model, img_tensor)
        print(f"Predicted Digit: {prediction}")

    clear_button = tk.Button(root, text="Clear", command=digit_canvas.clear)
    clear_button.pack()

    predict_button = tk.Button(root, text="Predict", command=predict)
    predict_button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
