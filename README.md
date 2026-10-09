# calc-ai

A calculator where you draw the numbers by hand. A neural network built from scratch reads each handwritten digit, and the app runs the operation you pick: addition, subtraction, multiplication or division.

No machine learning libraries. Every weight, forward pass and backpropagation step is written by hand, in plain PHP and JavaScript, so you can follow exactly what the network is doing.

## How it works

```
draw digit ─► crop & center ─► 28×28 grayscale ─► neural network ─► "3" (94%)
draw digit ─► crop & center ─► 28×28 grayscale ─► neural network ─► "4" (91%)
                                                                       │
                                        pick operation (+ − × ÷) ──────┴─► 3 + 4 = 7
```

1. You draw one digit in each canvas.
2. Each drawing is converted to the same format as the MNIST dataset: cropped, scaled to fit a 20×20 box, centered in a 28×28 image, with grayscale values from 0 to 1.
3. The 784 pixels go into the network, which outputs a confidence score for each digit from 0 to 9. The highest score wins.
4. The same trained network reads both digits.
5. The app applies the chosen operation to the two recognized digits.

### Why the math is not done by the network

The network only does what neural networks are good at: recognizing messy, human input. The arithmetic itself runs as regular code, because code is exact and works for any number, while a network trained to add would only give approximate answers within the range it was trained on. Choosing where a model belongs, and where it does not, is part of the design.

## Project structure

```
calc-ai/
├── training/          PHP training pipeline (runs once, offline)
│   ├── train.php      downloads MNIST, trains the network, saves the weights
│   └── NeuralNetwork.php
├── web/               JavaScript front end
│   ├── index.html     two drawing canvases and the operation picker
│   ├── preprocess.js  converts a drawing into the MNIST 28×28 format
│   ├── network.js     forward pass using the trained weights
│   └── model.json     trained weights and biases
└── docs/
```

## Training (PHP)

Training happens once, offline. The script downloads the MNIST handwritten digit dataset (60,000 training images and 10,000 test images), trains the network and exports the weights and biases to `web/model.json`.

```bash
php training/train.php
```

| Setting | Value |
|---|---|
| Architecture | 784 → _hidden layers_ → 10 |
| Hidden activation | _tanh / ReLU_ |
| Output | 10 scores, one per digit |
| Learning rate | _value_ |
| Epochs | _value_ |
| Test accuracy | _value_ % |

Pixel values are divided by 255 so every input is between 0 and 1.

## Front end (JavaScript)

The page loads `model.json` and runs only the forward pass, so recognition is instant and works entirely in the browser.

The hardest part is preprocessing. MNIST digits are always centered and about the same size, while people draw small, large, thick, thin or in a corner. Before a drawing reaches the network, the front end:

- finds the bounding box of the strokes and crops it
- scales it to fit a 20×20 box, keeping the proportions
- centers it in a 28×28 image
- downsamples the large canvas so each pixel becomes a grayscale value, not just 0 or 1

The page shows the 28×28 image that actually goes into the network, so you can see what the model sees.

## Run locally

```bash
cd web
php -S localhost:8000
```

Then open `http://localhost:8000`.

## Limitations

- Recognizes one digit per canvas, so numbers go from 0 to 9.
- The operator is chosen with buttons, not drawn.
- Accuracy on real drawings is lower than on the MNIST test set, mainly because of differences in stroke style.

## Roadmap

- [ ] Multi-digit numbers
- [ ] Recognize drawn operators (+ − × ÷)
- [ ] Visualize neuron activations while drawing
- [ ] Compare results against a TensorFlow.js model to validate the backpropagation

## License

MIT
