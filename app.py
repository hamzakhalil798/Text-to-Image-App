from flask import Flask, render_template, request
import torch
from min_dalle import MinDalle
from PIL import Image
import numpy as np

app = Flask(__name__)

dtype = "float16"


models_root='static/pretrained'

model = MinDalle(
    models_root=models_root,
    dtype=getattr(torch, dtype),
    device='cuda',
    is_mega=False, 
    is_reusable=False
)


def predict(text_input):
  
  image = model.generate_image(
      text=text_input,
      seed=-1,
      grid_size=1,
      is_seamless=False,
      temperature=1,
      top_k=256,
      supercondition_factor=32,
      is_verbose=False
  )

  image = np.array(image)
  image = Image.fromarray(image)

  # Save the generated image as a PNG file
  image.save('static/predictions/output.png')




@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/generate-image', methods=['POST'])
def generate_image():
    text_input = request.form['text-input']
    # do something with the text input, like generate an image
    predict(text_input)
    return render_template('display_image.html') 




if __name__ == '__main__':
    app.run()
