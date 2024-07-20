import random
from flask import Flask, request, send_file, render_template
from PIL import Image, ImageDraw
from pystac_client import Client
from odc.stac import stac_load

import rasterio
import rioxarray
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import copy


import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import tifffile
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image



import subprocess
from flask import Flask, request, send_file, render_template, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        input_path = 'app_inicial.png'
        file.save(input_path)
        return '', 200
    return '', 400

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    filename = data.get('filename')
    if filename:
        input_path = 'app_inicial.png'
        label_output_path = 'app_label_contours.png'
        final_output_path = 'app_oficial.png'
        
        # Ejecuta el script sentinel.py
        subprocess.run(['python', 'sentinel.py', input_path, label_output_path, final_output_path])
        
        response = {
            'label_url': f'/{label_output_path}',
            'final_url': f'/{final_output_path}'
        }
        return jsonify(response)
    return '', 400

@app.route('/<filename>')
def get_file(filename):
    return send_file(filename, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)