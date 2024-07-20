import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tifffile
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

def procesar_imagen(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def predict(model, img):
    img = img.squeeze(0)
    C, H, W = img.shape

    if C > 3:
        img = img[:3, :, :]
    elif C < 3:
        img = img.repeat(3 // C, 1, 1)

    img = img.permute(1, 2, 0).reshape(-1, C)
    
    pred = model.predict(img)
    pred = pred.reshape(H, W)
    return pred

def aplicar_contorno(label_image_path, original_image_path, num_lines, contour, off, min_contour_area):
    # Cargar la imagen label.png para encontrar contornos
    label_image = cv2.imread(label_image_path)
    label_image_rgb = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    # Definir el color específico para detectar
    blue_color = np.array([58, 82, 139])

    # Crear máscara para el color específico en label.png
    mask = cv2.inRange(label_image_rgb, blue_color, blue_color)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos pequeños
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Combinar todos los contornos en un solo array de puntos
    all_points = np.vstack([contour for contour in filtered_contours])

    # Encontrar el convex hull que envuelve todos los puntos
    hull = cv2.convexHull(all_points)

    # Interpolar puntos en el convex hull para tener num_lines líneas
    hull_points = hull.squeeze()
    if hull_points.ndim == 1:  # Manejar el caso de un solo punto
        hull_points = hull_points.reshape(-1, 2)
    
    # Interpolamos los puntos del convex hull para tener exactamente num_lines puntos
    hull_length = len(hull_points)
    step = hull_length // num_lines
    if step == 0:
        step = 1
    selected_points = [hull_points[i] for i in range(0, hull_length, step)]

    # Aseguramos que tengamos exactamente num_lines puntos
    while len(selected_points) > num_lines:
        selected_points.pop()
    while len(selected_points) < num_lines:
        selected_points.append(selected_points[-1])

    # Dibujar el convex hull en la imagen label.png
    label_image_with_contours = label_image_rgb.copy()
    for i in range(len(selected_points)):
        start_point = tuple(selected_points[i])
        end_point = tuple(selected_points[(i + 1) % len(selected_points)])
        cv2.line(label_image_with_contours, start_point, end_point, (255, 0, 0), 2)

    # Cargar la imagen original.png
    original_image = cv2.imread(original_image_path)
    original_image_with_hull = original_image.copy()

    # Dibujar las líneas en la imagen original.png
    for i in range(len(selected_points)):
        start_point = tuple(selected_points[i])
        end_point = tuple(selected_points[(i + 1) % len(selected_points)])
        cv2.line(original_image_with_hull, start_point, end_point, (0, 0, 255), 2)

    # Guardar las imágenes resultantes
    cv2.imwrite(contour, cv2.cvtColor(label_image_with_contours, cv2.COLOR_RGB2BGR))
    cv2.imwrite(off, original_image_with_hull)

if __name__ == "__main__":
    # Cargar el modelo entrenado
    model_path = "trained_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Cargar la nueva imagen
    new_image_path = sys.argv[1]
    img = procesar_imagen(new_image_path)


    # Predecir la etiqueta
    pred_label = predict(model, img)

    # Cargar el mapeo de clases
    base_pth = Path(f'datasets/preincendio_biobio_sentinel2/train/semantic_segmentation')
    with open(base_pth / "class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    cmap = mcolors.ListedColormap(['brown', 'blue', 'green', 'yellow', 'purple', 'red'])
    bounds = list(range(len(class_mapping) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la nueva imagen
    axs[0].imshow(img.squeeze().permute(1, 2, 0).numpy())
    axs[0].set_title('New Image')
    axs[0].axis('off')

    # Mostrar la etiqueta predicha con la paleta de colores
    axs[1].imshow(pred_label, cmap=cmap, norm=norm)
    axs[1].set_title('Predicted Label')
    axs[1].axis('off')

    #plt.show()

    # Guardar la etiqueta predicha
    label_save_path = 'app_label.png'
    plt.imsave(label_save_path, pred_label)
 
    original_image_path = sys.argv[1]
    num_lines = 8
    aplicar_contorno(label_save_path, original_image_path, num_lines, sys.argv[2], sys.argv[3], 500)
