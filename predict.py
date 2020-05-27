import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import argparse
def process_image(img):
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))   
    image = (image/255)
    return image

def predict(image_path, model_file, labels_path, top_k=5):
    with open(labels_path, 'r') as labels_file:
        class_names = json.load(labels_file)
    model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})
    image1 = Image.open(image_path)
    image1 = np.asarray(image1)
    image1 = process_image(image1)
    image = np.expand_dims(image1, axis = 0)
    probpredictions = model.predict(image)
    prob_predictions= probpredictions[0].tolist()
    values, indices= tf.math.top_k(prob_predictions, k=top_k)
    probs=values.numpy().tolist()
    class_cat=indices.numpy().tolist()
    classes=[]
    for i in class_cat:
        if (i!=0):
            classes.append(class_names[str(i+1)])
    return probs,classes
    

parser = argparse.ArgumentParser(description='Flower Classifier')
parser.add_argument("image_path")
parser.add_argument("model_file")
parser.add_argument('--top_k', dest="top_k", type=int, default=5)
parser.add_argument('--category_names', dest="labels_path", default='./label_map.json')


args = parser.parse_args()

probs, classes = predict(args.image_path, args.model_file, args.labels_path, args.top_k)

print("Probs:",probs)
print("Classes:",classes)
