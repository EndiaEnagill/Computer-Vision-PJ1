import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def visualize_weights_heatmap(W, layer_name):
    plt.figure()
    plt.imshow(W, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.title(f'Weight Matrix Heatmap - {layer_name}')
    plt.xlabel('Output Units')
    plt.ylabel('Input Units')
    save_path = os.path.join('visualization_results', layer_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


with open('./model_data/demo/best_model.pkl', 'rb') as f:
    model_para = pickle.load(f)
W_1 = model_para['Linear_0_W']
W_2 = model_para['Linear_2_W']
W_3 = model_para['Linear_4_W']

visualize_weights_heatmap(W_1, 'W1')
visualize_weights_heatmap(W_2, 'W2')
visualize_weights_heatmap(W_3, 'W3')

# W_1
save_dir = 'visualization_results/W_1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(W_1.shape[1]):
    image = W_1[:, i].reshape((3, 32, 32))
    image = (image - image.min())/(image.max() - image.min()) * 255
    image = image.astype(np.int8)
    plt.imshow(image.transpose(1, 2, 0)) 
    plt.title(f'Image {i+1}')
    plt.axis('off')  
    
    save_path = os.path.join(save_dir, f'image_{i+1}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  

# W_2
save_dir = 'visualization_results/W_2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(W_2.shape[1]):
    image = W_2[:, i].reshape((1, 8, 16))
    plt.imshow(image.transpose(1, 2, 0), cmap='gray') 
    plt.title(f'Image {i+1}')
    plt.axis('off')  
    
    save_path = os.path.join(save_dir, f'image_{i+1}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  

# W_3
save_dir = 'visualization_results/W_3'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(W_3.shape[1]):
    image = W_3[:, i].reshape((1, 8, 8))
    
    plt.imshow(image.transpose(1, 2, 0), cmap='gray') 
    plt.title(f'Image {i+1}')
    plt.axis('off')  
    
    save_path = os.path.join(save_dir, f'image_{i+1}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  