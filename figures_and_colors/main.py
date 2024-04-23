import numpy as np 
import matplotlib.pyplot as plt 
from skimage import color 
from skimage.measure import regionprops, label 
from pathlib import Path 

colors = ['красный', 'желтый', 'зеленый', 'светло-зеленый', 'синий', 'фиолетовый']

def calculate_means(hsv_image): 
    epsilon = np.diff(np.unique(hsv_image[:, :, 0])).mean() 

    return [np.mean(vals) * 360 for vals in np.array_split(np.unique(hsv_image[:, :, 0]), np.where(np.diff(np.unique(hsv_image[:, :, 0])) > epsilon)[0] + 1)] 
 
def calculate_midpoints(values): 
    return [(v1 + v2) / 2 for v1, v2 in zip(values, values[1:] + [values[0] + 360])] 
 
def determine_figure_color(region): 
    color_value = hsv_image[int(region.centroid[0]), int(region.centroid[1]), 0] * 360 

    return next((label for label, border_color in zip(colors, border_values) if color_value < border_color), 'red') 
 
image = plt.imread(Path(__file__).parent / 'balls_and_rects.png') 
hsv_image = color.rgb2hsv(image) 
 
binary_mask = (np.sum(image, 2) > 0).astype(int) 
labeled_regions = label(binary_mask) 
regions_info = regionprops(labeled_regions) 
 
color_means = calculate_means(hsv_image)[1:] 
border_values = calculate_midpoints(color_means) 
 
figures_circle = {} 
figures_rect = {} 
 
for region in regions_info: 
    color_figure = determine_figure_color(region) 
    figures_dict = figures_circle if np.all(region.image) else figures_rect 
    figures_dict[color_figure] = figures_dict.get(color_figure, 0) + 1 
 
print('Общее', labeled_regions.max()) 
print('Овалы', figures_circle) 
print('Прямоугольники', figures_rect)