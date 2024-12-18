import image_processing 
import grid_info
import visualization
from utilities import extract_bubbles,generate_random_colors
import bubble_classify
input_image_path = 'IMG_1581_iter_0.jpg'
input_data = 'IMG_1581_iter_0.txt'
output_image_path = 'test_image_processing.jpg'
output_txt_path = 'test_image_processing.txt'
#file_path = 'IMG_1581_iter_0.txt'

centers = image_processing.getNails(input_image_path, output_image_path, output_txt_path)
#print("The centers nails is", centers)
gridmatrix = grid_info.getGridmatrix(centers)
print("The grid matrix is", gridmatrix[5])
gridsection = grid_info.getExtractsections(gridmatrix)
print("The sections 0 is",gridsection[1][0])
visualization.draw_dots_on_image(input_image_path,gridsection[3][0])
dots=extract_bubbles(input_data)
print("dots length is: ",len(dots))
dotsinsdie = bubble_classify.dots_in_quadrilateral(gridsection[3][2],dots)
visualization.draw_dots_on_image(input_image_path,dotsinsdie)
classifieddots = bubble_classify.classify_batches(dotsinsdie)
print("The classified dots is: ",classifieddots)
visualization.visualize_batches(classifieddots)

