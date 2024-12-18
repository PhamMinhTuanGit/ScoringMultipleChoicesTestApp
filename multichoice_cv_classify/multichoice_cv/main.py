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

#################### HOW THIS CODE WORKS ##########################
# Buoc 1: Dau tien tim toa do cac o mau den
centers = image_processing.getNails(input_image_path, output_image_path, output_txt_path)
#print("The centers nails is", centers)

# Buoc 2: Sau do lay tu cac toa do ra cac phan vung
gridmatrix = grid_info.getGridmatrix(centers)
print("The grid matrix is", gridmatrix[5])
gridsection = grid_info.getExtractsections(gridmatrix)
print("The sections 0 is",gridsection[1][0])
visualization.draw_dots_on_image(input_image_path,gridsection[3][0])

# Buoc 3: Lay data va chon ra nhung hinh tron do co nam trong 
# phan vung minh muon, o day chon gridsection[3][2]
dots=extract_bubbles(input_data)
print("dots length is: ",len(dots))
dotsinsdie = bubble_classify.dots_in_quadrilateral(gridsection[3][2],dots)
visualization.draw_dots_on_image(input_image_path,dotsinsdie)

# Buoc 4: Phan loai tung hang 1 de biet no thuoc cau nao
classifieddots = bubble_classify.classify_batches(dotsinsdie)
print("The classified dots is: ",classifieddots)




