import image_processing
import grid_info
import visualization
import utilities
from bubble_classify import BubbleClassifier
#input_image_path = 'D:\\4_2_AREAS\\testset1\\testset1\\images\\IMG_3960_iter_0.jpg'
#input_image_path ="IMG_1581_iter_0.jpg"
input_image_path="D:\\4_2_AREAS\\Trainning_SET\\Images\\IMG_1581_iter_1.jpg"

# Buoc 1 tim tam
centers = image_processing.getNails(input_image_path)
visualization.drawDots(input_image_path,centers,"draw_centers.jpg")
gridmatrix=grid_info.getGridmatrix(centers)
print("The length of the grid is: ",len(gridmatrix))

# Buoc 2 tim toa do cua cac cac goc trong 1 section
gridsection = grid_info.getExtractsections(gridmatrix)
print(gridsection[3][1])
dots = utilities.extract_bubbles("IMG_1581_iter_0.txt")

# Buoc 3 tim cac o tron co trong 1 section
bubble_classifier = BubbleClassifier(gridsection, dots)
insidedots = bubble_classifier.dots_in_quadrilateral(gridsection[3][1])

# Buoc 4 phan loai theo hang hoac cot, mac dinh la theo hang == 1
clustered_batches = bubble_classifier.classify_batches(insidedots)
visualization.drawClustereddots(input_image_path,clustered_batches)
