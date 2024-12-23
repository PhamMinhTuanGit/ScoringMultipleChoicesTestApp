import image_processing
import grid_info
import visualization
#input_image_path = 'D:\\4_2_AREAS\\testset1\\testset1\\images\\IMG_3960_iter_0.jpg'
input_image_path = "IMG_1581_iter_0.jpg"
#input_data = 'IMG_1581_iter_0.txt'
result_txt_path = 'results_test_2.txt'

centers = image_processing.getNails(input_image_path)
#centers_duc=image_processing.detect_black_square_centers(input_image_path)
#visualization.drawDots(input_image_path,"draw_centers_duc.jpg")
#visualization.drawDots(input_image_path,centers,"draw_centers.jpg")
gridmatrix=grid_info.getGridmatrix(centers)
print("The length of the grid is: ",len(centers))
section_grid=grid_info.getExtractsections(gridmatrix)
print(section_grid[3][1])
#flaten_matrix=grid_info.flattenMatrix(gridmatrix)
visualization.drawDots(input_image_path,section_grid[3][0])