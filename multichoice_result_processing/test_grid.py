import image_processing
import grid_info
import visualization
input_image_path = 'IMG_1581_iter_0.jpg'
input_data = 'IMG_1581_iter_0.txt'
result_txt_path = 'results_test_2.txt'

centers = image_processing.getNails(input_image_path)
gridmatrix=grid_info.getGridmatrix(centers)
print("The length of the grid is: ",)
# section_grid=grid_info.getExtractsections(gridmatrix)
# print(section_grid[3][1])
flaten_matrix=grid_info.flattenMatrix(gridmatrix)
visualization.drawDots(input_image_path,flaten_matrix)