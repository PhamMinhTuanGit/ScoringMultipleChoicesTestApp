from utils import generate_grid_coordinates, cut_image_from_points

def add_points_by_region(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    size_x = x_max - x_min
    size_y = y_max - y_min
    for box in yolo_boxes:
      final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    
    return final_coordinate

def add_points_by_region_two(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        (centers[idx_list[0]-1][0]+0.09, centers[idx_list[0]-1][1]),  # Point 3
        (centers[idx_list[1]-1][0]+0.09, centers[idx_list[1]-1][1])   # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    size_x = x_max - x_min
    size_y = y_max - y_min
    for box in yolo_boxes:
      final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    
    return final_coordinate

def add_points_by_region_three(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        (centers[idx_list[1]-1][0], centers[idx_list[0]-1][1])   # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)

    size_x = x_max - x_min
    size_y = y_max - y_min
    for box in yolo_boxes:
      final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    
    return final_coordinate

def add_points_by_region_custom(image, idx_list, start_coord, grid_size, cell_spacing, centers, final_coordinate):
    points_of_interest = [
        centers[idx_list[0]-1],  # Point 1
        centers[idx_list[1]-1],  # Point 2
        centers[idx_list[2]-1],  # Point 3
        centers[idx_list[3]-1],  # Point 4
    ]

    x_min, y_min, x_max, y_max = cut_image_from_points(image, points_of_interest)

    yolo_boxes = generate_grid_coordinates(start_coord, grid_size, cell_spacing)
    yolo_boxes.append((0.36, 0.37-(0.04+0.017)*2, 0.1, 0.04))
    yolo_boxes.append((0.36+(0.1+0.044), 0.37-(0.04+0.017), 0.1, 0.04))
    yolo_boxes.append((0.36+(0.1+0.044)*2, 0.37-(0.04+0.017), 0.1, 0.04))

    size_x = x_max - x_min
    size_y = y_max - y_min
    for box in yolo_boxes:
      final_coordinate.append((x_min + size_x*box[0], y_min + size_y*box[1], int(box[2]*(size_x)), box[3]*size_y))
    
    return final_coordinate