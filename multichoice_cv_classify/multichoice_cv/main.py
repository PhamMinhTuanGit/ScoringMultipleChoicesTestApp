import image_processing 
import grid_info
import visualization
from utilities import extract_bubbles,generate_random_colors,append_to_file
from bubble_classify import BubbleClassifier
############### Chon anh muon tim o vuong ################
input_image_path = 'IMG_1581_iter_0.jpg'
############### Chon file output cua tap train ###########
##### Format giong nhu format cua txt trong data #########
input_data = 'IMG_1581_iter_0.txt'
#### Chon file ket qua ###################################
result_txt_path ='results_test'



########### Khong can quan tam ###########################
output_image_path = 'test_image_processing.jpg'
output_txt_path = 'test_image_processing.txt'


# Predefine configurations for sections
sections = [
    {
        "name": "SBD",
        "section_index": (0, 0),
        "axis": 0,
        "eps": 0.002,
        "input_string": "SBD",
        "gap_string": 0,
        "output_file": "SBD_results.txt"
    },
    {
        "name": "MDT",
        "section_index": (0, 1),
        "axis": 0,
        "eps": 0.002,
        "input_string": "MDT",
        "gap_string": 0,
        "output_file": "MDT_results.txt"
    },
    {
        "name": "phan1_1",
        "section_index": (1, 0),
        "axis": 1,
        "eps": 0.002,
        "input_string": "1.",
        "gap_string":0,
        "output_file": "Phan1_results.txt"
    },
    {
        "name": "phan1_2",
        "section_index": (1, 1),
        "axis": 1,
        "eps": 0.002,
        "input_string": "1.",
        "gap_string":10,
        "output_file": "Phan1_results.txt"
    },
    {
        "name": "phan1_3",
        "section_index": (1, 2),
        "axis": 1,
        "eps": 0.002,
        "input_string": "1.",
        "gap_string":20,
        "output_file": "Phan1_results.txt"
    },
    {
        "name": "phan1_4",
        "section_index": (1, 3),
        "axis": 1,
        "eps": 0.002,
        "input_string": "1.",
        "gap_string":30,
        "output_file": "Phan1_results.txt"
    },
    {
        "name": "phan2_1",
        "section_index": (2, 0),
        "axis": 1,
        "eps": 0.002,
        "input_string": "2.",
        "gap_string":0,
        "output_file": "Phan2_results.txt"
    },
    {
        "name": "phan2_2",
        "section_index": (2, 1),
        "axis": 1,
        "eps": 0.002,
        "input_string": "2.",
        "gap_string":1,
        "output_file": "Phan2_results.txt"
    },
    {
        "name": "phan2_3",
        "section_index": (2, 2),
        "axis": 1,
        "eps": 0.002,
        "input_string": "2.",
        "gap_string":4,
        "output_file": "Phan2_results.txt"
    },
    {
        "name": "phan2_4",
        "section_index": (2, 3),
        "axis": 1,
        "eps": 0.002,
        "input_string": "2.",
        "gap_string":6,
        "output_file": "Phan2_results.txt"
    },
    {
        "name": "phan3_1",
        "section_index": (3, 0),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.1",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    },
    {
        "name": "phan3_2",
        "section_index": (3, 1),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.2",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    },
    {
        "name": "phan3_3",
        "section_index": (3, 2),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.3",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    },
    {
        "name": "phan3_4",
        "section_index": (3, 3),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.4",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    },
    {
        "name": "phan3_5",
        "section_index": (3, 4),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.5",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    },
    {
        "name": "phan3_6",
        "section_index": (3, 5),
        "axis": 1,
        "eps": 0.002,
        "input_string": "3.6",
        "gap_string":"none",
        "output_file": "Phan3_results.txt"
    }
]

#################### HOW THIS CODE WORKS ##########################
# Buoc 1: Dau tien tim toa do cac o mau den
centers = image_processing.getNails(input_image_path, output_image_path, output_txt_path)
#print("The centers nails is", centers)

# Buoc 2: Sau do lay tu cac toa do ra cac phan vung
gridmatrix = grid_info.getGridmatrix(centers)
print("The grid matrix is", gridmatrix[5])
gridsection = grid_info.getExtractsections(gridmatrix) # This is the general grid
print("The sections 0 is",gridsection[0][1])
visualization.drawDots(input_image_path,gridsection[1][3])

# Buoc 3: Lay data va chon ra nhung hinh tron do co nam trong 
# phan vung minh muon, o day chon gridsection[3][2]
dots=extract_bubbles(input_data)
print("dots length is: ",len(dots))

# Buoc 4: Phan loai va dien ket qua
append_to_file(result_txt_path,input_image_path)
bubble_classifier = BubbleClassifier(gridsection, dots)
for section in sections:
    print(f"Processing {section['name']}...")
    bubble_classifier.process_grid_section(
        section_index=section["section_index"],
        axis=section["axis"],
        eps=section["eps"],
        input_string=section["input_string"],
        gap_string=section["gap_string"],
        #output_file=section["output_file"]
        output_file=result_txt_path
    )

# dotsinsdie = bubble_classify.dots_in_quadrilateral(gridsection[3][0],dots)

# # Buoc 4: Phan loai tung hang 1 de biet no thuoc cau nao
# classifieddots = bubble_classify.classify_batches(dotsinsdie,1)
# print("The classified dots is: ",classifieddots)
# bubble_classify.process_batches_to_file(classifieddots,"SBD",'test_results.txt')
# visualization.drawClustereddots(input_image_path,classifieddots)

# dotsinsdie = bubble_classify.dots_in_quadrilateral(gridsection[2][1],dots)

# # Buoc 4: Phan loai tung hang 1 de biet no thuoc cau nao
# classifieddots = bubble_classify.classify_batches(dotsinsdie,1)
# print("The classified dots is: ",classifieddots)
# bubble_classify.process_batches_to_file(classifieddots,"SBD",'test_results.txt')
# visualization.drawClustereddots('draw_batches.jpg',classifieddots)





