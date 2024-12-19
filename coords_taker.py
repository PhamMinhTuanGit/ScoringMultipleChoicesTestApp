import os

def average_of_row_i(directory, row_index):
    # Khởi tạo một danh sách để lưu trữ các giá trị của dòng thứ i từ tất cả các file
    row_values = []

    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Kiểm tra xem file có phải là file văn bản không
        if os.path.isfile(file_path) and filename.endswith('.txt'):  # Giả sử các file có đuôi .txt
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Kiểm tra nếu dòng thứ i có tồn tại trong file
            if row_index < len(lines):
                # Lấy giá trị dòng thứ i, chuyển thành list các giá trị số
                row = list(map(float, lines[row_index].split()))
                row_values.append(row)

    # Tính trung bình của các giá trị tại dòng thứ i
    if row_values:
        # Chuyển các giá trị thành numpy array để dễ tính toán trung bình theo cột
        import numpy as np
        row_values_np = np.array(row_values)
        avg_row = np.mean(row_values_np, axis=0)
        return avg_row
    else:
        return None  # Trả về None nếu không có dòng thứ i trong các file
def write_results_to_file(results, output_file):
    with open(output_file, 'w') as f:
        for result in results:
            if result is not None:
                # Ghi kết quả vào file với định dạng tương tự
                f.write(" ".join(map(str, result)) + "\n")
            else:
                f.write("\n")  # Ghi dòng trống nếu không có kết quả
def process_and_save_results(directory, output_file):
    # Lưu kết quả trung bình của từng dòng vào một danh sách
    results = []
    for i in range(0, 571):  # Giả sử bạn cần tính trung bình cho 571 dòng
        result = average_of_row_i(directory, i)
        results.append(result)

    # Ghi kết quả vào file
    write_results_to_file(results, output_file)
    print(f"Kết quả đã được ghi vào file: {output_file}")
process_and_save_results('/Users/phamminhtuan/Desktop/thư mục không có tiêu đề 2', '/Users/phamminhtuan/Desktop/AIChallenge/output.txt')