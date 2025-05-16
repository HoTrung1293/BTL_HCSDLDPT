import numpy as np

def cosine_similarity_np(vector_a, matrix_b):
        """
        Tính độ tương đồng cosine giữa một vector và một ma trận các vector
        
        Parameters:
        -----------
        vector_a : numpy.ndarray
            Vector đầu vào
        matrix_b : numpy.ndarray
            Ma trận các vector để so sánh
            
        Returns:
        --------
        numpy.ndarray
            Mảng các giá trị tương đồng cosine
        """
        # Tính tích vô hướng (dot product)
        dot_product = np.dot(matrix_b, vector_a)
        
        # Tính độ dài (norm) của vector_a
        norm_a = np.linalg.norm(vector_a)
        
        # Tính độ dài (norm) của từng vector trong matrix_b
        norm_b = np.linalg.norm(matrix_b, axis=1)
        
        # Tránh chia cho 0
        epsilon = 1e-10
        
        # Tính độ tương đồng cosine
        similarity = dot_product / (norm_a * norm_b + epsilon)
        
        return similarity

def euclidean_distance_np(vector_a, matrix_b):
        """
        Tính khoảng cách Euclidean giữa một vector và một ma trận các vector
        
        Parameters:
        -----------
        vector_a : numpy.ndarray
            Vector đầu vào
        matrix_b : numpy.ndarray
            Ma trận các vector để so sánh
            
        Returns:
        --------
        numpy.ndarray
            Mảng các giá trị khoảng cách Euclidean
        """
        # Tính khoảng cách Euclidean: căn bậc hai của tổng bình phương hiệu
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        squared_diff = np.sum(vector_a**2) + np.sum(matrix_b**2, axis=1) - 2 * np.dot(matrix_b, vector_a)
        
        # Xử lý các giá trị âm do lỗi làm tròn số
        squared_diff = np.maximum(squared_diff, 0)
        
        # Căn bậc hai để có khoảng cách Euclidean
        distances = np.sqrt(squared_diff)
        
        return distances

def manhattan_distance_np(vector_a, matrix_b):
        """
        Tính khoảng cách Manhattan giữa một vector và một ma trận các vector
        
        Parameters:
        -----------
        vector_a : numpy.ndarray
            Vector đầu vào
        matrix_b : numpy.ndarray
            Ma trận các vector để so sánh
            
        Returns:
        --------
        numpy.ndarray
            Mảng các giá trị khoảng cách Manhattan
        """
        # Tính khoảng cách Manhattan: tổng giá trị tuyệt đối của hiệu
        distances = np.sum(np.abs(matrix_b - vector_a), axis=1)
        
        return distances