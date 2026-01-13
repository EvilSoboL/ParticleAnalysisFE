from src.data_processing.cam_sorted import CamSorter

path_to_test_data = r'C:\Users\evils\PycharmProjects\ParticleAnalysisFE\tests\test_data'

# Простое использование
sorter = CamSorter(path_to_test_data)
cam1_pairs, cam2_pairs = sorter.sort_images()
stats = sorter.get_statistics()

