import numpy as np

def iou(bbox_aktual, bbox_prediksi):
  """
  Menghitung IOU (Intersection over Union) antara dua bounding box.

  Args:
    bbox_aktual: Bbox aktual, dalam bentuk list dengan format [x_awal, y_awal, x_akhir, y_akhir].
    bbox_prediksi: Bbox prediksi, dalam bentuk list dengan format [x_awal, y_awal, x_akhir, y_akhir].

  Returns:
    Nilai IOU.
  """

  # Memeriksa apakah bbox valid.

  if len(bbox_aktual) != 4 or len(bbox_prediksi) != 4:
    raise ValueError("Bbox harus memiliki panjang 4.")

  # Mengubah bbox ke bentuk numpy array.

  bbox_aktual = np.array(bbox_aktual)
  bbox_prediksi = np.array(bbox_prediksi)

  # Menghitung koordinat intersection.

  x_awal_inter = np.maximum(bbox_aktual[0], bbox_prediksi[0])
  y_awal_inter = np.maximum(bbox_aktual[1], bbox_prediksi[1])
  x_akhir_inter = np.minimum(bbox_aktual[2], bbox_prediksi[2])
  y_akhir_inter = np.minimum(bbox_aktual[3], bbox_prediksi[3])

  # Menghitung area intersection.

  area_inter = (x_akhir_inter - x_awal_inter) * (y_akhir_inter - y_awal_inter)

  # Menghitung area union.

  area_aktual = (bbox_aktual[2] - bbox_aktual[0]) * (bbox_aktual[3] - bbox_aktual[1])
  area_prediksi = (bbox_prediksi[2] - bbox_prediksi[0]) * (bbox_prediksi[3] - bbox_prediksi[1])
  area_union = area_aktual + area_prediksi - area_inter

  # Menghitung IOU.

  iou = area_inter / area_union

  return iou


# Contoh penggunaan.

bbox_aktual = [218.0, 355.0, 895.0, 463.0]
bbox_prediksi = [187.36895751953125, 360.5907287597656, 932.7528686523438, 448.6288757324219]

iou = iou(bbox_aktual, bbox_prediksi)

print(iou)