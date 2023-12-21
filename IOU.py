def fungsi_iou(Bbox_GroundT, Bbox_prediksi):
    x = max(Bbox_GroundT['x'], Bbox_prediksi['x'])
    y = max(Bbox_GroundT['y'], Bbox_prediksi['y'])
    w = min(Bbox_GroundT['w'], Bbox_prediksi['w'])
    h = min(Bbox_GroundT['h'], Bbox_prediksi['h'])
    if w<0 or h<0: 
        return 0
    interArea = (w-x) * (h-y)
    groundTArea = (Bbox_GroundT['w'] - Bbox_GroundT['x']) * (Bbox_GroundT['h'] - Bbox_GroundT['y'])
    prediksiArea = (Bbox_prediksi['w'] - Bbox_prediksi['x']) * (Bbox_prediksi['h'] - Bbox_prediksi['y'])
    iou = interArea / float(groundTArea + prediksiArea - interArea)
    return iou

# Bounding box aktual
Bbox_GroundT = {'x': 218.0, 'y': 355.0, 'w': 895.0, 'h': 463.0}

# Bounding box prediksi
Bbox_prediksi = {'x': 187.36895751953125, 'y': 360.5907287597656, 'w': 932.7528686523438, 'h': 448.6288757324219}

# Menghitung IOU
iou = fungsi_iou(Bbox_GroundT, Bbox_prediksi)
print('Nilai IOU:', iou)