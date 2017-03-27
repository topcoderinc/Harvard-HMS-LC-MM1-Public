import numpy as np
import cv2


path_to_image = '/home/thomio/datasets/lung_tumor/example_extracted_sample/ANON_LUNG_TC001/pngs/76.png'
original_image = cv2.imread( path_to_image, cv2.IMREAD_GRAYSCALE )

non_zero = np.nonzero( original_image  )
image = original_image.copy()
image[non_zero] = 250

# Dilatation to close small holes
kernel = np.ones((1,1),np.uint8)
dilation = cv2.dilate(image, kernel, iterations = 1)

# Erosion to remove internal lung cells
kernel = np.ones((12,12),np.uint8)
erosion = cv2.erode(dilation, kernel, iterations = 1)

#kernel = np.ones((5,5),np.uint8)
#gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)

ret, thresh = cv2.threshold( erosion, 127, 255, 0 )
image2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

lungs = erosion * 0
original_image = cv2.cvtColor(original_image ,cv2.COLOR_GRAY2RGB)
for i in range(1, len(contours)):
    x,y,w,h = cv2.boundingRect( contours[i] )
    cv2.rectangle( original_image, (x,y), (x+w,y+h), (0,255,0), 1)
    roi = erosion[y:y+h, x:x+w]

#    print 'O = (', x, y, ')'
#    print 'D = (', x+w, y+h, ')'

    lung_pixels_rows_01, lung_pixels_cols_01 = np.where( roi == 0 )
    lung_pixels_rows_01 = lung_pixels_rows_01 + y
    lung_pixels_cols_01 = lung_pixels_cols_01 + x

    lungs[lung_pixels_rows_01, lung_pixels_cols_01] = 255


cv2.imwrite('erosion.png', erosion)
cv2.imwrite('lungs.png', lungs)
cv2.imwrite('original_image.png', original_image)

#cv2.imshow('dilation', dilation)
#cv2.imshow('gradient', gradient)
cv2.imshow('erosion', erosion)
cv2.imshow('lungs', lungs)
cv2.imshow('original_image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
