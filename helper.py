import numpy as np
import cv2
from sklearn import linear_model, datasets
import settings

# Choose the number of sliding windows
nwindows = 9

# Set the width of the windows +/- margin
margin = 20
increase = 0

# Set minimum number of pixels found to recenter window
minpix = 50    

deg = 2

alpha = 0.6 

w = 50
h = 50

attention_percentage = 0.5
attention_level = 2


# project video
# ym_per_pix = 3*17/720
# xm_per_pix = 3.7/(960-320)

# # challenge
# ym_per_pix = 3*3/270
# xm_per_pix = 3.7/(1055-175)

# harder challenge
ym_per_pix = 3./(433-333)
xm_per_pix = 3.7/(1051-164)

red_color = (255,0,0)
yellow_color = (255,255,0)
green_color = (0,174,0)

def extract_feature(img):

	img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	img_hls = cv2.medianBlur(img_hls, 5)
	img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
	img_lab = cv2.medianBlur(img_lab, 5)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
	black = cv2.morphologyEx(img_lab[:,:, 0], cv2.MORPH_TOPHAT, kernel)
	lanes = cv2.morphologyEx(img_hls[:,:,1], cv2.MORPH_TOPHAT, kernel)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
	lanes_yellow = cv2.morphologyEx(img_lab[:, :, 2], cv2.MORPH_TOPHAT, kernel)
	
	mask = np.zeros((settings.UNWARPED_SIZE[1], settings.UNWARPED_SIZE[0], 3), dtype=np.uint8)
	mask[:, :, 0] = cv2.adaptiveThreshold(black, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -6)
	mask[:, :, 1] = cv2.adaptiveThreshold(lanes, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
	mask[:, :, 2] = cv2.adaptiveThreshold(lanes_yellow, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
	                                           13, -1.5)

	small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	features = np.any(mask, axis=2).astype(np.uint8)
	features = cv2.morphologyEx(features.astype(np.uint8), cv2.MORPH_ERODE, small_kernel)

	return features

def radius_of_curvature(fitx):
	# Fit new polynomials to x,y in world space
	fity = np.arange(len(fitx))
	y = fity - len(fity)
	x = fitx - fitx.mean()
	a,b,c = np.polyfit(y[-100:]*ym_per_pix, x[-100:]*xm_per_pix, 2)
	return ((1 + b**2)**1.5)/(2*a)

def find_window_centroids(binary_warped, window_width=100, window_height=80, l_start = 400., r_start = 500.):
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions

	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template 

	# Sum quarter bottom of image to get slice, could use a different ratio
	img_height, img_width = binary_warped.shape[:2]
	top, mid = img_height*2//3, img_width//2

	# l_sum = np.sum(binary_warped[top:,:mid], axis=0)
	# r_sum = np.sum(binary_warped[top:,mid:], axis=0)

	# l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
	# r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+mid

	l_center = l_start
	r_center = r_start

	# Add what we found for the first layer
	window_centroids.append((int(l_center),int(r_center)))

    # Go through each layer looking for max pixel locations
	for level in range(1,img_height//window_height):
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(binary_warped[img_height-(level+1)*window_height:img_height-level*window_height], axis=0)
		conv_signal = np.convolve(window, image_layer)

		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,img_width))
		if np.max(conv_signal[l_min_index:l_max_index]) > 30:
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,img_width))
		if np.max(conv_signal[r_min_index:r_max_index]) > 30:
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

		# Add what we found for that layer
		window_centroids.append((int(l_center),int(r_center)))

	return window_centroids

def find_lane_pixels_with_window_centroids(binary_warped, window_centroids, window_width=100, window_height=80):        
	img_height, img_width = binary_warped.shape[:2]

	def window_bound(centroid, level):
		bottom = min(img_height-level*window_height,img_height)
		top    = max(img_height-(level+1)*window_height,0)
		left   = max(centroid-margin,0)
		right  = min(centroid+margin,img_width)
		return top, bottom, left, right

	left_window_mask = np.zeros_like(binary_warped, dtype=bool)
	right_window_mask = np.zeros_like(binary_warped, dtype=bool)

	out_img = (np.dstack((binary_warped>0, binary_warped>0, binary_warped>0))*255).astype(np.uint8)

	for level in range(len(window_centroids)):
		left_centroid, right_centroid = window_centroids[level]

		t, b, l, r = window_bound(left_centroid, level)
		left_window_mask[np.arange(t,b,dtype=int), np.arange(l,r,dtype=int).reshape(-1,1)] = True
		cv2.rectangle(out_img,(l,b),(r,t),(0,255,0), 2) 

		t, b, l, r = window_bound(right_centroid, level)
		right_window_mask[np.arange(t,b,dtype=int), np.arange(l,r,dtype=int).reshape(-1,1)] = True
		cv2.rectangle(out_img,(l,b),(r,t),(0,255,0), 2) 
		                
	lefty, leftx = ((binary_warped>0) & left_window_mask).nonzero()
	righty, rightx = ((binary_warped>0) & right_window_mask).nonzero()

	leftw = binary_warped[lefty, leftx]
	rightw = binary_warped[righty, rightx]

	return lefty, leftx, righty, rightx, leftw, rightw, out_img

def fit_lane(lefty, leftx, righty, rightx, leftw, rightw, img_height, img_width):

	def make_feature(y,right):
		yy = y - img_height
		polynomial = yy**np.arange(1,3).reshape(-1,1)
		dv_right = right*np.ones_like(yy)
		interaction = dv_right*(yy**2)
		return np.vstack([polynomial, dv_right, interaction]).transpose()

	features_left = make_feature(lefty, False)
	features_right = make_feature(righty, True)

	features = np.vstack([features_left, features_right])
	output = np.hstack([leftx, rightx])

	# Attention
	leftw[lefty > attention_percentage * img_height] *= attention_level
	rightw[righty > attention_percentage * img_height] *= attention_level

	weight = np.hstack([leftw*lefty, rightw*righty])

	if len(features) < 20:
	    return None, None, (None, None, None)

	# regr = linear_model.RANSACRegressor(linear_model.LinearRegression())
	regr=linear_model.LinearRegression()
	regr.fit(features, output, weight)
	    
	fity = np.arange(img_height)

	fit_features_left = make_feature(fity, False)
	fit_features_right = make_feature(fity, True)

	fit_leftx = regr.predict(fit_features_left)
	fit_rightx = regr.predict(fit_features_right)

	return regr, make_feature, (fity, fit_leftx, fit_rightx)

# def find_lanes_pipeline(binary_warped,window_width=100, window_height=80):
# 	window_centroids = find_window_centroids(binary_warped,window_width,window_height)
# 	lefty,leftx,righty,rightx,leftw,rightw,out_img = find_lane_pixels_with_window_centroids(binary_warped, window_centroids,window_width,window_height)
# 	windows_found = out_img.copy()
# 	regr,make_feature,(fity,fit_leftx,fit_rightx) = fit_lane(lefty, leftx, righty, rightx, leftw,rightw, settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0])

# 	out_img[lefty, leftx] = [255, 0, 0]
# 	out_img[righty, rightx] = [0, 0, 255]
# 	# cv2.polylines(out_img, [np.int_(list(zip(fit_leftx, fity)))], False, (0,255,255), thickness=7)
# 	# cv2.polylines(out_img, [np.int_(list(zip(fit_rightx, fity)))], False, (0,255,255), thickness=7)

# 	return out_img,fity,fit_leftx,fit_rightx,regr, windows_found, len(lefty), len(righty)

def draw_poly(out_img, fit_leftx, fit_rightx, fity, margin_increase, margin_color):
	# Generate a polygon to illustrate the search window area
	window_img = np.zeros_like(out_img)

	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx-margin-margin_increase[0], np.arange(len(fit_leftx))]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx+margin+margin_increase[0], np.arange(len(fit_leftx))])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))

	right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx-margin-margin_increase[1], np.arange(len(fit_rightx))]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx+margin+margin_increase[1], np.arange(len(fit_rightx))])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), margin_color[0])
	cv2.fillPoly(window_img, np.int_([right_line_pts]), margin_color[1])
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	cv2.polylines(result, [np.int_(list(zip(fit_leftx, np.arange(len(fit_leftx)))))], False, (0,255,255), thickness=2)
	cv2.polylines(result, [np.int_(list(zip(fit_rightx, np.arange(len(fit_rightx)))))], False, (0,255,255), thickness=2)

	return result

def find_lane_pixels_with_previous_fit(binary_warped, fit_leftx, fit_rightx, margin_increase):
	nonzeroy, nonzerox = np.where(binary_warped>0)

	left_lane_inds = np.abs(nonzerox - fit_leftx[nonzeroy]) <= margin + margin_increase[0]
	right_lane_inds = np.abs(nonzerox - fit_rightx[nonzeroy]) <= margin + margin_increase[1]

	fity = np.arange(binary_warped.shape[0])

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped>0, binary_warped>0, binary_warped>0))*255).astype(np.uint8)

	# Color in left and right line pixels
	out_img[lefty, leftx] = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]

	# result = draw_poly(out_img, fit_leftx, fit_rightx, fity)

	leftw = binary_warped[lefty, leftx]
	rightw = binary_warped[righty, rightx]

	return lefty, leftx, righty, rightx, leftw, rightw, out_img


# def find_lanes_pipeline2(binary_warped,fit_leftx,fit_rightx,window_width=100, window_height=80):
# 	lefty,leftx,righty,rightx,leftw, rightw, out_img = find_lane_pixels_with_previous_fit(binary_warped,fit_leftx,fit_rightx)
# 	windows_found = out_img.copy()
# 	regr,make_feature,(fity,fit_leftx,fit_rightx) = fit_lane(lefty, leftx, righty, rightx, leftw, rightw, settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0])
# 	out_img[lefty, leftx] = [255, 0, 0]
# 	out_img[righty, rightx] = [0, 0, 255]
# 	return out_img,fity,fit_leftx,fit_rightx,regr,windows_found, len(lefty), len(righty)

