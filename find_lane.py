import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
import math
from moviepy.editor import VideoFileClip
from scipy.signal import convolve2d
from skimage.util.shape import view_as_windows
from yolo_pipeline import *
from helper import *
import settings

dnn_noise_reduction = True
# dnn_result = VideoFileClip("input_videos/dnn_result.mp4")

class Lane:
	def __init__(self, img_size, unwarped_size, cam_matrix, dist_coeffs, transform_matrix, pixels_per_meter):
		self.cam_matrix = cam_matrix
		self.dist_coeffs = dist_coeffs
		self.img_size = img_size
		self.unwarped_size = unwarped_size
		self.M = transform_matrix
		self.cn23_out = None

		# Lane-marking model inference
		self.prev_leftx_fit = None
		self.prev_rightx_fit = None
		self.margin_increase = [0, 0]
		self.margin_color = [(0, 255, 0), (0, 255, 0)]
		self.model_age = 0
		self.curve_strength = [0, 0]
		self.intersect = 1000
		self.prev_pixel_count = [0, 0]
		self.prev_pixel_count_x = [0, 0]
		self.prev_pixel_count_y = [0, 0]
		self.lane_posi = None
		self.w_ratio = None
		self.warning_msg = ""
		self.warning_color = green_color

		# Frame Number
		self.frame_count = 0

	def unwarp(self, img):
		return cv2.warpPerspective(img, self.M, self.unwarped_size)

	def warp(self, img):
		return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.WARP_INVERSE_MAP)

	def reduce_noise_CPN(self, img, features):
		# Execution
		cv2.imwrite('temp/cur_frame.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		os.system("../../cn24/build/classifyImage ../../cn24/example/kitti_um_road.set ../../cn24/example/kitti.net ../../cn24/example/kitti_pretrained.Tensor ./temp/cur_frame.jpg ./temp/output.jpg")
		self.cn23_out = cv2.imread('temp/output.jpg')

		# lane_detected_img = dnn_result.get_frame(self.frame_count*1.0/dnn_result.fps)
		unwarped_lane_detected_img = self.unwarp(self.cn23_out)
		unwarped_lane_detected_img_gray = cv2.cvtColor( unwarped_lane_detected_img, cv2.COLOR_RGB2GRAY )

		# Calculate new mask
		patch_w = 10
		evidence_threshold = 75.

		nonzero_pixels = np.nonzero(features)
		padded_lane_detected = np.pad(unwarped_lane_detected_img_gray, \
			((patch_w, patch_w), (patch_w, patch_w)), \
			'constant', constant_values=0)
		padded_lane_detected_patches = view_as_windows(padded_lane_detected, (patch_w * 2 + 1, patch_w * 2 + 1))
		pixel_patches = padded_lane_detected_patches[nonzero_pixels[0], nonzero_pixels[1]]
		pixel_patches = pixel_patches / 255.
		pixel_evidence = np.sum(pixel_patches, axis = 2)
		pixel_evidence = np.sum(pixel_evidence, axis = 1)
		evidence_mask = np.nonzero(pixel_evidence > evidence_threshold)
		nonzero_pixels = np.swapaxes(np.array(nonzero_pixels), 0, 1)
		lane_pixels = nonzero_pixels[pixel_evidence > evidence_threshold]
		lane_pixels = np.swapaxes(lane_pixels, 0, 1)
		new_features = np.zeros(features.shape)
		new_features[lane_pixels[0],lane_pixels[1]] = 1.
		new_features = np.array(new_features, dtype=np.uint8)

		return new_features

	def reduce_noise_YOLO(self, img):
		# Vehicle Removal using YOLO
		detect_from_file(yolo, img)
		results = yolo.result_list
		for i in range(len(results)):
			car_x = int(results[i][1])
			car_y = int(results[i][2])
			car_w = int(results[i][3])//2
			car_h = int(results[i][4])//2
			ratio = car_w * 1. / car_h
			if ratio < 1.7:
				if car_x < self.img_size[0] / 2:
					img[car_y - int(car_h * 1.5):car_y + car_h, car_x - car_w : car_x + int(car_w * 1.5)] = [0, 0, 0]
				else:
					img[car_y - int(car_h * 1.5):car_y + car_h, car_x - int(car_w * 1.5) : car_x + car_w] = [0, 0, 0]					

		return img

	def find(self, frame):
		def scaled(x, maximum=255.0, outtype=np.uint8):
			return (x*maximum/np.max(x)).astype(outtype)

		original_frame = frame.copy()
		frame = cv2.undistort(frame, self.cam_matrix, self.dist_coeffs)
		frame = self.reduce_noise_YOLO(frame)

		unwarped_frame = self.unwarp(frame)

		features_with_noise = extract_feature(unwarped_frame)
		features = self.reduce_noise_CPN(frame, features_with_noise)

		num_segments = 48
		segment_height = settings.UNWARPED_SIZE[1] // num_segments
		if self.prev_leftx_fit is None or self.prev_rightx_fit is None:
			window_centroids = find_window_centroids(features,w,h)
			lefty,leftx,righty,rightx,leftw,rightw,out_img = \
					find_lane_pixels_with_window_centroids(features, window_centroids,w,h)
			regr,make_feature,(fity,fit_leftx,fit_rightx) = \
					fit_lane(lefty, leftx, righty, rightx, leftw,rightw, self.img_size[1], self.img_size[0])
			out_img[lefty, leftx] = [255, 0, 0]
			out_img[righty, rightx] = [0, 0, 255]
			self.prev_leftx_fit = fit_leftx
			self.prev_rightx_fit = fit_rightx
			self.prev_pixel_count_x = [len(np.unique(leftx)) // segment_height, len(np.unique(rightx)) // segment_height]
			self.prev_pixel_count_y = [len(np.unique(lefty)) // segment_height, len(np.unique(righty)) // segment_height]
			self.prev_pixel_count = [self.prev_pixel_count_x[0] + self.prev_pixel_count_y[0], \
									self.prev_pixel_count_x[1] + self.prev_pixel_count_y[1]]
			self.model_age = 0
		else:
			lefty,leftx,righty,rightx,leftw, rightw, out_img = \
					find_lane_pixels_with_previous_fit(features,self.prev_leftx_fit,self.prev_rightx_fit, self.margin_increase)
			regr,make_feature,(fity,fit_leftx,fit_rightx) = \
					fit_lane(lefty, leftx, righty, rightx, leftw, rightw, self.img_size[1], self.img_size[0])

			# Monitor
			if fit_leftx is not None and fit_rightx is not None:
				self.lane_posi = fit_leftx[-1], fit_rightx[-1]
				lane_width_threshold = (0.5, 6)
				lane_width = (fit_rightx - fit_leftx)*xm_per_pix
				max_lane_width, min_lane_width = max(lane_width), min(lane_width)
				self.w_ratio = max_lane_width/min_lane_width
				self.curve_strength = [np.max(fit_leftx) - np.min(fit_leftx), \
										np.max(fit_rightx) - np.min(fit_rightx)]
				self.intersect = np.abs(fit_rightx - fit_leftx)

			use_prev_threshold = 35
			max_margin_increase = 10
			per_margin_increase = 2
			recover_threshold = 35
			model_age_threshold = 15
			curve_strength_threshold = 75.
			intersect_threshold = [15., 200]
			cur_pixel_count_x = [len(np.unique(leftx)) // segment_height, len(np.unique(rightx)) // segment_height]
			cur_pixel_count_y = [len(np.unique(lefty)) // segment_height, len(np.unique(righty)) // segment_height]
			cur_pixel_count_xy = [cur_pixel_count_x[0] + cur_pixel_count_y[0], cur_pixel_count_x[1] + cur_pixel_count_y[1]]

			if cur_pixel_count_xy[0] < use_prev_threshold and cur_pixel_count_xy[1] < use_prev_threshold:
				if self.curve_strength[0] < curve_strength_threshold \
							and self.curve_strength[1] < curve_strength_threshold:
					fit_leftx = self.prev_leftx_fit
					fit_rightx = self.prev_rightx_fit
					if self.margin_increase[0] < max_margin_increase:
						self.margin_increase[0] += per_margin_increase
					if self.margin_increase[1] < max_margin_increase:
						self.margin_increase[1] += per_margin_increase
					self.warning_msg = "Model Not Updated, using previous model"
					self.warning_color = red_color
					self.margin_color = [red_color, red_color]
					self.model_age += 1
					if self.model_age > model_age_threshold:
						fit_leftx -= fit_leftx[-1] - 400.
						fit_rightx -= fit_rightx[-1] - 500.
				else:
					curve_threshold = 20
					if fit_leftx is not None and fit_rightx is not None:
						if cur_pixel_count_xy[0] < curve_threshold and cur_pixel_count_xy[1] < curve_threshold:
							fit_leftx = self.prev_leftx_fit
							fit_right = self.prev_rightx_fit

							if self.margin_increase[0] < max_margin_increase:
								self.margin_increase[0] += per_margin_increase
							if self.margin_increase[1] < max_margin_increase:
								self.margin_increase[1] += per_margin_increase
							self.warning_msg = "At a curve, using previous model"
							self.warning_color = red_color
							self.margin_color = [red_color, red_color]
							self.model_age += 1
						elif cur_pixel_count_xy[0] >= curve_threshold and cur_pixel_count_xy[1] < curve_threshold:
							lane_w = 100.
							prev_w = (self.prev_rightx_fit[-1] - self.prev_leftx_fit[-1])
							if 70 < prev_w and prev_w < 120:
								 lane_w = prev_w
							fit_rightx = fit_leftx + lane_w

							self.margin_increase[0] = 0
							if self.margin_increase[1] < max_margin_increase:
								self.margin_increase[1] += per_margin_increase
							self.warning_msg = "At a curve, Using left lane marking only"
							self.warning_color = yellow_color
							self.margin_color = [(0, 255, 0), yellow_color]

						elif cur_pixel_count_xy[0] < curve_threshold and cur_pixel_count_xy[1] >= curve_threshold:
							lane_w = 100.
							prev_w = (self.prev_rightx_fit[-1] - self.prev_leftx_fit[-1])
							if 70 < prev_w and prev_w < 120:
								 lane_w = prev_w
							fit_leftx = fit_rightx - lane_w

							self.margin_increase[1] = 0
							if self.margin_increase[0] < max_margin_increase:
								self.margin_increase[0] += per_margin_increase
							self.warning_msg = "At a curve, Using right lane marking only"
							self.warning_color = yellow_color
							self.margin_color = [yellow_color, (0, 255, 0)]
						else:
							self.warning_msg = ""
							self.warning_color = green_color

			elif cur_pixel_count_xy[0] >= use_prev_threshold and cur_pixel_count_xy[1] < use_prev_threshold:
				if self.curve_strength[1] < curve_strength_threshold \
						or np.min(self.intersect) < intersect_threshold[0] \
						or np.max(self.intersect) > intersect_threshold[1]:
					lane_w = 100.
					prev_w = (self.prev_rightx_fit[-1] - self.prev_leftx_fit[-1])
					if 70 < prev_w and prev_w < 120:
						 lane_w = prev_w
					fit_rightx = fit_leftx + lane_w
					self.margin_increase[0] = 0
					if self.margin_increase[1] < max_margin_increase:
						self.margin_increase[1] += per_margin_increase
					self.warning_msg = "Using left lane marking only"
					self.warning_color = yellow_color
					self.margin_color = [(0, 255, 0), yellow_color]
				else:
					self.margin_increase = [0, 0]
					self.margin_color = [(0, 255, 0), (0, 255, 0)]
					self.warning_msg = ""
					self.warning_color = green_color

			elif cur_pixel_count_xy[0] < use_prev_threshold and cur_pixel_count_xy[1] >= use_prev_threshold:
				if self.curve_strength[0] < curve_strength_threshold \
						or np.min(self.intersect) < intersect_threshold[0] \
						or np.max(self.intersect) > intersect_threshold[1]:
					lane_w = 100.
					prev_w = (self.prev_rightx_fit[-1] - self.prev_leftx_fit[-1])
					if 70 < prev_w and prev_w < 120:
						 lane_w = prev_w
					fit_leftx = fit_rightx - lane_w
					self.margin_increase[1] = 0
					if self.margin_increase[0] < max_margin_increase:
						self.margin_increase[0] += per_margin_increase
					self.warning_msg = "Using right lane marking only"
					self.warning_color = yellow_color
					self.margin_color = [yellow_color, (0, 255, 0)]
				else:
					self.margin_increase = [0, 0]
					self.margin_color = [(0, 255, 0), (0, 255, 0)]
					self.warning_msg = ""
					self.warning_color = green_color

			else:
				self.margin_increase = [0, 0]
				self.margin_color = [(0, 255, 0), (0, 255, 0)]
				if cur_pixel_count_xy[0] > recover_threshold and cur_pixel_count_xy[1] > recover_threshold:
					self.model_age = 0
				else:
					self.model_age -= 2
					if self.model_age < 0: self.model_age = 0
				self.warning_msg = ""
				self.warning_color = green_color

			if fit_leftx is None or fit_rightx is None:
				fit_leftx = self.prev_leftx_fit
				fit_right = self.prev_rightx_fit
				self.warning_msg = "No lane pixels, using previous model"
				self.warning_color = red_color
				self.margin_color = [red_color, red_color]
				self.model_age += 1
				if self.margin_increase[0] < max_margin_increase:
					self.margin_increase[0] += per_margin_increase
				if self.margin_increase[1] < max_margin_increase:
					self.margin_increase[1] += per_margin_increase

			if self.model_age > 0:
				l_change = (fit_leftx - self.prev_leftx_fit) * cur_pixel_count_xy[0] / recover_threshold / 3.
				fit_leftx = self.prev_leftx_fit + l_change
				r_change = (fit_rightx - self.prev_rightx_fit) * cur_pixel_count_xy[1] / recover_threshold / 3.
				fit_rightx = self.prev_rightx_fit + r_change

			if fit_leftx is not None and fit_rightx is not None:
				out_img = draw_poly(out_img, fit_leftx, fit_rightx, fity, self.margin_increase, self.margin_color)
			out_img[lefty, leftx] = [255, 0, 0]
			out_img[righty, rightx] = [0, 0, 255]

		# Annotate of information about the curvature of the lanes onto the image
		font = cv2.FONT_HERSHEY_SIMPLEX

		# Driver's View
		lane_img = np.zeros(settings.UNWARPED_SIZE + (3,))
		left_boundary = np.array([np.transpose(np.vstack([fit_leftx, np.arange(len(fit_leftx))]))])
		right_boundary = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, np.arange(len(fit_leftx))])))])
		boundaries = np.hstack((left_boundary, right_boundary))
		cv2.fillPoly(lane_img, np.int_([boundaries]), self.warning_color)
		if self.margin_color[0] == (0, 255, 0):
			cv2.polylines(lane_img, [np.int_(list(zip(fit_leftx, np.arange(len(fit_leftx)))))], \
				False, (255, 0, 0), thickness=5)
		else:
			cv2.polylines(lane_img, [np.int_(list(zip(fit_leftx, np.arange(len(fit_leftx)))))], \
				False, self.margin_color[0], thickness=5)
		
		if self.margin_color[1] == (0, 255, 0):
			cv2.polylines(lane_img, [np.int_(list(zip(fit_rightx, np.arange(len(fit_rightx)))))], \
				False, (0, 0, 255), thickness=5)
		else:
			cv2.polylines(lane_img, [np.int_(list(zip(fit_rightx, np.arange(len(fit_rightx)))))], \
				False, self.margin_color[1], thickness=5)
		lane_img = self.warp(lane_img)
		drivers_view = cv2.addWeighted(original_frame, 1, np.uint8(lane_img), 0.5, 0)

		# Lane-marking Models View
		cv2.putText(out_img, str(self.prev_pixel_count), (40,20), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.prev_pixel_count_x), (40,50), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.prev_pixel_count_y), (40,80), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.lane_posi), (40,110), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.w_ratio), (40,140), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.model_age), (40,170), font, 0.6, green_color, 2)
		cv2.putText(out_img, str(self.curve_strength), (40,200), font, 0.6, green_color, 2)
		cv2.putText(out_img, str([np.min(self.intersect), np.max(self.intersect)]), (40,230), font, 0.6, green_color, 2)
		cv2.putText(out_img, self.warning_msg, (40,260), font, 0.75, self.warning_color, 2)

		upper = np.hstack([cv2.resize(drivers_view, (1280, 720)),
							cv2.resize(out_img, (1280, 720))])

		# -----------------------Lower Part-----------------------
		cv2.putText(unwarped_frame, 'Inverse perspective view', (40,90), font, 1, green_color, 5)

		l2 = cv2.cvtColor(features_with_noise * 255, cv2.COLOR_GRAY2RGB)
		l4 = cv2.cvtColor(features * 255, cv2.COLOR_GRAY2RGB)

		lower = np.hstack([cv2.resize(unwarped_frame, (640, 360)), \
							cv2.resize(l2, (640, 360)), \
							cv2.resize(self.cn23_out, (640, 360)), \
							cv2.resize(l4, (640, 360))])
		
		tiled = np.vstack([upper, lower])
		cv2.imwrite('temp/cur_tiled.png', cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR))

		self.frame_count += 1

		return tiled

if __name__ == "__main__":

	with open(settings.CALIB_FILE_NAME, 'rb') as f:
		calib_data = pickle.load(f)
	cam_matrix = calib_data["cam_matrix"]
	dist_coeffs = calib_data["dist_coeffs"]

	with open(settings.PERSPECTIVE_FILE_NAME, 'rb') as f:
		perspective_data = pickle.load(f)
	perspective_transform = perspective_data["perspective_transform"]
	pixels_per_meter = perspective_data['pixels_per_meter']
	orig_points = perspective_data["orig_points"]

	# Load input videos and detect lane
	input_path = 'input_videos'
	input_video_file = 'harder_challenge_video.mp4'
	# input_video_file = 'challenge_video.mp4'
	output_path = 'output_videos'
	output = os.path.join(output_path, "new_fitting_algo.mp4")

	lane = Lane(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, cam_matrix, dist_coeffs,
				perspective_transform, pixels_per_meter)

	clip = VideoFileClip(os.path.join(input_path, input_video_file))
	result_clip = clip.fl_image(lambda x: lane.find(x))
	result_clip.write_videofile(output, audio=False)

	# img = cv2.imread('test_images/test3c.jpg')
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# output_img = lane.find(img)
	# plt.imshow(output_img)
	# plt.show()

