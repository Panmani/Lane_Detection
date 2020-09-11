# GCN
This pipeline made a big improvement on detecting lane markings in a challenging video where the brightness changes rapidly and the road is very curvy. Published paper: [Vision-Based Lane Detection and Lane-Marking Model Inference: A Three-Step Deep Learning Approach](https://ieeexplore.ieee.org/document/8701850Â)

# Result
The lane marking regions are green when the model is confident that the lane markings are inside the regions; a region becomes yellow when the marking is not clear enough and the model infers this lane marking from the lane marking on the other side; when they become red, it means that the model cannot find enough pixels for both lane markings. If available, Ego-motion can be used to update the lane markings (future work).

> Final result video

[![Lane marking detection comparison](http://img.youtube.com/vi/Eb-_uPb5D9M/0.jpg)](https://www.youtube.com/watch?v=Eb-_uPb5D9M "Lane marking detection comparison")

> Comparison with the baseline algorithm

[![Lane marking detection](http://img.youtube.com/vi/i3MAK13_ki0/0.jpg)](https://www.youtube.com/watch?v=i3MAK13_ki0 "Lane marking detection")


# Usage
Get the perspective transform information
```
$ python get_perspective_transform.py
```

Detect lane-markings in a video
```
$ python find_lane.py
```

# Dependency
* Tensorflow
* Numpy
* CV2
* Moviepy
* Scikit-image
* Scipy
1. Convolutional Patch Networks with Spatial Prior for Road Detection and Urban Scene Understanding: http://cvjena.github.io/cn24/

2. You Only Look Once: Unified, Real-Time Object Detection (for detecting vehicles): https://github.com/JunshengFu/vehicle-detection

> Note: Given the obsolescence of the project, it is expected to be hard to run the code using the current versions of the packages
