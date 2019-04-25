# Rotator_for_angle
A method to rotate some images and their one type of annotations for a fixed angle or a fixed range of angles.

Requirements
-------
Python3

OpenCV3.4

Numpy1.15

Details
-------
At first, it is necessary to import the Rotator by: from image_rotation import Rotator

Then you need to build a rotator for one type of annotation and a rotation angle or a range of rotation angles. The type should be choosed from the list:

‘points’ is a list of some point annotations on the image formatted as tuples (x,y). It can be empty while the output will be None. An example of the parameter is [(113, 127), (156, 127), (156, 170)]

‘rects’ is a list of some rectangles on the image formatted as tuples (x,y,width,height). It can be empty while the output will be None. An example of the parameter is [(117, 188, 108, 57), (118, 189, 109, 58)]

'np_rotated_rects' is some rotated rectangles on the image formated as the type of numpy.array. Each row represents a rectangle and the columns represents x_min, y_min, x_max, y_max and rotation angle respectively. It can be empty while the output will be None.

'cv_rotated_rects' is a list of rotated rectangles formatted as those in opencv. Each element has three subelements, in which the first tuple means the center of the rectangle, the second tuple includes width and height, and the third number is the rotation angle. It can be empty while the output will be None. The structure is ((center_x, center_y), (width, height), angle)

'quadrilaterals' is a list of some quadrilaterals. One quadrilateral is a tuple with 8 numbers, which means the 4 vertexes of it. The first two numbers represents the x and y of the first vertex and the rest may be deduced by analogy. It can be empty while the output will be None.

'polygons' is a list of polygons. Each element contains some tuples which represents the coordinates of the vertixes. It can be empty while the output will be None.

Besides, a parameter that represents the angle should be provided. It can be an integer or a tuple with length 2:

-When it is an integer, the inputs will be rotated rotation_angle degrees

-When it is a tuple with length 2, the rotation angle will be generated randomly with a value between [rotation_angle[0], rotation_angle[1]]

And 'expand_edge' is a boolean variable, when its value is True, the image will be expanded before rotated to ensure that no pixel of the image will be abandoned. The process will look like this:![1](https://github.com/Alpaca07/Rotator_for_angle/blob/master/examples/sketch1.png)

If it is set to False, the rotation will not change the size of the image. The process will look like this:![2](https://github.com/Alpaca07/Rotator_for_angle/blob/master/examples/sketch2.png)

Then a rotator has been created. You can invoke the method rotate(image, annotation) to rotate the image and its annotation. The method will return two objects in which the first is the rotated image and the second is the annotations after rotation.

Tips: While using this class, you can transform numpy.ndarray to list with .tolist() or transform list to numpy.ndarray with numpy.array(list).

It is worth noting that the method is not suitable for rotations with a large angle as there are accumulative errors while calculating floating numbers.
