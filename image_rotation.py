import numpy as np
import cv2
import math
import random

annotation_types = ('points', 'rects', 'np_rotated_rects', 'cv_rotated_rects', 'quadrilaterals', 'polygons')


# Definition of rotated rectangles, where box_points is a list of the four vertexes of the rectangle
# The range of the angle is [0, 360)
class RotatedRect:
    def __init__(self, center, width, height, angle, box_points):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.box_points = box_points

    def get_box_points(self):
        return self.box_points


class Rotator:
    def __init__(self, annotation_type='points', rotation_angle=(-15,15)):
        self.image = np.array((0,0))
        self.points = []
        self.rects = []
        self.np_rotated_rects = np.zeros((1,5))
        self.cv_rotated_rects = []
        self.quadrilaterals = []
        self.polygons = []
        self.annotation_type = annotation_type
        self.rotation_angle = rotation_angle

        self.angle = 0
        self.radian = 0
        self.width_increased = 0
        self.height_increased = 0
        self.new_w = 0
        self.new_h = 0

        self.check_parameters()

        self.results = [None, []]

    def check_parameters(self):
        assert isinstance(self.image, np.ndarray)
        assert self.annotation_type in annotation_types
        assert (isinstance(self.rotation_angle, tuple) and self.rotation_angle.__len__() == 2) \
               or isinstance(self.rotation_angle, int)
        if self.annotation_type == 'points':
            assert isinstance(self.points, list) or isinstance(self.points, tuple)
            if self.points.__len__() > 0:
                for point in self.points:
                    assert isinstance(point, tuple) and point.__len__() == 2
        elif self.annotation_type == 'rects':
            assert isinstance(self.rects, list) or isinstance(self.rects, tuple)
            if self.rects.__len__() > 0:
                for rect in rects:
                    assert isinstance(rect, tuple) and rect.__len__() == 4
        elif self.annotation_type == 'np_rotated_rects':
            assert isinstance(self.np_rotated_rects, np.ndarray) and self.np_rotated_rects.shape.__len__() == 2
            if self.np_rotated_rects.nonzero().__len__() != 0:
                for r in range(self.np_rotated_rects.shape[0]):
                    line = self.np_rotated_rects[r, :]
                    assert line.__len__() == 5
        elif self.annotation_type == 'cv_rotated_rects':
            assert isinstance(self.cv_rotated_rects, tuple) or isinstance(self.cv_rotated_rects, list)
            if self.cv_rotated_rects.__len__() != 0:
                for rect in self.cv_rotated_rects:
                    assert isinstance(rect, tuple) and rect.__len__() == 3
                    assert rect[0].__len__() == 2 and rect[1].__len__() == 2
                    assert isinstance(rect[2], float)
        elif self.annotation_type == 'quadrilaterals':
            assert isinstance(self.quadrilaterals, list) or isinstance(self.quadrilaterals, tuple)
            if self.quadrilaterals.__len__() != 0:
                for quadrilateral in self.quadrilaterals:
                    assert (isinstance(quadrilateral, tuple) or isinstance(quadrilateral, list))\
                           and quadrilateral.__len__() == 8
        elif self.annotation_type == 'polygons':
            assert isinstance(self.polygons, list) or isinstance(self.polygons, tuple)
            if self.polygons.__len__() != 0:
                for polygon in polygons:
                    assert isinstance(polygon, tuple) or isinstance(polygon, list)
                    for pt in polygon:
                        assert isinstance(pt, tuple) and pt.__len__() == 2

    # A method to calculate the coordinate of the points after rotating
    def rotate_point(self, point):
        assert isinstance(point, tuple) and point.__len__() == 2

        point_expanded = (point[0] + self.width_increased / 2., point[1] + self.height_increased / 2.)
        point_translated = (point_expanded[0] - self.new_w / 2., self.new_h / 2. - point_expanded[1])
        point_rotated = (point_translated[0] * math.cos(self.radian) - point_translated[1] * math.sin(self.radian),
                         point_translated[0] * math.sin(self.radian) + point_translated[1] * math.cos(self.radian))
        point_restored = (int(math.ceil(point_rotated[0] + self.new_w / 2.)),
                          int(math.ceil(-1 * point_rotated[1] + self.new_h / 2.)))

        return point_restored

    def rotate(self, image, annotation=[]):
        self.image = image
        if self.annotation_type == 'points':
            self.points = annotation
        elif self.annotation_type == 'rects':
            self.rects = annotation
        elif self.annotation_type == 'np_rotated_rects':
            self.np_rotated_rects = annotation
        elif self.annotation_type == 'cv_rotated_rects':
            self.cv_rotated_rects = annotation
        elif self.annotation_type == 'quadrilaterals':
            self.quadrilaterals = annotation
        elif self.annotation_type == 'polygons':
            self.polygons = annotation

        self.check_parameters()

        # Generate a random angle with the given range for data augmentation if the input is not an integer
        if isinstance(self.rotation_angle, int):
            angle = self.rotation_angle
        else:
            angle = random.randint(self.rotation_angle[0], self.rotation_angle[1])
        # Transform the rotation angle to [0, 360) and calculate the corresponding radian
        while angle < 0:
            angle += 360
        angle = angle % 360
        radian = angle * 3.1416 / 180.
        self.angle = angle
        self.radian = radian

        # Calculate the new width and height after rotation
        (h, w) = self.image.shape[:2]
        new_w = math.ceil(abs(h * math.sin(radian)) + abs(w * math.cos(radian)))
        new_h = math.ceil(abs(w * math.sin(radian)) + abs(h * math.cos(radian)))
        height_increment = int(math.ceil(new_h - h))
        width_increment = int(math.ceil(new_w - w))
        self.new_w = new_w
        self.new_h = new_h
        self.height_increased = height_increment
        self.width_increased = width_increment

        # Extend the edges of the initial image according to the new width and height
        image_expanded = np.zeros((int(new_h), int(new_w), 3), np.uint8)
        image_expanded[height_increment // 2:height_increment // 2 + h, width_increment // 2:width_increment // 2 + w]\
            = self.image

        # Rotate the image
        M = cv2.getRotationMatrix2D((new_h / 2., new_w / 2.), angle, 1.0)
        image_rotated = cv2.warpAffine(image_expanded, M, (int(new_w), int(new_h)))
        self.results[0] = image_rotated

        if self.annotation_type == 'points':
            # Calculate the new coordinates of the input points
            new_points = []
            for point in points:
                new_point = self.rotate_point(point)
                new_points.append(new_point)
            self.results[1] = new_points

        if self.annotation_type == 'rects':
            # Produce the rotated rect from rect
            new_rects = []
            for rect in rects:
                center = (rect[0] + rect[2] / 2., rect[1] + rect[3] / 2.)
                new_center = self.rotate_point(center)
                tl_point = (rect[0], rect[1])
                new_tl_point = self.rotate_point(tl_point)
                tr_point = (rect[0] + rect[2], rect[1])
                new_tr_point = self.rotate_point(tr_point)
                br_point = (rect[0] + rect[2], rect[1] + rect[3])
                new_br_point = self.rotate_point(br_point)
                bl_point = (rect[0], rect[1] + rect[3])
                new_bl_point = self.rotate_point(bl_point)
                new_rect = RotatedRect(new_center, rect[2], rect[3], angle,
                                       [new_tl_point, new_tr_point, new_br_point, new_bl_point])
                new_rects.append(new_rect)
            self.results[1] = new_rects

        if self.annotation_type == 'np_rotated_rects':
            # Produce the rotated rect from np_rotated_rect
            new_np_rotated_rects = np.zeros(self.np_rotated_rects.shape)
            for r in range(self.np_rotated_rects.shape[0]):
                x_min = self.np_rotated_rects[r, 0]
                y_min = self.np_rotated_rects[r, 1]
                x_max = self.np_rotated_rects[r, 2]
                y_max = self.np_rotated_rects[r, 3]
                theta = self.np_rotated_rects[r, 4]
                tl_point = (x_min, y_min)
                br_point = (x_max, y_max)
                new_tl_point = self.rotate_point(tl_point)
                new_br_point = self.rotate_point(br_point)
                new_theta = theta - angle
                new_np_rotated_rects[r, 0] = new_tl_point[0]
                new_np_rotated_rects[r, 1] = new_tl_point[1]
                new_np_rotated_rects[r, 2] = new_br_point[0]
                new_np_rotated_rects[r, 3] = new_br_point[1]
                new_np_rotated_rects[r, 4] = new_theta

                self.results[1] = new_np_rotated_rects

        if self.annotation_type == 'cv_rotated_rects':
            # Produce the rotated rect from cv_rotated_rect
            new_cv_rotated_rects = []
            for cv_rotated_rect in self.cv_rotated_rects:
                box_points = cv2.boxPoints(cv_rotated_rect)
                pt1 = (box_points[0][0], box_points[0][1])
                pt2 = (box_points[1][0], box_points[1][1])
                pt3 = (box_points[2][0], box_points[2][1])
                pt4 = (box_points[3][0], box_points[3][1])
                new_pt1 = self.rotate_point(pt1)
                new_pt2 = self.rotate_point(pt2)
                new_pt3 = self.rotate_point(pt3)
                new_pt4 = self.rotate_point(pt4)
                pts = [new_pt1, new_pt2, new_pt3, new_pt4]
                new_cv_rotated_rect = cv2.minAreaRect(np.array(pts))
                new_cv_rotated_rects.append(new_cv_rotated_rect)
            self.results[1] = new_cv_rotated_rects

        if self.annotation_type == 'quadrilaterals':
            # Produce the rotated quadrilaterals from input ones
            new_quadrilaterals = []
            for quadrilateral in self.quadrilaterals:
                point1 = (quadrilateral[0], quadrilateral[1])
                point2 = (quadrilateral[2], quadrilateral[3])
                point3 = (quadrilateral[4], quadrilateral[5])
                point4 = (quadrilateral[6], quadrilateral[7])
                new_point1 = self.rotate_point(point1)
                new_point2 = self.rotate_point(point2)
                new_point3 = self.rotate_point(point3)
                new_point4 = self.rotate_point(point4)
                new_quadrilateral = (new_point1[0], new_point1[1], new_point2[0], new_point2[1],
                                     new_point3[0], new_point3[1], new_point4[0], new_point4[1],)
                new_quadrilaterals.append(new_quadrilateral)
            self.results[1] = new_quadrilaterals

        if self.annotation_type == 'polygons':
            # Produce the rotated polygons from input ones
            new_polygons = []
            for polygon in polygons:
                new_polygon = []
                for pt in polygon:
                    new_pt = self.rotate_point(pt)
                    new_polygon.append(new_pt)
                new_polygons.append(tuple(new_polygon))
            self.results[1] = new_polygons

        return self.results[0], self.results[1]


if __name__ == '__main__':
    # Read image to be rotated
    image = cv2.imread('./test.jpg')
    image_ = image.copy()
    # Annotations formatted as points
    points = [(113, 127), (156, 127), (156, 170), (113, 170), (179, 127), (224, 127), (224, 171), (179, 171)]
    for point in points:
        cv2.circle(image_, point, 1, (0, 0, 255))
    # Annotations formatted as rects [x, y, width, height]
    rects = [(117, 188, 108, 57)]
    cv2.rectangle(image_, (rects[0][0], rects[0][1]), (rects[0][0] + rects[0][2], rects[0][1] + rects[0][3]),
                  (0, 255, 0))
    # Annotations of rotated rectangles formatted as numpy.array
    np_rotated_rects = np.zeros((1, 5))
    np_rotated_rects[0, 0] = 154
    np_rotated_rects[0, 1] = 66
    np_rotated_rects[0, 2] = 186
    np_rotated_rects[0, 3] = 90
    np_rotated_rects[0, 4] = 0

    for r in range(np_rotated_rects.shape[1]):
        x_min = int(np_rotated_rects[0, 0])
        y_min = int(np_rotated_rects[0, 1])
        x_max = int(np_rotated_rects[0, 2])
        y_max = int(np_rotated_rects[0, 3])
        theta = int(np_rotated_rects[0, 4])
        cv2.circle(image_, (x_min, y_min), 1, (255, 0, 0))
        cv2.circle(image_, (x_max, y_max), 1, (255, 0, 0))
    # Annotations of rotated rectangles formatted as opencv
    pts = [(133, 142), (136, 141), (141, 147), (139, 148), (132, 151), (137, 154), (128, 150), (141, 154)]
    cv_rotated_rect = cv2.minAreaRect(np.array(pts))
    cv_rotated_rects = [cv_rotated_rect]
    box = cv2.boxPoints(cv_rotated_rect)
    for i in range(box.__len__()):
        cv2.line(image_, (int(box[i][0]), int(box[i][1])),
                 (int(box[(i+1) % 4][0]), int(box[(i+1) % 4][1])), (255, 128, 255))
    # Annotations of quadrilaterals
    quadrilaterals = [(163, 276, 206, 300, 217, 337, 143, 316)]
    for n in range(quadrilaterals.__len__()):
        for i in range(0, 7, 2):
            pt1 = (quadrilaterals[n][i], quadrilaterals[n][i+1])
            pt2 = (quadrilaterals[n][(i+2) % 8], quadrilaterals[n][(i+3) % 8])
            cv2.line(image_, pt1, pt2, (255, 255, 0))
    # Annotations of polygons
    polygons = [((271, 286), (296, 305), (286, 337), (255, 337), (245, 305)),
                ((231, 266), (243, 291), (220, 291))]
    for n in range(polygons.__len__()):
        for i in range(polygons[n].__len__()):
            cv2.line(image_, polygons[n][i], polygons[n][(i+1) % polygons[n].__len__()], (255, 0, 255))

    cv2.imshow('initial_image', image_)

    # Define the rotater and rotate the image
    rotator1 = Rotator('points', 15)
    image_rotated, new_points = rotator1.rotate(image.copy(), points)
    rotator2 = Rotator('rects', 15)
    _, new_rects = rotator2.rotate(image.copy(), rects)
    rotator3 = Rotator('np_rotated_rects', 15)
    _, new_np_rotated_rects = rotator3.rotate(image.copy(), np_rotated_rects)
    rotator4 = Rotator('cv_rotated_rects', 15)
    _, new_cv_rotated_rects = rotator4.rotate(image.copy(), cv_rotated_rects)
    rotator5 = Rotator('quadrilaterals', 15)
    _, new_quadrilaterals = rotator5.rotate(image.copy(), quadrilaterals)
    rotator6 = Rotator('polygons', 15)
    _, new_polygons = rotator6.rotate(image.copy(), polygons)

    # Draw rotated annotations of points
    for new_point in new_points:
        cv2.circle(image_rotated, new_point, 1, (0, 0, 255))
    # Draw rotated annotations of rectangles
    for new_rect in new_rects:
        box_points = new_rect.get_box_points()
        for i in range(box_points.__len__()):
            cv2.line(image_rotated, box_points[i], box_points[(i + 1) % 4], (0, 255, 0))
    # Draw rotated annotations of rectangles formatted as numpy.array
    for r in range(new_np_rotated_rects.shape[1]):
        x_min = int(new_np_rotated_rects[0, 0])
        y_min = int(new_np_rotated_rects[0, 1])
        x_max = int(new_np_rotated_rects[0, 2])
        y_max = int(new_np_rotated_rects[0, 3])
        theta = int(new_np_rotated_rects[0, 4])
        cv2.circle(image_rotated, (x_min, y_min), 1, (255, 0, 0))
        cv2.circle(image_rotated, (x_max, y_max), 1, (255, 0, 0))
    # Draw rotated annotations of rectangles formatted as opencv
    new_cv_rotated_rects = new_cv_rotated_rects[0]
    box = cv2.boxPoints(new_cv_rotated_rects)
    for i in range(new_cv_rotated_rects.__len__()):
        cv2.line(image_rotated, (int(box[i][0]),int(box[i][1])),
                 (int(box[(i+1) % 4][0]), int(box[(i+1) % 4][1])), (255, 128, 255))
    # Draw rotated annotations of quadrilaterals
    for n in range(new_quadrilaterals.__len__()):
        for i in range(0, 7, 2):
            pt1 = (new_quadrilaterals[n][i], new_quadrilaterals[n][i+1])
            pt2 = (new_quadrilaterals[n][(i+2) % 8], new_quadrilaterals[n][(i+3) % 8])
            cv2.line(image_rotated, pt1, pt2, (255,255,0))
    # Draw rotated annotations of polygons
    for n in range(new_polygons.__len__()):
        for i in range(new_polygons[n].__len__()):
            cv2.line(image_rotated, new_polygons[n][i], new_polygons[n][(i+1) % new_polygons[n].__len__()], (255, 0, 255))
    # Show the results
    cv2.imshow('rotated_image', image_rotated)
    cv2.waitKey(0)


