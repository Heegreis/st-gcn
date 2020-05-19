# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """利用中心點的距離進行物件追蹤
    """
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def getRectFromSkeleton(self, poseData):
        if poseData['Neck'][2] != 0 and poseData['Nose'][2] != 0:
            d_x = poseData['Neck'][0] - poseData['Nose'][0]
            d_y = poseData['Neck'][1] - poseData['Nose'][1]
            radius = math.sqrt(d_x**2 + d_y**2)
            center_x = poseData['Nose'][0]
            center_y = poseData['Nose'][1]
            poseData['face_top'] = ((center_x), (center_y - radius), 1.0)
            poseData['face_down'] = ((center_x), (center_y + radius), 1.0)
            poseData['face_left'] = ((center_x - radius), (center_y), 1.0)
            poseData['face_right'] = ((center_x + radius), (center_y), 1.0)
        first = True
        for _, value in poseData.items():
            if value[2] != 0:
                if first:
                    max_x, min_x = value[0], value[0]
                    max_y, min_y = value[1], value[1]
                    first = False
                else:
                    if value[0] > max_x:
                        max_x = value[0]
                    if value[0] < min_x:
                        min_x = value[0]
                    if value[1] > max_y:
                        max_y = value[1]
                    if value[1] < min_y:
                        min_y = value[1]
        return (min_x, min_y, max_x, max_y)

    def update(self, rects):
        """更新objects的ID
        
        Args:
            rects (list): 新的關鍵點資料
        
        Returns:
            OrderedDict: 新的objects資料，(人物ID, [人物中心x座標, 人物中心y座標, 對應的關節資料索引])))
        """

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                #print(objectID)
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 3), dtype="int")

        # loop over the bounding box rectangles
        for (i, (poseData, pose_index)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            if poseData['Neck'][2] != 0:
                startX = int(poseData['Neck'][0])
                startY = int(poseData['Neck'][1])
                endX = int(poseData['Neck'][0])
                endY = int(poseData['Neck'][1])
            else:
                startX, startY, endX, endY = self.getRectFromSkeleton(poseData)
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY, pose_index)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            #print(np.array(objectCentroids), inputCentroids)

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(
                np.array(objectCentroids)[:, 0:2], inputCentroids[:, 0:2])
            #print(D)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            #print(rows)

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            #print(cols)

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            #print(usedRows)

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects
