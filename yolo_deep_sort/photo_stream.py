import glob
import os
import time
import cv2

class PhotoStream:
    def __init__(self, folder, delete=False):
        self.folder = folder
        self.delete = delete
        if not os.path.exists(self.folder):
            raise Exception('Folder not found')
        if not self.folder.endswith("/"):
            self.folder += "/"
        self.photos = glob.glob(self.folder + "*.jpg")
        self.photos.sort()
        self.photo_idx = 0


    def next(self):
        start = time.time()
        new_list = False
        while self.photo_idx >= len(self.photos):
            new_list = True
            self.photos = glob.glob(self.folder + "*.jpg")
            if time.time() - start > 15:
                raise Exception('Input stopped')
        if new_list:
            self.photos.sort()
        image = cv2.imread(self.photos[self.photo_idx])
        if self.delete:
            os.remove(self.photos[self.photo_idx])
        self.photo_idx += 1
        return image
