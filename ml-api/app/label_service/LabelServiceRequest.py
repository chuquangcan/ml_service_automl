from dataclasses import dataclass


class ProjectType(object):
    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"
    TEXT_CLASSIFICATION = "TEXT_CLASSIFICATION"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    IMAGE_SEGMENTATION = "IMAGE_SEGMENTATION"


PROJECT_TYPES = ProjectType()


@dataclass
class DataUploadType:
    IMAGE_LABELED_FOLDER: str = "IMAGE_LABELED_FOLDER"
    IMAGE_UNLABELED: str = "IMAGE_UNLABELED"