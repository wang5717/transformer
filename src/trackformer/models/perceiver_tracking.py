from trackformer.models.detr_tracking import DETRTrackingBase as TrackingBase
from trackformer.models.perceiver_detection import PerceiverDetection


class PerceiverTracking(TrackingBase, PerceiverDetection):
    def __init__(self, tracking_kwargs, detection_model_kwargs):
        TrackingBase.__init__(self, **tracking_kwargs)
        PerceiverDetection.__init__(self, **detection_model_kwargs)
