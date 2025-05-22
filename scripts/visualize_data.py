import coopscenes
from coopscenes import Dataloader

dataset_path = "/mnt/hot_data/dataset/coopscenes_raw/seq_3"

frame = Dataloader(dataset_path)[15][30]

coopscenes.save_image(frame.vehicle.cameras.STEREO_LEFT, './raw', str(frame.frame_id))
