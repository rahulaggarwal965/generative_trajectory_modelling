from generative_trajectory_modelling.datasets.m3ed_dataset import M3EDDataset

m3ed_dataset = M3EDDataset("data/m3ed/spot_indoor_building_loop")
print("length: ", len(m3ed_dataset))
_, pose = m3ed_dataset[50]
print(pose)