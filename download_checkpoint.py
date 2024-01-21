import gdown

checkpoint_dir = "./checkpoints/"

gdown.download("https://drive.google.com/uc?id=13b6V0Eb9p0oFgqlM9KKmd6_x8gyCWwAE",
               checkpoint_dir + "final_run/" + "checkpoint.pth", quiet=True)
