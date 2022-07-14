# Ke Chen
# knutchen@ucsd.edu
# Zero-shot Audio Source Separation via Query-based Learning from Weakly-labeled Data
# The configuration file

# for model training
exp_name = "exp_zs_asp_full" # the saved ckpt prefix name of the model 
workspace = "/home/Research/ZS_ASP/" # the folder of your code
dataset_path = "/home/Research/ZS_ASP/data/audioset" # the dataset path
index_type = "full_train"
idc_path = "/home/Research/ZS_ASP/" # the folder of audioset class count files
balanced_data = True

# trained from a checkpoint, or evaluate a single model 
resume_checkpoint = None
# "/home/Research/ZS_ASP/model_backup/zeroshot_asp_full.ckpt"

loss_type = "mae"

gather_mode = False
debug = False

classes_num = 527
eval_list = [] # left blank to preserve all classes, otherwise will filter the specified classes
# [15, 63, 81, 184, 335, 449, 474, 348, 486, 4] # randomly generated from the 527-classes for held-out evaludation


batch_size = 16 * 8   # batch size per GPU x GPU number , default is 16 x 8 = 128
learning_rate = 1e-3 # 3e-4 is also workable
max_epoch = 100
num_workers = 3
lr_scheduler_epoch = [90, 110]
latent_dim = 2048

# for signal processing
sample_rate = 32000
clip_samples = sample_rate * 10 # audio_set 10-sec clip
segment_frames = 200 
hop_samples = 320
random_seed = 12412 # 444612 1536123 12412
random_mode = "one_class" # "no_random, one_class, random, order", one class is the best

# for evaluation
musdb_path = "/home/Research/ZS_ASP/data/musdb-wav/" # musdb download folder
testavg_path = "/home/Research/ZS_ASP/data/musdb30-train-32000fs.npy" # the processed training set (to get the latent query)
testset_path = "/home/Research/ZS_ASP/data/musdb-test-32000fs.npy" # the processed testing set (to calculate the performance)
test_key = ["vocals", "drums", "bass", "other"] # four tracks for musdb, and your named track for other inference
test_type = "mix"
infer_type = "mean"
energy_thres = 0.1
wave_output_path = "/home/Research/ZS_ASP/wavoutput" # output folder
using_wiener = True # use wiener filter or not (default: True)
using_whiting = False # use whiting or not (default: False)

# weight average
wa_model_folder = "/home/Research/ZS_ASP/version_3/checkpoints/"
wa_model_path = "zs_wa.ckpt"

# for inference
inference_file = "/home/Research/ZS_ASP/data/pagenini.wav" # an audio file to separate
inference_query = "/home/Research/ZS_ASP/data/query" # a folder containing all samples for obtaining the query
overlap_rate = 0.0 # [0.0, 1.0), 0 to disabled, recommand 0.5 for 50% overlap. Overlap will increase computation time and improve result quality
