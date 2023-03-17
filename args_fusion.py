
class args():

	# training args
	epochs = 10 #"number of training epochs, default is 2"
	batch_size = 4 #"batch size for training, default is 4"
	dataset_ir = "../MSRS/train/Infrared/train/MSRS"
	dataset_vi = "../MSRS/train/Visible/train/MSRS"
	dataset = "/mnt/Disk1/huxinglong/data/COCO/"
    
    
	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "./change/fusion"
	save_model_dir_autoencoder="./change/autoencoder/"
	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
    
	ssim_weight = [1,10,100,1000,10000]
# 	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 10  #"number of images after which the training loss is logged, default is 500"
	resume = None

	
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = './change/autoencoder/Final_epoch_2_Mon_Jan_30_14_43_08_2023_1e2.model'	
    # test fusion
	fusion_default = "./change/fusion/Finnaly_Epoch_9.model"




