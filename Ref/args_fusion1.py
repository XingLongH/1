
class args():

	# training args
	epochs = 10 #"number of training epochs, default is 2"
	batch_size = 4 #"batch size for training, default is 4"
	dataset_ir = "./MSRS/Infrared/train/MSRS"
	dataset_vi = "./MSRS/Visible/train/MSRS"

	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "./fusionmodel/train/fusionnet/"
	save_loss_dir = './fusionmodel/train/loss_fusionnet/'

	# save_fusion_model_noshort = "models/train/fusionnet_noshort/"
	# save_loss_dir_noshort = './models/train/loss_fusionnet_noshort/'
	#
	# save_fusion_model_onestage = "models/train/fusionnet_onestage/"
	# save_loss_dir_onestage = './models/train/loss_fusionnet_onestage/'

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-3 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 500"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = './fusionmodel/nestfuse/nestfuse_1e2.model'
	# resume_nestfuse = None
	# fusion net(RFN) model
	
	model_default = "./fusionmodel/train/fusionnet/6.0/Final_epoch_10_alpha_700_wir_6.0_wvi_3.0.model"
	fusion_model = './fusionmodel/rfn_twostage/'



