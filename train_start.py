from train_noddp import train_models
train_models(epochs = 50,\
            model_name ='Unet',\
            perceptual_loss =['vgg19'],\
            channel_loss = ['dark_channel_loss'],\
            structural_loss=['ssim'],\
            dataset_name="TURBID", dataset_path="data")
#LSUI
