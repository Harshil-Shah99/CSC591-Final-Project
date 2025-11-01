# Efficient Super-Resolution: Optimizing Real-ESRGAN with Pruning and Quantization
* [SRCNN](https://github.com/Harshil-Shah99/CSC591-Final-Project/blob/main/SRCNN.ipynb) contains all the python scripts to build a basic Super resolution model from scratch, train and evaluate the model with creating a onnx file. 
* [model.onnx](https://github.com/Harshil-Shah99/CSC591-Final-Project/blob/main/model.onnx) is a the onnx file created from [SRCNN](https://github.com/Harshil-Shah99/CSC591-Final-Project/blob/main/SRCNN.ipynb) model. 

To run the code for the Real-ESRGAN, clone the git repository at https://github.com/xinntao/Real-ESRGAN

Go to /Real-ESRGAN/realesrgan/
Open the file 'utils.py'
At line 75, paste the following code:

```
        import nni
        from nni.compression.pytorch.pruning import L1NormPruner
        from nni.compression.pytorch.pruning import LevelPruner
        config_list = [{
              'sparsity_per_layer': 0.5,
              'op_types': ['Conv2d']
          }]
        pruner = LevelPruner(self.model, config_list)
        masked_model, masks = pruner.compress()
        self.model = masked_model

        x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
        torch.onnx.export(
                self.model,  # model being run
                x.half(),  # model input
                "./model.onnx",  # where to save the model
                do_constant_folding=True,
                input_names=['input'],  # the model's input names (an arbitrary string)
                output_names=['output'],  # the model's output names (an arbitrary string)
                opset_version=11  # XGen supports 11 or 9
            )
```

You can edit the nni code to make optimizations as you desire

To test with speedup, you can add:
 	  pruner._unwrap_model()
        from nni.compression.pytorch.speedup import ModelSpeedup
        ModelSpeedup(self.model, torch.rand(1, 3, 28, 28).half().cuda(), masks).speedup_model()

Finally, run the following in the terminal:
```
!python inference_realesrgan.py -n RealESRGAN_x4plus -i upload --outscale 3.5
```
Alternatively, you can use the .ipynb code file in this repository and run it either locally or on google colab.
Here is our Google Colab Notebook:
https://colab.research.google.com/drive/1IccOVepCpXX5IxKwfM18NsOwyQvTKvua?usp=sharing

