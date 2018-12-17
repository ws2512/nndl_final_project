# nndl_final_project

The VGG dataset we used is the file 'imagenet-vgg-verydeep-19.mat'. This is from the website: http://www.vlfeat.org/matconvnet/pretrained/

- Our style images are from the Internet, using google image searching.

- Our content images are either from the Internet or from our past own creation.

To perform a style transfer task, please type lines like below:

    from Artistic_Style import ArtisticStyle

    art = ArtisticStyle()

    art.training(content="Bangkok_TH", styles=["Composition_VII"], 
                   style_weights=[1], output="output0", alpha=5, beta=100, optimizer='Adam', 
                   learning_rate=2.0, iterations=1000, original_colors=False, verbose=False)
  
You can adjust the parameters according to your own need.
