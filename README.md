# Artistic Style Transfer Using Neural Algorithm


#### Dataset description:

The VGG dataset we used is the file 'imagenet-vgg-verydeep-19.mat'. This is from the website: http://www.vlfeat.org/matconvnet/pretrained/

- Our style images are from the Internet, using google image searching.

- Our content images are either from the Internet or from our past own creation.

----------------------------------------------------------------------------------------------

#### Function of each file in the project:

- Artistic_Style.py, all our functions are in this file. It's well wrapped up and easy to use. 

    To perform a style transfer task, please type lines like below:

        from Artistic_Style import ArtisticStyle

        art = ArtisticStyle()

        art.training(content="Bangkok_TH", styles=["Composition_VII"], 
                       style_weights=[1], output="output0", alpha=5, beta=100, optimizer='Adam', 
                       learning_rate=2.0, iterations=1000, original_colors=False, verbose=False)
  
    You can adjust the parameters according to your own need.

- contents folder stores all the content images

- styles folder stores all the style images

- outputs folder stores all the output images

- model folder stores all the saved models

----------------------------------------------------------------------------------------------

#### Example:

| Content | Style | Output |
|---|---|---|
| <img src="https://github.com/ws2512/nndl_final_project/raw/master/contents/Panda.jpg" width="280px" height="210px" alt="Content" > |<img src="https://github.com/ws2512/nndl_final_project/raw/master/styles/Guernica.jpg" width="280px" height="210px" alt="Style" > | <img src="https://github.com/ws2512/nndl_final_project/raw/master/outputs/Panda_output.jpg" width="280px" height="210px" alt="Output" > |
| <img src="https://github.com/ws2512/nndl_final_project/raw/master/contents/istanbul_bosphorus.jpg" width="280px" height="210px" alt="Content" > |<img src="https://github.com/ws2512/nndl_final_project/raw/master/styles/Sunrise.jpg" width="280px" height="210px" alt="Style" > | <img src="https://github.com/ws2512/nndl_final_project/raw/master/outputs/istanbul_Sunrise.jpg" width="280px" height="210px" alt="Output" > |

##### Three style transfer modes
| Original | Multiple Style Blending | Color Preservation |
|---|---|---|
| <img src="https://github.com/ws2512/nndl_final_project/raw/master/outputs/Panda_output.jpg" width="280px" height="210px" alt="Ori" > |<img src="https://github.com/ws2512/nndl_final_project/raw/master/outputs/Panda_output_2style.jpg" width="280px" height="210px" alt="Multi" > | <img src="https://github.com/ws2512/nndl_final_project/raw/master/outputs/Panda_output_preserve_color.jpg" width="280px" height="210px" alt="Color pre" > |
