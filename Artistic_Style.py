import numpy as np
import scipy.io 
import scipy.misc
import tensorflow as tf 
import cv2
import time

class ArtisticStyle():

    def __init__(self):
        self.vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat') # link: http://www.vlfeat.org/matconvnet/pretrained/
        self.img_width = 800
        self.img_height = 600
        self.channels = 3
        self.content_dir = 'contents/'
        self.style_dir = 'styles/'
        self.output_dir = 'outputs/'


    def get_weights(self, layer):
        W = self.vgg['layers'][0][layer][0][0][2][0][0] 
        b = self.vgg['layers'][0][layer][0][0][2][0][1]
        return W, b

    def conv2d(self, prev_layer, layer):
        W, b = self.get_weights(layer)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        conv2d_layer = tf.nn.conv2d(prev_layer, W, strides=[1, 1, 1, 1], padding='SAME') + b
        return conv2d_layer

    def relu(self, conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def avg_pool(self, prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def load_vgg(self):
        model = {}
        model["input"] = tf.Variable(np.zeros((1, self.img_height, self.img_width, self.channels), dtype=np.float32))

        # group 1
        model["conv1_1"] = self.conv2d(model["input"], 0)
        model["relu1_1"] = self.relu(model["conv1_1"])
        model["conv1_2"] = self.conv2d(model["relu1_1"], 2)
        model["relu1_2"] = self.relu(model["conv1_2"])
        model["avg_pool1"] = self.avg_pool(model["relu1_2"])

        # group 2
        model["conv2_1"] = self.conv2d(model["avg_pool1"], 5)
        model["relu2_1"] = self.relu(model["conv2_1"])
        model["conv2_2"] = self.conv2d(model["relu2_1"], 7)
        model["relu2_2"] = self.relu(model["conv2_2"])
        model["avg_pool2"] = self.avg_pool(model["relu2_2"])

        # group 3
        model["conv3_1"] = self.conv2d(model["avg_pool2"], 10)
        model["relu3_1"] = self.relu(model["conv3_1"])
        model["conv3_2"] = self.conv2d(model["relu3_1"], 12)
        model["relu3_2"] = self.relu(model["conv3_2"])
        model["conv3_3"] = self.conv2d(model["relu3_2"], 14)
        model["relu3_3"] = self.relu(model["conv3_3"])
        model["conv3_4"] = self.conv2d(model["relu3_3"], 16)
        model["relu3_4"] = self.relu(model["conv3_4"])
        model["avg_pool3"] = self.avg_pool(model["relu3_4"])

        # group 4
        model["conv4_1"] = self.conv2d(model["avg_pool3"], 19)
        model["relu4_1"] = self.relu(model["conv4_1"])
        model["conv4_2"] = self.conv2d(model["relu4_1"], 21)
        model["relu4_2"] = self.relu(model["conv4_2"])
        model["conv4_3"] = self.conv2d(model["relu4_2"], 23)
        model["relu4_3"] = self.relu(model["conv4_3"])
        model["conv4_4"] = self.conv2d(model["relu4_3"], 25)
        model["relu4_4"] = self.relu(model["conv4_4"])
        model["avg_pool4"] = self.avg_pool(model["relu4_4"])

        # group 5
        model["conv5_1"] = self.conv2d(model["avg_pool4"], 28)
        model["relu5_1"] = self.relu(model["conv5_1"])
        model["conv5_2"] = self.conv2d(model["relu5_1"], 30)
        model["relu5_2"] = self.relu(model["conv5_2"])
        model["conv5_3"] = self.conv2d(model["relu5_2"], 32)
        model["relu5_3"] = self.relu(model["conv5_3"])
        model["conv5_4"] = self.conv2d(model["relu5_3"], 34)
        model["relu5_4"] = self.relu(model["conv5_4"])
        model["avg_pool5"] = self.avg_pool(model["relu5_4"])

        return model

    def load_image(self, img_dir, img_name):
        image = scipy.misc.imread(img_dir+img_name+".jpg")
        image = scipy.misc.imresize(image, (self.img_height, self.img_width))
        image = image[np.newaxis, :, :, :]
        image = np.copy(image).astype('float32')
        image -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) #Subtracts the mean RGB value
        return image

    def save_image(self, output_dir, img_name, image):
        image += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        image = image[0]
        image = np.clip(image, 0, 255).astype('uint8')
        scipy.misc.imsave(output_dir+img_name+".jpg", image)

    def calc_content_loss(self, p, x):
        M = p.shape[1] * p.shape[2] # height * width
        N = p.shape[3] # # of filters
        #loss = 0.5 * tf.reduce_sum(tf.pow(x - p, 2))  #original way in the paper
        loss = (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
        return loss

    def content_loss(self, sess, model):
        '''
        return content loss
        '''
        return self.calc_content_loss(sess.run(model['relu4_2']), model['relu4_2']) #relu later

    def gram_matrix(self, x, M, N):
        F = tf.reshape(x, (M, N))
        G = tf.matmul(tf.transpose(F), F)
        return G

    def calc_style_loss(self, a, x):
        M = a.shape[1] * a.shape[2] # height * width
        N = a.shape[3] # # of filters
        A = self.gram_matrix(a, M, N) # gram matrix of original image
        G = self.gram_matrix(x, M, N)  # gram matrix of generated image
        loss = (1 / (4 * (N**2) * (M**2))) * tf.reduce_sum(tf.pow((G - A), 2))
        return loss

    def style_loss(self, sess, model, style_images, style_weights):
        '''
        return style loss
        '''
        total_style_loss = 0
        layers = [("relu1_1",0.2), ("relu2_1",0.2), ("relu3_1",0.2), ("relu4_1",0.2), ("relu5_1",0.2)] #relu later
        for img, weight in zip(style_images, style_weights):
            sess.run(model["input"].assign(img))
            E = [self.calc_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in layers]
            W = [w for _, w in layers]
            style_loss = sum([E[l]*W[l] for l in range(len(layers))])
            total_style_loss += style_loss * weight
        return total_style_loss

    def add_noise(self, content_img, noise_ratio=0.6):
        """
        Add noise to the content image
        """
        noise_img = np.random.uniform(-20, 20, (1, self.img_height, self.img_width, self.channels)).astype('float32')
        # white noise image  
        input_img = noise_img*noise_ratio + content_img*(1 - noise_ratio)
        return input_img

    def get_optimizer(self, optimizer, learning_rate):
        if optimizer == "Adam":
            return tf.train.AdamOptimizer(learning_rate)
        elif optimizer == "Adagrad":
            return tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == "Adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == "Rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate)
        elif optimizer == "Sgd":
            return tf.train.GradientDescentOptimizer(learning_rate)

    
    def keep_original_colors(self, content_image, output_image):
    
        # Preprocessing for content image
        content_image = np.copy(content_image).astype('float64')
        content_image += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        content_image = content_image[0]
        content_image = np.clip(content_image, 0, 255).astype('uint8')
        content_image = content_image[...,::-1]  #RGB to BGR

        # Preprocessing for output image    
        output_image = np.copy(output_image).astype('float64')
        output_image += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
        output_image = output_image[0]
        output_image = np.clip(output_image, 0, 255).astype('uint8')
        output_image = output_image[...,::-1]  #RGB to BGR

        convert_type = cv2.COLOR_BGR2YUV
        inverse_convert_type = cv2.COLOR_YUV2BGR
        content_cvt = cv2.cvtColor(content_image, convert_type)
        output_cvt = cv2.cvtColor(output_image, convert_type)
        x, _, _ = cv2.split(output_cvt)
        _, y, z = cv2.split(content_cvt)

        merged_image = cv2.merge((x, y, z))
        image = cv2.cvtColor(merged_image, inverse_convert_type).astype(np.float32)

        # Postprocessing the image     
        image = np.copy(image).astype('float64')
        image = image[...,::-1]
        image = image[np.newaxis, :, :, :]
        image -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

        return image

    def training(self, content="Bangkok_TH", styles=["Composition_VII"], 
                 style_weights=[1], output="output0", alpha=5, beta=100, optimizer='Adam', 
                 learning_rate=2.0, iterations=1000, original_colors=False, pre_trained=None, verbose=False):
        '''arg styles and style_weights: 
               - when there's only one style picture, no need to enter the style_weights, the default value is [1], 
                 the function is the vanilla version tranfer
               - when there're multiple style images and the corresponding style weights, eg: there're two styles and 
                 the style_weights=[0.3, 0.7], the function would implement a multi-style transfer
           arg optimizer: the choices are "Adam", "Lbfgsb", "Adagrad", "Adadelta", "Rmsprop" and "Sgd"
           arg original_colors: if True, then conduct color preservation transfer
           arg verbose: if True, return all costs 
        '''
        model_name = 'my_artist_{}'.format(int(time.time()))
        best_loss = np.Inf

        with tf.Session() as sess:
            content_image = self.load_image(self.content_dir, content)
            style_images = [self.load_image(self.style_dir, style) for style in styles]
            input_image = self.add_noise(content_image)
            model = self.load_vgg()
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            sess.run(model["input"].assign(content_image))
            content_loss = self.content_loss(sess, model)
            style_loss = self.style_loss(sess, model, style_images, style_weights)
            total_loss = alpha * content_loss + beta * style_loss
            costs = []
            
            if pre_trained != None:
                try:
                    print("Load the model from: {}".format(pre_trained))
                    saver.restore(sess, 'model/{}'.format(pre_trained))
                except Exception:
                    raise ValueError("Load model Failed!")
            
            if optimizer == "Lbfgsb":
                train_step = tf.contrib.opt.ScipyOptimizerInterface(total_loss, method="L-BFGS-B", 
                                                        options={"maxiter": iterations, "disp": 100})
                sess.run(tf.global_variables_initializer())
                sess.run(model["input"].assign(input_image))
                train_step.minimize(sess)
                saver.save(sess, 'model/{}'.format(model_name))
            else:  
                op = self.get_optimizer(optimizer, learning_rate)
                train_step = op.minimize(total_loss)
                sess.run(tf.global_variables_initializer())
                sess.run(model["input"].assign(input_image))
                for i in range(iterations):
                    sess.run(train_step)
                    cost = sess.run(total_loss)
                    costs.append(cost)
                    
                    if (i+1) % 100 == 0:
                        print('Iteration {}/{}, cost: {}'.format((i+1), iterations, cost))
                        
                        if cost < best_loss:
                            best_loss = cost
                            print('  Best loss! Iteration {}/{}, cost: {}'.format((i+1), iterations, cost))
                            saver.save(sess, 'model/{}'.format(model_name))
#                         output_image = sess.run(model["input"])
#                         filename = 'check_t%d'%(i+1)
#                         self.save_image(self.output_dir, filename, output_image)
            
            output_image = sess.run(model["input"])
            if original_colors:
                output_image = self.keep_original_colors(content_image, output_image)
            
            self.save_image(self.output_dir, output, output_image)
            print("\n")
            print("Training ends. Output image {}.jpg and Model {} have already been saved.".format(output, model_name))
            
            if (verbose) & (costs!=[]) :
                return costs




