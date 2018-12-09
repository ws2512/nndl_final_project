import numpy as np
import scipy.io 
import scipy.misc
import tensorflow as tf 

class ArtisticStyle():

    def __init__(self):
        self.vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat') 
        self.img_width = 800
        self.img_height = 600
        self.channels = 3
        self.model = self.load_vgg()
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
        b = tf.constant(np.reshape(b, (b.shape[0],)))
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
        return image

    def save_image(self, output_dir, img_name, image):
        scipy.misc.imsave(output_dir+img_name+".jpg", image[0])

    def calc_content_loss(self, p, x):
        M = p.shape[1] * p.shape[2] # height * width
        N = p.shape[3] # # of filters
        loss = 0.5 * tf.reduce_sum(tf.pow(x - p, 2))
        return loss

    def content_loss(self, sess):
        '''
        return content loss
        '''
        return self.calc_content_loss(sess.run(self.model['conv4_2']), self.model['conv4_2'])

    def gram_matrix(self, x, M, N):
        F = tf.reshape(x, (M, N))
        G = tf.matmul(tf.transpose(F), F)
        return G

    def calc_style_loss(self, a, x):
        M = a.shape[1] * a.shape[2] # height * width
        N = a.shape[3] # # of filters
        A = self.gram_matrix(a, M, N) # gram matrix of original image
        G = self.gram_matrix(x, M, N)  # gram matrix of generated image
        loss = (1 / (4 * (N ^ 2) * (M ^ 2))) * tf.reduce_sum(tf.pow((G - A), 2))
        return loss

    def style_loss(self, sess):
        '''
        return style loss
        '''
        style_loss = 0
        layers = [("conv1_1", 1), ("conv2_1", 2), ("conv3_1", 3), ("conv4_1", 4), ("conv5_1", 5)]
        for layer in layers:
            E = self.calc_style_loss(sess.run(self.model[layer[0]]), self.model[layer[0]])
            W = layer[1]
            style_loss += E * W
        return style_loss

    def add_noise(self, content_img, noise_ratio=0.5):
        """
        Add noise to the content image
        """
        noise_img = np.random.uniform(-20, 20, (1, self.img_height, self.img_width, self.channels)).astype('float32')
        # white noise image  
        input_img = noise_img*noise_ratio + content_img*(1 - noise_ratio)
        return input_img

    def training(self, content, style, output, alpha=1, beta=100, learning_rate=1e-2, iterations=1000):
        content_image = self.load_image(self.content_dir, content)
        style_image = self.load_image(self.style_dir, style)
        input_image = self.add_noise(content_image)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sess.run(self.model["input"].assign(content_image))
            content_loss = self.content_loss(sess)
            sess.run(self.model["input"].assign(style_image))
            style_loss = self.style_loss(sess)
            total_loss = alpha * content_loss + beta * style_loss
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(total_loss)
            
            sess.run(tf.global_variables_initializer())
            sess.run(self.model["input"].assign(input_image))  
            for i in range(iterations):
                sess.run(train_step)
                count = i+1
                if count % 100 == 0:
                    print('Iteration {}/{}, cost: {}'.format(count, iterations, sess.run(total_loss)))

            # save the final image
            output_image = sess.run(self.model["input"])
            self.save_image(self.output_dir, output, output_image)
            print("\n")
            print("Training ends. Output image has already been saved.")




