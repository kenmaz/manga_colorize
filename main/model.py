from __future__ import division
import os
import sys
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from ops import *
from utils import *
import scipy.misc

class pix2pix(object):
    def __init__(self,
                 sess,
                 image_size,
                 load_size,
                 output_size,
                 batch_size=1,
                 sample_size=1,
                 gf_dim=64,
                 df_dim=64,
                 L1_lambda=100,
                 input_c_dim=4,
                 output_c_dim=3,
                 dataset_name="manga",
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.load_size = load_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')
        self.d_bn6 = batch_norm(name='d_bn6')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        # 3ch image (real-color)
        self.real_B = tf.placeholder(tf.float32,
                                    [self.batch_size, self.image_size[0], self.image_size[1],self.output_c_dim],
                                    name='real_B')

        # 4ch image (zoom-color + mono-line)
        self.real_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size[0], self.image_size[1], self.input_c_dim],
                                     name='real_A')

        self.fake_B = self.generator(self.real_A)

        self.D, self.D_logits = self.discriminator(self.real_B, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_B, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            print("dis image:%s" % image.get_shape)
            # image (:, 1200, 800, 3)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # image (:, 600, 400, 64)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # image (:, 300, 200, 128)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # image (:, 150, 100, 256)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            # image (:, 75, 50, 512)
            h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, name='d_h4_conv')))
            # image (:, 38, 25, 1024)
            h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim*32, name='d_h5_conv')))
            # image (:, 19, 13, 2048)
            h6 = lrelu(self.d_bn6(conv2d(h5, self.df_dim*64, d_h=1, d_w=1, name='d_h6_conv')))
            h7 = linear(tf.reshape(h6, [self.batch_size, -1]), 1, 'd_h7_lin')

            for h in [h0,h1,h2,h3,h4,h5,h6,h7]:
                print("h:%s" % h)

            return tf.nn.sigmoid(h7), h7

    def generator(self, image, y=None):
        #[1200, 800]
        s = self.output_size

        # image (2, 1200, 800, 4)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
        # e1 is (2, 600, 400, 64)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
        # e2 is (2, 300, 200, 128)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
        # e3 is (2, 150, 100k, 256)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
        # e4 is (2, 75, 50, 512)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
        # e5 is (2, 38, 25, 512)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
        # e6 is (2, 19, 13, 512)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
        # e7 is (2, 10, 7, 512]
        e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
        # e8 is (2, 5, 4, 512)

        self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8), e7.get_shape(), name='g_d1', with_w=True)
        d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
        d1 = tf.concat(axis=3, values=[d1, e7])
        print(d1.get_shape())
        # d1 is (2, 10, 7, 1024)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1), e6.get_shape(), name='g_d2', with_w=True)
        d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
        d2 = tf.concat(axis=3, values=[d2, e6])
        print(d2.get_shape())
        # d2 is (2, 20, 13, 1024)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2), e5.get_shape(), name='g_d3', with_w=True)
        d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
        d3 = tf.concat(axis=3, values=[d3, e5])
        print(d3.get_shape())
        # d3 is (2, 40, 25, 1024)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3), e4.get_shape(), name='g_d4', with_w=True)
        d4 = self.g_bn_d4(self.d4)
        d4 = tf.concat(axis=3, values=[d4, e4])
        print(d4.get_shape())
        # d4 is (2, 80, 50, 1024)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), e3.get_shape(), name='g_d5', with_w=True)
        d5 = self.g_bn_d5(self.d5)
        d5 = tf.concat(axis=3, values=[d5, e3])
        print(d5.get_shape())
        # d5 is (2, 160, 100, 512)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), e2.get_shape(), name='g_d6', with_w=True)
        d6 = self.g_bn_d6(self.d6)
        d6 = tf.concat(axis=3, values=[d6, e2])
        print(d6.get_shape())
        # d6 is (2, 320, 200, 256)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), e1.get_shape(), name='g_d7', with_w=True)
        d7 = self.g_bn_d7(self.d7)
        d7 = tf.concat(axis=3, values=[d7, e1])
        print(d7.get_shape)
        # d7 is (2, 640, 400, 128)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
            [self.batch_size, int(s[0]), int(s[1]), self.output_c_dim], name='g_d8', with_w=True)
        # d8 is (2, 1200, 800, 3)
        print(self.d8.get_shape())

        return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size[0], self.output_size[1])
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...", checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

    def train_zoom(self, args):

        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        self.sess.run(tf.global_variables_initializer())

        self.g_sum = tf.summary.merge([self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        start_time = time.time()
        self.load(self.checkpoint_dir)

        for epoch in xrange(args.epoch):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_images = [load_data(batch_file, self.load_size, self.image_size) for batch_file in batch_files]
                mono_imgs, color_imgs = self.resize_for_zoom(batch_images)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={ self.real_A: mono_imgs, self.real_B: color_imgs })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str, out_images = self.sess.run([g_optim, self.g_sum, self.d8], feed_dict={ self.real_A: mono_imgs, self.real_B: color_imgs })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({ self.real_A: mono_imgs, self.real_B: color_imgs })
                errD_real = self.d_loss_real.eval({ self.real_A: mono_imgs, self.real_B: color_imgs })
                errG = self.g_loss.eval({ self.real_A: mono_imgs, self.real_B: color_imgs })

                if np.mod(counter, 10) == 0:
                    result_img = scipy.misc.imresize(out_images[0], self.image_size)
                    scipy.misc.imsave("out_%d.jpg" % counter, result_img)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                            % (epoch, idx, batch_idxs,
                                time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)

    def resize_for_zoom(self, batch_images):
        color_imgs = []
        mono_imgs = []
        for i in xrange(self.batch_size):
            img_color = np.array(batch_images[i][0]).astype(np.float32)
            img_mono =  np.array(batch_images[i][1]).astype(np.float32)
            min_color = scipy.misc.imresize(img_color, self.load_size)
            min_color_zoom = scipy.misc.imresize(min_color, self.image_size)
            big_mono = scipy.misc.imresize(img_mono, self.image_size)
            src_imgs = np.insert(min_color_zoom, 3, big_mono, axis=2)
            dst_imgs = scipy.misc.imresize(img_color, self.image_size)
            mono_imgs.append(src_imgs/127.5 - 1.)
            color_imgs.append(dst_imgs/127.5 - 1.)
        return np.array(mono_imgs), np.array(color_imgs)

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def test(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.load(self.checkpoint_dir)

        data = glob('./datasets/manga/val/*.jpg')
        batch_idxs = min(len(data), args.train_size) // self.batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            print("sampling image ", idx, batch_files)

            batch_images = [load_data(batch_file, self.load_size, self.image_size) for batch_file in batch_files]
            mono_imgs, color_imgs = self.resize_for_zoom(batch_images)

            samples = self.sess.run(
                self.fake_B,
                feed_dict={self.real_A: mono_imgs}
            )

            fake_img = scipy.misc.imresize(samples[0], self.image_size)
            real_img = scipy.misc.imresize(color_imgs[0], self.image_size)
            scipy.misc.imsave("test/%d_fake.jpg" % idx, fake_img)
            scipy.misc.imsave("test/%d_real.jpg" % idx, real_img)
