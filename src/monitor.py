import tensorflow as tf
from generator import latent_dim
import matplotlib.pyplot as plt


class Monitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=latent_dim):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%5 == 0 or epoch==0:
            fig, axs = plt.subplots(1, self.num_img, figsize= (11,11))
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)
            for i, ax in zip(range(self.num_img), axs):
#                 generated_images *= 255
                generated_images = (generated_images+1)*127.5
                generated_images.numpy()
                img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
                ax.set_title(f"{i+1}")
                ax.axis('off')
                ax.imshow(img)
            plt.tight_layout()
            plt.subplots_adjust(wspace= 0, hspace= 0)
            plt.show()
#         print(f"{epoch+1}: Gen_loss: {logs['g_loss']:.2f}\t Disc_loss: {logs['d_loss']:.2f}")
plots = Monitor()