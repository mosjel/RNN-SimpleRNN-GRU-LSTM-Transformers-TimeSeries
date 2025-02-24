import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import tensorflow_addons as tfa
num_classes=100
input_shape=(32,32,3)
image_size=72
batch_size=64
num_epochs=100
patch_size=6
transformer_layers=8
num_heads=4
num_patches=(image_size//patch_size)**2
projection_dim=64
transformer_units=[projection_dim*2,projection_dim]
mlp_head_units=[2048,1024]
learning_rate=0.001
weight_decay=0.0001
(x_train, y_train),(x_test, y_test)=keras.datasets.cifar100.load_data()
print(f"x_train shape : {x_train.shape}- y_train shape {y_train.shape}",f"x_test shape : {x_test.shape}- y_test shape {y_test.shape}")


def mlp(x,hidden_units,dropout_rate):
    for units in hidden_units:
        print("HO",units)
        x=layers.Dense(units,activation=tf.nn.gelu)(x)
        x=layers.Dropout(dropout_rate)(x)
    return x


data_augmentation=keras.Sequential([
        layers.Normalization(),
        layers.Resizing(image_size,image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
    layers.RandomZoom(height_factor=0.2,width_factor=0.2),

],name="data_augmentation")

data_augmentation.layers[0].adapt(x_train)
class Patches(layers.Layer):
    def __init__(self,patch_size):
        super(Patches,self).__init__()
        self.patch_size=patch_size
    def call(self,images):
        batch_size=tf.shape(images)[0]
        print(batch_size,"batch size")
        patches=tf.image.extract_patches(
            images=images,
            sizes=[1,self.patch_size,self.patch_size,1],
            strides=[1,patch_size,patch_size,1],
            rates=[1,1,1,1],
            padding="VALID",
        )
        patch_dims=patches.shape[-1]
        patches=tf.reshape(patches,[batch_size,-1,patch_dims])
        return patches

import matplotlib.pyplot as plt
plt.figure(figsize=(32,32))
image=x_train[np.random.choice(range(x_train.shape[0]))]
# plt.imshow(image.astype("uint8"))
# plt.axis("off")

resized_image=tf.image.resize(tf.convert_to_tensor([image]),size=(image_size,image_size))
patches=Patches(patch_size)(resized_image)



print(f"image size:{image_size} X{image_size}")
print(f"patch size:{patch_size} X{patch_size}")
print(f"patches per image:{patches.shape[1]}")
print(f"elements per patches:{patches.shape[-1]}")
print(patches.shape)
n=int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4,4))
# for i, patch in enumerate(patches[0]):
#     ax=plt.subplot(n,n,i+1)
#     patch_img=tf.reshape(patch,(patch_size,patch_size,3))
    # plt.imshow(patch_img.numpy().astype("uint8"))
    # plt.axis("off")

# plt.show()


class Patchencoder(layers.Layer):
    def __init__(self,num_patches,projection_dim):
        super(Patchencoder,self).__init__()
        self.num_patches=num_patches
        self.projection=layers.Dense(units=projection_dim)
        self.position_embedding=layers.Embedding(
            input_dim=num_patches,output_dim=projection_dim
        )    
    def call(self, patch):
        positions=tf.range(start=0,limit=self.num_patches,delta=1)
        encoded=self.projection(patch)+self.position_embedding(positions)
        return encoded

def create_vit_classifier():
    inputs=layers.Input(shape=(input_shape))
    augmented=data_augmentation(inputs)
    patches=Patches(patch_size)(augmented)
    encoded_patches=Patchencoder(num_patches,projection_dim)(patches)
    for _ in range(transformer_layers):
        x1= layers.LayerNormalization(epsilon=1e-6) (encoded_patches)
        attention_output=layers.MultiHeadAttention(num_heads=num_heads,key_dim=projection_dim,dropout=0.1)(x1,x1)

        x2=layers.Add()([attention_output,encoded_patches])
        print(x2.shape)
        x3=layers.LayerNormalization(epsilon=1e-6,)(x2)
        print(x3.shape)
        x3=mlp(x3,hidden_units=transformer_units,dropout_rate=0.1)
        print(x3.shape)

        encoded_patches=layers.Add()([x3,x2])
    representation=layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation=layers.Flatten()(representation)
    representation=layers.Dropout(0.5)(representation)
    
    features=mlp(representation,hidden_units=mlp_head_units,dropout_rate=0.5)
    logits=layers.Dense(num_classes)(features)
    model=keras.Model(inputs=inputs,outputs=logits)
    return model

def run_experiment(model):
    optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5,name="top-5-accuracy")
        ]
    )
    checkpoint_filepath="/tmp/checkpoint"
    checkpoint_callback=keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history=model.fit(x=x_train,y=y_train,batch_size=batch_size,
                        epochs=num_epochs,
                        validation_split=0.1,
                        callbacks=[checkpoint_callback]
                        )
    model.load_weights(checkpoint_filepath)
    _,accuracy,top_5_accuracy=model.evaluate(x_test,y_test)
    print(f"Test Accuracy:{round(accuracy*100*2)}%")
    print(f"Test top 5 Accuracy:{round(top_5_accuracy*100*2)}%")

    return history
print("googooli")
vit_classifier=create_vit_classifier()
print(vit_classifier.summary())
history=run_experiment(vit_classifier)
print(history)
