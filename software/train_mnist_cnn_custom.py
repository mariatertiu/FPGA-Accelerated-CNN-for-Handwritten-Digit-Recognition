import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

                                                   
SEED = 42
# pentru rezultate cat de cat repetabile
np.random.seed(SEED)
                                             
tf.random.set_seed(SEED)
                                                                                                                      
# setari principale
CUSTOM_DATASET_DIR = "DATASET_MNIST_PROCESAT"
                                                              
BATCH_SIZE = 128
                                      
EPOCHS = 40

                                                                                                                       
                                                                
CUSTOM_REPEAT = 3
# repet datasetul meu ca sa conteze mai mult la antrenare

                                                                       
                               
                                        
ROTATION_FACTOR = 0.03                             
# rotire mica, cam +/-10.8 grade


                                                        

                                                                                             
# citesc imaginile mele din folderele 0-9
def load_custom_dataset(dataset_dir):
    images = []
                                                               
    labels = []
                                                              

    if not os.path.exists(dataset_dir):
        # daca nu gasesc folderul meu, merg doar pe MNIST
                                                                       
        print(f"[INFO] Folderul {dataset_dir} nu exista. Se foloseste doar MNIST.")
        return None, None

    print()
    print("======================================")
    print(" INCARC DATASET CUSTOM")
    print("======================================")

    for label in range(10):
        # fiecare cifra are folderul ei
                                                         
        folder = os.path.join(dataset_dir, str(label))
                                                           

        if not os.path.exists(folder):
                                                                    
            print(f"[WARN] Lipseste folderul pentru cifra {label}: {folder}")
            continue

        files = [
                                                       
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        print(f"Cifra {label}: {len(files)} imagini gasite")

        for name in files:
            path = os.path.join(folder, name)
                                                        

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # citesc direct alb-negru
                                                

            if img is None:
                                                                       
                print(f"[WARN] Nu pot citi imaginea: {path}")
                continue

            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            # toate imaginile trebuie sa fie 28x28
                                                                           
            img = img.astype("float32") / 255.0
            # normalizez pixelii intre 0 si 1
                                                     

                                                         
                                                                         
            if np.mean(img) > 0.5:
                # daca e fundal alb, inversez imaginea
                                                              
                img = 1.0 - img
                                                                           

            images.append(img)
                                                               
            labels.append(label)
                                                             

    if len(images) == 0:
        print("[INFO] Nu am gasit imagini in dataset custom. Se foloseste doar MNIST.")
        return None, None

    images = np.array(images, dtype=np.float32)
                                                 
    labels = np.array(labels, dtype=np.int64)
                                           

    images = np.expand_dims(images, axis=-1)
    # adaug canalul grayscale
                                                         

    print("--------------------------------------")
    print(f"Total imagini custom: {len(images)}")
    print("Shape imagini:", images.shape)
    print("======================================")
    print()

    return images, labels


                                  

# incarc MNIST-ul original
print("Incarc MNIST...")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# setul vine deja impartit in train si test
                                                                 

x_train = x_train.astype("float32") / 255.0
                                                              
x_test = x_test.astype("float32") / 255.0
                                                         

x_train = np.expand_dims(x_train, axis=-1)
                                                         
x_test = np.expand_dims(x_test, axis=-1)
                                                    

y_train_cat = keras.utils.to_categorical(y_train, 10)
# etichetele devin vectori pentru cele 10 clase
                                                                
y_test_cat = keras.utils.to_categorical(y_test, 10)
                                                  

print(f"MNIST train: {x_train.shape}")
print(f"MNIST test:  {x_test.shape}")


                                                           

# pastrez 5000 imagini pentru validare
x_val = x_train[-5000:]
                                                            
y_val = y_train_cat[-5000:]
                                                                

x_train_small = x_train[:-5000]
                                                 
y_train_small = y_train_cat[:-5000]
                                            


                                                             

# adaug si datasetul meu peste MNIST
x_custom, y_custom = load_custom_dataset(CUSTOM_DATASET_DIR)
                                                  

# daca am dataset propriu, il bag in train
if x_custom is not None:
                                                                                  
    y_custom_cat = keras.utils.to_categorical(y_custom, 10)
                                                      

    x_custom_rep = np.concatenate([x_custom] * CUSTOM_REPEAT, axis=0)
    # repet imaginile mele
                                                                             
    y_custom_rep = np.concatenate([y_custom_cat] * CUSTOM_REPEAT, axis=0)
                                                     

    x_train_small = np.concatenate([x_train_small, x_custom_rep], axis=0)
    # unesc MNIST cu datasetul meu
                                                                  
    y_train_small = np.concatenate([y_train_small, y_custom_rep], axis=0)
                                                       

    print("======================================")
    print(" TRAINING FINAL")
    print("======================================")
    print(f"MNIST train fara validare: {len(x_train[:-5000])}")
    print(f"Dataset custom original:   {len(x_custom)}")
    print(f"Custom repeat:             {CUSTOM_REPEAT}")
    print(f"Custom folosit efectiv:    {len(x_custom_rep)}")
    print(f"Total train:               {len(x_train_small)}")
    print("======================================")
    print()
else:
    print("[INFO] Continui doar cu MNIST.")


                                                 

# mici modificari pe imagini, ca modelul sa fie mai robust
data_augmentation = keras.Sequential([
                                                                          
    layers.RandomRotation(
                                                                 
        ROTATION_FACTOR,
        fill_mode="constant",
        fill_value=0.0
    ),
    layers.RandomTranslation(
                                         
        0.06,
        0.06,
        fill_mode="constant",
        fill_value=0.0
    ),
    layers.RandomZoom(
                                        
        0.06,
        fill_mode="constant",
        fill_value=0.0
    ),
    layers.RandomContrast(0.10),
                                          
], name="augmentation")

print("Augmentare training:")
print(f"  RandomRotation = {ROTATION_FACTOR} => +/- {ROTATION_FACTOR * 360:.1f} grade")
print("  RandomTranslation = 0.06")
print("  RandomZoom = 0.06")
print("  RandomContrast = 0.10")
print()


                                                                  

# arhitectura CNN folosita in proiect
model = keras.Sequential([
                                                    
    layers.Input(shape=(28, 28, 1), name="input_img"),
                                                      

    layers.Conv2D(
        # primul conv
                                                               
        4,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv1"
    ),

    layers.MaxPooling2D(
        # primul pooling
                                                               
        pool_size=(2, 2),
        name="pool1"
    ),

    layers.Conv2D(
        # al doilea conv
                                                               
        8,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        activation="relu",
        name="conv2"
    ),

    layers.MaxPooling2D(
        # al doilea pooling
                                                               
        pool_size=(2, 2),
        name="pool2"
    ),

    layers.Flatten(name="flatten"),
    # transform in vector pentru straturile dense
                                             

    layers.Dense(
                                                       
        32,
        activation="relu",
        name="fc1"
    ),

    layers.Dense(
                                                       
        10,
        name="fc2"
    )
])

model.summary()
                                                         


                                                                            

# pregatesc datele pentru antrenare
train_ds = tf.data.Dataset.from_tensor_slices((x_train_small, y_train_small))
                                                
train_ds = train_ds.shuffle(30000, seed=SEED)
# amestec datele
                                                                        
train_ds = train_ds.batch(BATCH_SIZE)
# le impart pe batch-uri
                              

train_ds = train_ds.map(
                                                 
    lambda x, y: (data_augmentation(x, training=True), y),
    # modific doar imaginea, eticheta ramane aceeasi
                                                            
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
                                                         

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
                                
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat))
                            
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


                                         

# setez modul de antrenare
model.compile(
                                                                  
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                                                                  
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    # folosesc from_logits deoarece fc2 da scoruri, nu softmax
                                                                                        
    metrics=["accuracy"]
                                                               
)


# reguli pentru oprire, learning rate si salvarea celui mai bun model
callbacks = [
                                                
    keras.callbacks.EarlyStopping(
        # opreste daca nu mai creste validarea
                                                                
        monitor="val_accuracy",
                                                    
        patience=8,
                                                                 
        restore_best_weights=True
                                                    
    ),

    keras.callbacks.ReduceLROnPlateau(
        # scade learning rate-ul cand modelul stagneaza
                                                                    
        monitor="val_loss",
        factor=0.5,
                                         
        patience=2,
        min_lr=1e-5,
                                                                
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        # salveaza cea mai buna varianta
                                                         
        "mnist_manual_cnn_best.h5",
                                                                    
        monitor="val_accuracy",
                                                    
        save_best_only=True,
                                                                 
        verbose=1
    )
]


                          

print()
# pornesc antrenarea
print("Incep antrenarea...")
print("Arhitectura este aceeasi ca in modelul vechi.")
print("Se modifica doar greutatile.")
print()

model.fit(
    # antrenarea efectiva
                               
    train_ds,
    validation_data=val_ds,
                                                                   
    epochs=EPOCHS,
                                                            
    callbacks=callbacks,
                                                                                     
    verbose=1
)


                                         

# test pe MNIST
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
                                                 

print()
print("======================================")
print(" TEST NORMAL MNIST")
print("======================================")
print(f"Test loss:     {test_loss:.6f}")
print(f"Test accuracy: {test_acc:.6f}")
print("======================================")
print()


                                                       

# test si pe imaginile mele
if x_custom is not None:
                                                                                  
    y_custom_cat = keras.utils.to_categorical(y_custom, 10)
                                                      

    custom_loss, custom_acc = model.evaluate(
        # vad cum merge pe datele mele
                                                          
        x_custom,
        y_custom_cat,
        verbose=0
    )

    print()
    print("======================================")
    print(" TEST PE DATASET_MNIST_PROCESAT")
    print("======================================")
    print(f"Custom loss:     {custom_loss:.6f}")
    print(f"Custom accuracy: {custom_acc:.6f}")
    print("======================================")
    print()


                                   

# salvez modelul final
model.save("mnist_manual_cnn_final.h5")
                                                    

print("Modele salvate:")
print(" - mnist_manual_cnn_best.h5")
print(" - mnist_manual_cnn_final.h5")


                                        

# verificare rapida pe cateva imagini
pred = model.predict(x_test[:10], verbose=0)
                                                    

print()
print("Etichete reale MNIST:  ", y_test[:10])
print("Predictii model MNIST: ", np.argmax(pred, axis=1))
# argmax alege cifra cu scorul cel mai mare
                                                                   
