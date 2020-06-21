from keras.preprocessing.image import ImageDataGenerator


def multiple_outputs(generator, image_dir, batch_size, image_size,classes):
    gen = generator.flow_from_directory(
        image_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        classes = classes,
        class_mode='categorical',
        )
    
    while True:
        gnext = gen.next()
        # return image batch and 3 sets of lables
        yield gnext[0], [gnext[1], gnext[0]]


def Srinivasan_2014(batch_size,image_size,data_dir):

    train_path = data_dir+'/Train'
    test_path = data_dir+'/Test'

    classes=['AMD', 'DME', 'NORMAL']
    train_datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                rescale=1.0/255,
                                horizontal_flip=True,
                                fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = multiple_outputs(
    train_datagen,
    image_dir=train_path,
    batch_size=batch_size,
    image_size=image_size,
    classes=classes,
    )
     
    validation_generator = multiple_outputs(
    test_datagen,
    image_dir=test_path,
    batch_size=batch_size,
    image_size=image_size,
    classes=classes,
    )
    return train_generator, validation_generator

        