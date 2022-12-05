from django.shortcuts import render, redirect
from .forms import *
import numpy as np
from keras.models import load_model
import cv2
import os


def index(request):
    form = ImageForm(request.POST)
    model = load_model('C:/Users/ASUS/Documents/model_unet.hdf5')
    if request.method == "POST":
        color = request.POST.get("color")
        color = color.split(',')
        r = int(color[0])
        g = int(color[1])
        b = int(color[2])
        try:
            image = request.FILES["image"]
            obj = ImageModel(image=image)
            obj.save()
            image_reader = ImageModel.objects.all()[len(ImageModel.objects.all()) - 1]
            img = cv2.imread(image_reader.image.url[1:])
            size = (int(img.shape[1]), int(img.shape[0]))
            cv2.imwrite('statics/result/source.jpg', cv2.resize(img, size, interpolation=cv2.INTER_LINEAR))
            resize_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

            images_test = [gray_img]
            images_test = np.array(images_test)
            images_test = np.expand_dims(images_test, axis=3)

            images_test = images_test / 255

            threshold = 0.5
            test_img = images_test[0]

            test_img_input = np.expand_dims(test_img, 0)
            prediction = (model.predict(test_img_input)[0, :, :, 0] > threshold).astype(np.uint8)

            rgb = []
            for i in prediction.ravel():
                temp = [i] * 3
                if i == 0.0:
                    temp = 0
                elif i == 1.0:
                    temp = 255
                rgb.append(temp)
            rgb = np.array(rgb)
            rgb = rgb.reshape(256, 256).astype(np.uint8)

            test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            _, rgb = cv2.threshold(rgb, 127, 255, cv2.THRESH_BINARY)

            img2 = test.copy()
            for i in range(len(img2)):
                for j in range(len(img2[0])):
                    if rgb[i, j] == 255:
                        img2[i, j] = [r, g, b]
                    else:
                        img2[i, j] = [0, 0, 0]

            img1 = test.copy()

            # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2[:,:,0], 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

            # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)

            d = cv2.addWeighted(test, 0.6, dst, 0.4, 0)
            result = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
            cv2.imwrite('statics/result/result.jpg', result)
            data_context = {
                'data': image_reader,
                'form': form,
                'width': int(img.shape[1]),
            }
            return render(request, 'index.html', data_context)
        except:
            data_context = {
                'form': form,
            }
            return render(request, 'index.html', data_context)

    data_context = {
        'form': form,
    }
    return render(request, 'index.html', data_context)
