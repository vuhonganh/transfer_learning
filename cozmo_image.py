import cozmo
import time
from scipy.misc import imread, imresize, imsave, imshow


def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True  # setting this make it None
    robot.camera.color_image_enabled = True
    robot.drive_off_charger_on_connect = False
    #time.sleep(4)
    print("enter exit to quit program, any other input will be considered ready")
    cnt = 1
    path = "/home/hav/workplace/camcoz/"
    while True:
        print("enter: ", end='')
        ready = input()
        if ready == "exit":
            break
        cur_img_name = "im%02d.JPEG" % cnt
        robot.say_text("take picture").wait_for_completed()
        latest_img = robot.world.latest_image
        img = latest_img.raw_image
        #img = imresize(img, (224, 224))
        imsave(path + cur_img_name, img)
        cnt += 1


# cozmo.run_program(cozmo_program, use_viewer=True)
im1 = imread("/home/hav/workplace/camcoz/try2/im13_pen.JPEG")
im2 = imread("/home/hav/workplace/camcoz/try2/im14_pen.JPEG")
imshow(im1 - im2)


