import cozmo
import time
from scipy.misc import imread, imresize, imsave, imshow
import os

def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True  # setting this make it None
    robot.camera.color_image_enabled = True
    robot.drive_off_charger_on_connect = False
    print('min head angle in rad', cozmo.robot.MIN_HEAD_ANGLE)
    print('max head angle in rad', cozmo.robot.MAX_HEAD_ANGLE)
    robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE)
    time.sleep(2)
    # robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE)
    # camera's information
    print(robot.camera.config.focal_length)
    print(robot.camera.config.center)
    print(type(cozmo.robot.MAX_HEAD_ANGLE))
    robot.set_head_angle(cozmo.util.Angle(0))
    time.sleep(1)

    # robot.set_head_angle(0.01)
    print("enter exit to quit program, any other input will be considered ready")
    # path = "/home/hav/workplace/camcoz/"
    path = "/home/hav/workspace/camcoz/"
    # find max id images
    im_names = sorted(os.listdir(path))
    im_names = [e for e in im_names if 'JPEG' in e]
    if len(im_names) > 0:
        cur_max = int(im_names[-1][0:3])
        cnt = cur_max + 1
    else:
        cnt = 1
    print(cnt)

    while True:
        print("enter: ", end='')
        ready = input()
        if ready == "exit":
            break
        cur_img_name = "%03d.JPEG" % cnt
        #robot.say_text("take picture").wait_for_completed()
        latest_img = robot.world.latest_image
        img = latest_img.raw_image
        img = imresize(img, (224, 224))
        imsave(path + cur_img_name, img)
        print("done %d" % cnt)
        cnt += 1



cozmo.run_program(cozmo_program, use_viewer=True)
# im1 = imread("/home/hav/workplace/camcoz/try2/im13_pen.JPEG")
# im2 = imread("/home/hav/workplace/camcoz/try2/im14_pen.JPEG")
# imshow(im1 - im2)


