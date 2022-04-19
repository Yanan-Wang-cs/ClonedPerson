import cv2
import numpy as np


palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
]]


def process_mmdet_results(mmdet_results, cat_id=0):
        """Process mmdet results, and return a list of bboxes.

        :param mmdet_results:
        :param cat_id: category id (default: 0 for human)
        :return: a list of detected bounding boxes
        """
        if isinstance(mmdet_results, tuple):
            det_results = mmdet_results[0]
        else:
            det_results = mmdet_results

        bboxes = det_results[cat_id]

        person_results = []
        for bbox in bboxes:
            person = {}
            person['bbox'] = bbox
            person_results.append(person)

        return person_results

def pose_image(img, res):
    bbox = res['bbox']
    detection_score = bbox[4]
    bbox = [int(x) for x in bbox[:4]]
    left_top = (bbox[0], bbox[1])
    right_bottom = (bbox[2], bbox[3])

    image_tmp = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image_tmp = cv2.resize(image_tmp, (256, 256))
    blank_image = np.ones((256, 256, 3), np.uint8)

    bboxWidth = bbox[2] - bbox[0]
    bboxHeight = bbox[3] - bbox[1]
    kpts = res['keypoints']
    ratioWidth = 256. / bboxWidth
    ratioHeight = 256. / bboxHeight
    img_h = 256
    img_w = 256
    kpts[:, 0] = (kpts[:, 0] - bbox[0]) * ratioWidth
    kpts[:, 1] = (kpts[:, 1] - bbox[1]) * ratioHeight
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score > 0.1:
            r, g, b = pose_kpt_color[kid]

            cv2.circle(image_tmp, (int(x_coord), int(y_coord)),
                       4, (int(r), int(g), int(b)), -1)
            cv2.circle(blank_image, (int(x_coord), int(y_coord)),
                    4, (int(r), int(g), int(b)), -1)

    return image_tmp

    # if skeleton is not None and pose_limb_color is not None:
    #     assert len(pose_limb_color) == len(skeleton)
    #     for sk_id, sk in enumerate(skeleton):
    #         pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
    #         pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
    #         if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
    #                 and pos1[1] < img_h and pos2[0] > 0
    #                 and pos2[0] < img_w and pos2[1] > 0
    #                 and pos2[1] < img_h
    #                 and kpts[sk[0] - 1, 2] > 0.1
    #                 and kpts[sk[1] - 1, 2] > 0.1):
    #             r, g, b = pose_limb_color[sk_id]
    #             # cv2.line(
    #             #     image_tmp,
    #             #     pos1,
    #             #     pos2, (int(r), int(g), int(b)),
    #             #     thickness=2)
    #             cv2.line(
    #                 blank_image,
    #                 pos1,
    #                 pos2, (int(r), int(g), int(b)),
    #                 thickness=2)
    #     return blank_image


    