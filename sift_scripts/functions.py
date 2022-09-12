import cv2
import os
import matplotlib.pyplot as plt


# Loads training images from given path
def load_images(path):

    image_label = []
    train = []
    for filename in os.listdir(path):
        # kitten image
        if "k" in filename:
            image_label.append('k')
        # dog image
        if "d" in filename:
            image_label.append('d')
        # tree image
        if "r" in filename:
            image_label.append('t')
        # wheel image
        if "w" in filename:
            image_label.append('w')

        full_path = os.path.join(path, filename)
        # reads image
        img = cv2.imread(full_path)
        if img is None:
            print('Could not open or find the images!')
            exit(0)

        # append image in training list
        train.append(img)

    return train, image_label


# apply sift algorithm on training photos
def apply_sift_algo(sift_model, image_list):

    flag = 0
    directory = "img_keypoints"
    ask_save = input("\nDo you want pictures with keypoints to be saved? (yes or no) ")
    if ask_save == "yes":
        flag = 1
    kp1, des1 = sift_model.detectAndCompute(image_list[0], None)

    if len(kp1) == 0:
        print("\nNo features found in initial photo. Exiting...\n")
        exit(0)
    out1 = cv2.drawKeypoints(image_list[0], kp1, None)
    if flag == 1:
        filename = "initial.jpg"
        cv2.imwrite(os.path.join(directory, filename), out1)

    descr_list = [[kp1, des1]]
    print("\nKeypoints found on initial photo: ", len(kp1))

    for i in range(1, len(image_list)):

        kp2, des2 = sift_model.detectAndCompute(image_list[i], None)
        out2 = cv2.drawKeypoints(image_list[1], kp2, None)
        if flag == 1:
            filename = f"{i}.jpg"
            cv2.imwrite(os.path.join(directory, filename), out2)
        descr_list.append([kp2, des2])
        print("Keypoints on photo #", i, " : ", len(kp2))

    success = matcher_function(descr_list, image_list)

    return len(kp1), success

# Finds matches between photos, computes the best of them
def matcher_function(kp_des_list, image_list):

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    i = 0
    good = []
    rate = 0
    success_rate = 0
    counter = 0
    print("\n")

    directory = "matches"
    flag = 0
    ask_save = input("Do you want to save pictures of matches? (yes or no) ")
    if ask_save == "yes":
        flag = 1

    for i in range(1, len(image_list)):

        if len(kp_des_list[i][0]) == 0:
            print("Good matches for photo # ", i, " : 0 out of 0")
            counter += 1
            continue

        matches = bf.knnMatch(kp_des_list[0][1], kp_des_list[i][1], k=2)

        if len(matches) == 0:
            counter += 1
            continue

        good.clear()
        # Apply ratio test from D. Lowe's paper
        for match in matches:
            if len(match) != 1:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append([m])

        # cv.drawMatchesKnn expects list of lists as matches.
        photo_match = cv2.drawMatchesKnn(
            image_list[0], kp_des_list[0][0], image_list[i], kp_des_list[i][0],
            good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        if flag == 1:
            filename = f"{i}.jpg"
            cv2.imwrite(os.path.join(directory, filename), photo_match)

        print("Good matches for photo #", i, " : ", len(good), " out of ", len(matches))
        rate = (len(good)/len(matches))*100
        success_rate += rate

    success_rate /= (len(image_list) - 1 - counter)
    print("\nSuccess rate of this model is: ", success_rate)

    return success_rate


# Makes graphs of keypoints and success rate and saves them
def save_graphs(select_list, success_rate, init_keypoints, parameter):

    directory = "figures"
    filename1 = "success_rate.png"
    filename2 = "num_keypoints.png"

    plt.plot(select_list, success_rate, 'g', marker='o')
    plt.grid()
    plt.title('Success Rate')
    plt.xlabel(parameter)
    plt.ylabel('Rate')
    plt.savefig(os.path.join(directory, filename1))

    plt.clf()

    plt.plot(select_list, init_keypoints, 'b', marker='o')
    plt.grid()
    plt.title('Number of keypoints found on training photo')
    plt.xlabel(parameter)
    plt.ylabel('Keypoints')
    plt.savefig(os.path.join(directory, filename2))
