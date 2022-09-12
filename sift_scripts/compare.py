import cv2
import os
import functions
import img_proc

if __name__ == '__main__':

    # Get data path
    # data_path = input("Give file path to images : ")
    # data_path.strip("\n")
    data_path = "training_images"

    # Load images and save in lists
    init_images, label_list = functions.load_images(data_path)

    # Choose one of the photos to experiment with
    while True:

        chosen_photo = input("Which photo would you like to choose for the experiment? ")
        # chosen_photo.strip("\n")
        if chosen_photo == "dog":
            chosen_label = 'd'
            break
        elif chosen_photo == "cat":
            chosen_label = 'k'
            break
        elif chosen_photo == "wheel":
            chosen_label = 'w'
            break
        elif chosen_photo == "tree":
            chosen_label = 't'
            break
        else:
            print("This photo doesn't exist. Try again.")

    # Perform transformations on the chosen photo
    training_list = img_proc.process_img(init_images, label_list, chosen_label)

    # SIFT model 1
    print("\nBuilding model #1...")
    sift_model_1 = cv2.SIFT_create()

    print("Give parameters for the second model\n")
    print("Model 2: ")

    num_feat2 = int(input("\nGive number of features: "))

    num_octave2 = int(input("Give number of octave layers: "))

    contrast2 = float(input("Give contrast threshold: "))

    edge2 = float(input("Give edge threshold: "))

    gauss_sig2 = float(input("Give Gaussian sigma: "))

    # SIFT model 1
    print("\nBuilding model #2...")
    sift_model_2 = cv2.SIFT_create(
        nfeatures=num_feat2, nOctaveLayers=num_octave2,
        contrastThreshold=contrast2, edgeThreshold=edge2, sigma=gauss_sig2
    )

    # Numbers are given in the paper
    selection = int(input("Select the number of the image you want to compare with the initial: "))
    initial = training_list[0]
    query = training_list[selection]

    # Detect and compute keypoints
    kp1, des1 = sift_model_1.detectAndCompute(initial, None)
    kp2, des2 = sift_model_2.detectAndCompute(query, None)

    if len(kp1) == 0:
        print("\nNo features found on initial photo. Therefore, there are no matches.\n")
        exit(0)
    if len(kp2) == 0:
        print("\nNo features found on query photo. Therefore, there are no matches.\n")
        exit(0)

    # Draw keypoints on each image  and print
    out1 = cv2.drawKeypoints(initial, kp1, None)
    out2 = cv2.drawKeypoints(query, kp2, None)

    # Save photos with keypoints
    directory = "img_keypoints"
    filename1 = "comp_photo1.jpg"
    cv2.imwrite(os.path.join(directory, filename1), out1)
    filename2 = "comp_photo2.jpg"
    cv2.imwrite(os.path.join(directory, filename2), out2)

    cv2.imshow('Initial photo', out1)
    cv2.waitKey(6000)

    cv2.imshow('Query photo', out2)
    cv2.waitKey(6000)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    good = []

    matches = bf.knnMatch(des1, des2, k=2)

    if len(matches) == 0:
        print("There were no matches. Exiting... ")
        exit(0)

    good.clear()
    # Apply ratio test from D. Lowe's paper
    for match in matches:
        if len(match) != 1:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    photo_match = cv2.drawMatchesKnn(
        initial, kp1, query, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("Matches found on initial and query photo: ", len(good), " out of ", len(matches))

    # Save photo of matches
    directory = "matches"
    filename = "comp_matches.jpg"
    cv2.imwrite(os.path.join(directory, filename), photo_match)

    # Print out matches
    cv2.imshow('Matches', photo_match)
    cv2.waitKey(10000)
